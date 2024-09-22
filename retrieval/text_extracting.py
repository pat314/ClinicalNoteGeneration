import json
from copy import copy
from typing import List, Union, Iterable, Tuple, Any, Dict

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from ..model_manager import get_sbert_model, auto_detect_device


def extract_supporting_texts(texts: List[str],
                             prompt_text: Union[str, List[str]],
                             model: Union[SentenceTransformer, str] = 'paraphrase-MiniLM-L6-v2',
                             device: str = 'cpu',
                             splitter: str = '\n',
                             min_gram: int = None,
                             min_output_length: int = 150,
                             beam_size: int = 20,
                             max_iter: int = None,
                             max_output_length: int = 512,
                             return_scores: bool = False,
                             verbose: bool = False) -> Union[List[str], Tuple[List[str], List[float]]]:
    """
    Return a RANKED list which may be related to the prompt_text
    model: SentenceTransformer or its model name on huggingface
    device: device running this model (cuda:x or cpu), default: auto detech and prefer GPU,
        only activate when model is a string.
    splitter: the character used for marking concatenation in the output
    beam_size: beam size in beam search expanding algorithm
    max_iter: number of running iteration (none means no limitation)
    max_output_length: the max size (in word) of each chunk.
        Note that this condition won't check the given input
    return_scores: return the corresponding scores (optional)
    """
    if isinstance(model, str):
        model = get_sbert_model(model)
    if device == 'auto' and not next(model.parameters()).is_cuda:
        device = auto_detect_device(model)
    else:
        device = model.device
    n_iter = 0 if max_iter is not None else None
    base_unit: List[str] = copy(texts)
    score_queue: List[int] = []
    index_queue: List[List[int]] = []
    des_vec = model.encode(prompt_text,
                           show_progress_bar=verbose,
                           convert_to_numpy=True, device=device)
    if des_vec.ndim == 1:
        des_vec = des_vec.reshape(1, -1)

    seen_steps = []

    def _get_text(idx_list: Iterable[int]):
        return splitter.join([base_unit[_id] for _id in idx_list])

    _whole_text = _get_text(range(len(base_unit)))

    if min_output_length is not None:
        if min_output_length >= len(_whole_text.split()):
            if verbose:
                print(
                    f"WARM: 'min_output_length' is set to {min_output_length}, while the input length is {len(_whole_text.split())}")
                print(f"WARM: return the exacted input!")
            if return_scores:
                return [_whole_text], [1.0]
            return [_whole_text]
        if min_gram is not None:
            min_gram = None
            if verbose:
                print(f"WARM: both 'min_gram'  and 'min_output_length' are set. Please select EITHER!")

    def _update_candidate(candidate_idxs_: List[List[int]]):
        if len(candidate_idxs_) == 0:
            return
        candidates = []
        for idx_list in candidate_idxs_:
            candidates.append(_get_text(idx_list))
        embeddings = model.encode(candidates, show_progress_bar=verbose, convert_to_numpy=True, device=device)
        sim_mtx = np.matmul(des_vec, embeddings.T)
        scores = np.amax(sim_mtx, axis=0)
        for idx, score in zip(candidate_idxs_, scores):
            score_queue.append(score)
            index_queue.append(idx)

    def _prune_queue(index_queue, score_queue):
        if len(index_queue) > beam_size:
            # sort by score in descending order
            _idx = np.flip(np.argsort(score_queue))
            score_queue = [score_queue[_id] for _id in _idx]
            index_queue = [index_queue[_id] for _id in _idx]
            score_queue = score_queue[:beam_size]
            index_queue = index_queue[:beam_size]
        return index_queue, score_queue

    # seeding BEAM iter
    if min_gram is not None:
        _update_candidate([list(range(_i, _i + min_gram)) for _i in range(len(base_unit) - min_gram)])
    else:
        initial_candidates = []
        _base_unit_lengths = [len(u.split()) for u in base_unit]
        for i in range(len(base_unit)):
            for j in range(i + 1, len(base_unit)):
                _text_length = sum(_base_unit_lengths[i:j])
                if _text_length >= min_output_length:
                    initial_candidates.append(list(range(i, j)))
                    break
        _update_candidate(initial_candidates)

    index_queue, score_queue = _prune_queue(index_queue, score_queue)
    previous_iter = None

    def _is_stop():
        if n_iter is not None and n_iter > max_iter:
            if verbose:
                print("Stop by max iter reach!")
            return True
        return False

    def _try_to_expand():
        """
        Given [left+1...right-1]. Try to expand to:
        [left,] + [left+1...right-1]
        [left,] + [left+1...right-1] + [right,]
        left+1...right-1] + [right,]
        [left-1,] + [left+1...right-1] # gap
        [left-1,] + [left+1...right-1] + [right+1,] # gap
        [left+1...right-1] + [right+1,] # gap
        """

        def _check_can_expand(idx_list: List[int]):
            idx_list.sort()
            if idx_list[0] == 0 and idx_list[-1] == len(base_unit) - 1:
                if verbose:
                    print(f"{idx_list} can not expand due to out of range {range(len(base_unit))}!")
                return False
            txt = _get_text(idx_list)
            if txt.split().__len__() >= max_output_length:
                if verbose:
                    print(
                        f"{idx_list} can not expand due to length limit of {max_output_length} (having {txt.split().__len__()} words)!")
                return False
            if idx_list in seen_steps:
                if verbose:
                    print(f"{idx_list} found no better candidate!")
                return False
            return True

        can_expand_index_list = None
        for can_expand_index_list in index_queue:
            if _check_can_expand(can_expand_index_list):
                break
        if can_expand_index_list is None:
            return None, None

        left, right = can_expand_index_list[0] - 1, can_expand_index_list[-1] + 1
        candidate_idxs = [
            [left, ] + can_expand_index_list,
            [left, ] + can_expand_index_list + [right, ],
            can_expand_index_list + [right, ],
            [left - 1, ] + can_expand_index_list,
            [left - 1, ] + can_expand_index_list + [right + 1, ],
            can_expand_index_list + [right + 1, ],
        ]
        # filter out of range index
        candidate_idxs = [[_id
                           for _id in sub_idx_list if _id in range(len(base_unit))]
                          for sub_idx_list in candidate_idxs]
        # filter duplicate
        string_index = ['_'.join([str(y) for y in sub_idx_list]) for sub_idx_list in candidate_idxs]
        string_index = list(set(string_index))
        candidate_idxs = [[int(x) for x in txt_sub.split('_')] for txt_sub in string_index]

        # filter too long candidate
        candidate_idxs = [sub_idx_list for sub_idx_list in candidate_idxs
                          if len(_get_text(sub_idx_list).split()) <= max_output_length]

        return copy(can_expand_index_list), candidate_idxs

    if verbose:
        import tqdm
        if n_iter is not None:
            process = tqdm.tqdm(range(n_iter), "running BEAM SEARCH")
        else:
            process = tqdm.tqdm(iterable=None, desc="running BEAM SEARCH")

    while not _is_stop():
        if n_iter is not None:
            n_iter += 1
        if verbose:
            process.update(1)
        current_iter, candidate_idxs = _try_to_expand()
        if current_iter == previous_iter:
            if verbose:
                print(f"Stop by current iter same as the previous: {current_iter}!")
            break
        if candidate_idxs is None:
            if verbose:
                print(f"Stop by returned candidate_idxs is None!")
            break
        _update_candidate(candidate_idxs)
        index_queue, score_queue = _prune_queue(index_queue, score_queue)
        previous_iter = current_iter
        seen_steps.append(current_iter)
    if verbose:
        print('End after: ', n_iter)
        print('End with index', index_queue)
        print('End with scores', score_queue)
    return_texts = [_get_text(idx_list) for idx_list in index_queue]
    if return_scores:
        return return_texts, np.array(score_queue).tolist()
    return return_texts


def form_retrieval(dialogue_list: Any,
                   df: pd.DataFrame,
                   seeding_form: List[Tuple[str]],
                   dialogue_column: str,
                   Hverbose: bool = True,
                   **kwargs) -> Dict:
    from tqdm import tqdm
    assert dialogue_column in df.columns, f" Column '{dialogue_column}' not found in the given dataframe!"
    if 'return_scores' in kwargs:
        del kwargs['return_scores']
    if Hverbose:
        Hverbose = tqdm(range(len(df.dialogue)), "Extracting supporting texts....")

    def _find_related(row: dict, _verbose: tqdm = None) -> dict:
        text = row[dialogue_column]
        if "\n<GROUP>\n" in text:
            texts = text.split("\n<GROUP>\n")
        else:
            texts = text.split('\n')
        rs = []
        for section, query, prompt_decoder in seeding_form:
            support_texts, scores = extract_supporting_texts(texts,
                                                             prompt_text=query,
                                                             return_scores=True,
                                                             **kwargs)
            rs.append((section, query, prompt_decoder, support_texts, scores))
        row['retrieval_text'] = json.dumps(rs, ensure_ascii=False, indent=0)
        if isinstance(_verbose, tqdm):
            _verbose.update(1)
        return row

    df = df.apply(lambda r: _find_related(r, Hverbose), axis=1)

    return {
        "dialogue_list": dialogue_list, "df": df
    }
