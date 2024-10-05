"""
Modified code from taskC/src/retrieval/text_extracting_v2.py - previous paper
'no_BEAM_form_retrieval' for retrieving texts into sections, saved as json file in 'retrieval_result' row of Dataframe
'naive_extract_after_terms' taken from summarize phase for assessment_and_plan extracting
"""

import dataclasses
import json
import logging
import re
from copy import deepcopy as copy
from typing import List, Union, Dict

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


from util import RetrievalResult, split_sentence

log = logging.getLogger(__file__)

_MEANINGLESS_TEXT = '<--NOISY-->'
_EXTREME_LOW = -1e9


def normalize_to_usual_dialogue(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, r"\1:", text)
    return normalized_string


def remove_role(text: str) -> str:
    pattern = r"\[(\w+)\]"
    normalized_string = re.sub(pattern, "", text)
    return normalized_string


def cosine_matrix(a: np.array, b: np.array) -> np.array:
    """ Calculate cosine matrix of two given matrix"""
    # normalize each vector in a and b
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = a / a_norm
    b_normalized = b / b_norm

    # calculate cosine similarity between each vector in a and b
    similarity = np.matmul(a_normalized, b_normalized.T)
    return similarity


def retrieval_no_beam_mul_query(texts: List[str],
                                query: Union[str, List[str]],
                                min_score: float = None,
                                max_group: int = None,
                                max_sentence: int = None,
                                max_word: int = None,
                                normalize: bool = False,
                                do_remove_role: bool = False,
                                model: Union[str, SentenceTransformer] = 'paraphrase-MiniLM-L6-v2',
                                verbose: bool = False,
                                intra_group_max_pooling: bool = False,
                                simple_sent_split: bool = False,
                                use_cosine: bool = False,
                                drop_noisy_sentence: bool = False,
                                device: str = 'auto',
                                **kwargs
                                ) -> RetrievalResult:
    """
    Retrieval related base-unit depends on query-driven ranking
    """
    base_unit = [text for text in texts if len(text.strip()) > 5]
    raw_base_unit = [text for text in texts if len(text.strip()) > 5]
    if normalize:
        base_unit = [normalize_to_usual_dialogue(d) for d in texts]
    if do_remove_role:
        base_unit = [remove_role(d) for d in texts]
    if isinstance(model, str):
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
    
    device = 'cpu'

    des_vec = model.encode(query, device=device,
                           show_progress_bar=verbose,
                           convert_to_numpy=True)
    if des_vec.ndim == 1:
        des_vec = des_vec.reshape(1, -1)

    if not intra_group_max_pooling:
        base_embeds = model.encode(base_unit, device=device,
                                   show_progress_bar=verbose,
                                   convert_to_numpy=True)
        if base_embeds.ndim == 1:
            base_embeds = des_vec.reshape(1, -1)

        if not use_cosine:
            raw_markovw = np.matmul(des_vec, base_embeds.T)  # q x dim * dim x k = q x k
        else:
            raw_markovw = cosine_matrix(des_vec, base_embeds)  # q x dim * dim x k = q x k

        markovw = np.amax(raw_markovw, axis=0).reshape(-1)  # k,
        detailed_texts = base_unit  # num_base_unit
        detailed_scores = raw_markovw.tolist()  # q x k
    else:
        _sentences = []
        detailed_texts = []
        for i, group in enumerate(base_unit):
            sentences = split_sentence(group,
                                       mark_noisy_sentence=drop_noisy_sentence,
                                       marker=_MEANINGLESS_TEXT,
                                       simple=simple_sent_split)
            _sentences.extend([(sent, i) for sent in sentences])
            detailed_texts.append(sentences)

        _sentence_texts = [sent[0] for sent in _sentences]
        _idx = [sent[1] for sent in _sentences]
        base_embeds = model.encode(_sentence_texts,
                                   device=device,
                                   show_progress_bar=verbose,
                                   convert_to_numpy=True)
        meaningless_idxs = [_sent[:len(_MEANINGLESS_TEXT)] == _MEANINGLESS_TEXT
                            for _sent in _sentence_texts]

        if base_embeds.ndim == 1:
            base_embeds = base_embeds.reshape(1, -1)
        if not use_cosine:
            raw_markovw = np.matmul(des_vec, base_embeds.T)  # q x dim * dim x k = q x k
        else:
            raw_markovw = cosine_matrix(des_vec, base_embeds)  # q x dim * dim x k = q x k
        markovw = []

        # detailed_texts shape: num_base_unit x sentence
        detailed_scores = []
        # set _EXTREME_LOW to padded sentences
        raw_markovw[:, meaningless_idxs] = _EXTREME_LOW
        assert len(set(_idx)) == len(raw_base_unit), f"There is an empty input! {set(_idx)}"

        for group_idx in range(len(base_unit)):
            # get meaningful sentence indexes
            is_in_group = np.array(_idx) == group_idx
            sent_idx = np.argwhere(is_in_group).reshape(-1)

            intra_group_scores = raw_markovw[:, sent_idx]
            detailed_scores.append(intra_group_scores.tolist())
            group_score = np.amax(intra_group_scores)
            markovw.append(group_score)

        # detailed_scores shape: num_base_unit x query x sentence
        markovw = np.array(markovw)

    l = list(zip(raw_base_unit, markovw, range(len(raw_base_unit))))

    assert max_group is not None or \
           max_sentence is not None or \
           max_word is not None or \
           min_score is not None, "Either 'max_group', 'max_sentence', 'min_score'" \
                                  ", or 'max_word' must not be None"
    if min_score is not None:
        higher_score_idx = np.argwhere(markovw >= min_score).reshape(-1)
        retrieval_results = [l[i] for i in higher_score_idx]
    else:
        l.sort(key=lambda x: -x[1])
        if max_group is None:
            n_sent = 0
            n_word = 0
            for idx, (txt, _, _) in enumerate(l):
                n_word += len(txt.split())
                n_sent += len(split_sentence(txt,
                                             mark_noisy_sentence=drop_noisy_sentence,
                                             marker=_MEANINGLESS_TEXT,
                                             simple=simple_sent_split))
                max_group = idx + 1
                if (max_word is not None and n_word >= max_word) \
                        or (max_sentence is not None and n_sent >= max_sentence):
                    break
        retrieval_results = l[:max_group]

    retrieval_results.sort(key=lambda x: [x[2]])
    idxs = [_d[-1] for _d in retrieval_results]
    text_result = [texts[_i] for _i in idxs]
    scores = [float(_d[1]) for _d in retrieval_results]

    return RetrievalResult(text_result, scores,
                           detailed_texts=detailed_texts,
                           detailed_scores=detailed_scores)


def strict_matching(texts: List[str],
                    query: Union[str, List[str]],
                    span_length: int = 3,
                    left: int = 1,
                    right: int = 1,
                    ignore_case: bool = True,
                    **kwargs) -> RetrievalResult:
    """
    Terms-finding matching
    if texts = [a, b, c, d, e, f, g, h], and d, f contains any word in query
    span_length: if span_length is 3, it will return [c,d,e,f,g] (concat [c,d,e] and [e,f,g])
    """
    base_texts = copy(texts)
    if isinstance(query, str):
        query = [query]
    if ignore_case:
        query = [w.lower() for w in query]
        base_texts = [w.lower() for w in texts]
    idxs = []
    if left is None or right is None and span_length is not None:
        left, right = int(span_length / 2), int(span_length / 2)
    detailed_scores = []
    for idx, txt in enumerate(base_texts):
        if any([w in txt for w in query]):
            idxs.extend(list(range(idx - left, idx + right + 1)))
            detailed_scores.append(1)
        else:
            detailed_scores.append(0)
    idxs = list(set(idxs))
    idxs.sort()
    idxs = [idx for idx in idxs if idx in range(len(texts))]
    return RetrievalResult(
        texts=[texts[idx] for idx in idxs],
        scores=[1, ] * len(idxs),
        detailed_texts=base_texts,
        detailed_scores=detailed_scores
    )


def no_BEAM_form_retrieval(df: pd.DataFrame,
                           seeding_form: List[Dict],
                           dialogue_column: str,
                           Hverbose: bool = True,
                           **kwargs) -> Dict:
    from tqdm import tqdm
    assert dialogue_column in df.columns, f" Column '{dialogue_column}' not found in the given dataframe!"
    if Hverbose:
        Hverbose = tqdm(range(len(df.group_dialogue)), "Extracting supporting texts....")

    def _find_related(row: dict, _verbose: tqdm = None) -> dict:
        rs = []
        for each_section_des in seeding_form:
            each_section_des = copy(each_section_des)
            
            if 'retrieval' in each_section_des:
                assert 'query' in each_section_des['retrieval'], \
                    "if seeding_form contains 'retrieval', " \
                    "it must contain 'retrieval':'query' variable!"

                use_group: Union[bool, int] = False
                if 'use_group' in each_section_des['retrieval']:
                    use_group = each_section_des['retrieval']['use_group']
                text = row[dialogue_column]
                if use_group == -1:
                    texts = text.replace('\n<GROUP>\n', '.').split('.')
                    texts = [text for text in texts if text.strip()]
                else:
                    if "\n<GROUP>\n" not in text and use_group:
                        log.warning(
                            f'"use_group" is True but not found <GROUP> in the given column {dialogue_column}!'
                            f'"use_group" will be False for this sample.')
                    if "\n<GROUP>\n" in text and use_group:
                        texts = text.split("\n<GROUP>\n")
                    else:
                        texts = text.replace('\n<GROUP>\n', '\n').split('\n')
                        texts = [text for text in texts if text.strip()]

                retrieval_kwargs = copy(each_section_des['retrieval'])

                

                if 'method' in retrieval_kwargs:
                    method = globals()[retrieval_kwargs['method']]
                    del retrieval_kwargs['method']
                    retrieval_result = method(texts, **retrieval_kwargs)
                else:
                    retrieval_result = retrieval_no_beam_mul_query(texts,
                                                                   **retrieval_kwargs)
                    
                each_section_des['retrieval_result'] = dataclasses.asdict(retrieval_result)
                
            
            if 'assessment_and_plan' in each_section_des['division']:
                retrieval_kwargs = copy(each_section_des['summarizer'])
                if 'method' in retrieval_kwargs:
                    method = globals()[retrieval_kwargs['method']]
                    del retrieval_kwargs['method']

                    # demo
                    text = row[dialogue_column]
                    filtered_kwargs = {k: v for k, v in retrieval_kwargs.items() if k in ['exeptions', 'column']}
                    retrieval_result = method(text, **filtered_kwargs)

                    each_section_des['retrieval_result'] = dataclasses.asdict(retrieval_result)
                    # demo
                

            rs.append(each_section_des)
        row['retrieval_text'] = json.dumps(rs, ensure_ascii=False, indent=0)
        if isinstance(_verbose, tqdm):
            _verbose.update(1)
        return row

    df = df.apply(lambda r: _find_related(r, Hverbose), axis=1)

    return {
        "df": df
    }


def naive_extract_after_terms(text: str,
                              column: str,
                              suffix: str = None,
                              exceptions: List[str] = None,
                              ) -> RetrievalResult:
    
    substring2 = "-year-old"
    ext1, ext2 = "", ""
    r1, r2 = "", ""
    ""
    substring1 = [
        'assessment', 'my impression', ' plan'
    ]

    for s in substring1:

        if text.find(s) != -1:
            index = text.find(s)
            role_index = text.rfind('[', 0, index)
            result = text[role_index:].strip()
            if exceptions:
                if any([result[:len(i)] == i for i in exceptions]):
                    continue
            result = "\n".join(result.split("\n<GROUP>\n"))
            r1 = result
            # result = re.sub("(\\[doctor\\]|\\[patient\\])", "", result)
            ext1 = result

    new_text = text.split(".")
    for sent in new_text:
        if re.search(substring2, sent):
            result = sent
            result = "\n".join(result.split("\n<GROUP>\n"))
            r2 = result
            # result = re.sub("(\\[doctor\\] |\\[patient\\] |\n<GROUP>\n)", "", result)
            ext2 = result

    ext = ext2 + "\n" + ext1
    if suffix is not None:
        ext += suffix
    ext_rr = RetrievalResult(ext.split('\n'), [], [], [])
    return ext_rr