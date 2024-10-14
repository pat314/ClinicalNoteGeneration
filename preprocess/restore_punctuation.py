import pandas as pd
from tqdm import tqdm

from model_manager import get_punc_restore_model
from util import init_utterance_text_list


def has_punc(text: str) -> bool:
    return any([punc in text for punc in '?,;!'])


def restore_punctuation_row(row, task: str, dialogue_column: str,
                            enforce_punc: bool = True, verbose: tqdm = None):
    cleaned_spoken = row[dialogue_column]
    if not has_punc(cleaned_spoken) or enforce_punc:
        role_w_utterance = init_utterance_text_list(text=cleaned_spoken, task=task)
        restore_punctuation = lambda text: get_punc_restore_model().restore_punctuation(text)
        task_splitter = ': ' if task != "C" else " "

        restored_texts = []
        for role, utterance in role_w_utterance:
            restored_texts.append(f'{role}{task_splitter}{restore_punctuation(utterance)}')

        row['restore_punctuation_dialogue'] = '\n'.join(restored_texts)
    else:
        row['restore_punctuation_dialogue'] = cleaned_spoken
    if isinstance(verbose, tqdm):
        verbose.update(1)
    return row


def restore_punc(df: pd.DataFrame,
                 dialogue_column: str,
                 enforce_punc: bool = True,
                 verbose=True,
                 **kwargs) -> pd.DataFrame:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return
        a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    if verbose:
        from tqdm import tqdm
        verbose = tqdm(range(len(df.dialogue)), "Restoring punctuation....")
    df = df.apply(lambda row: restore_punctuation_row(row, "C",
                                                      enforce_punc=enforce_punc,
                                                      verbose=verbose,
                                                      dialogue_column=dialogue_column),
                  axis=1)

    return df
