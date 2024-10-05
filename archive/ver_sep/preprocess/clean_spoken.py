import re
from typing import List, Dict, Tuple, Any

import pandas as pd

from ..model_manager import get_spacy_model
from ..util import Sentence


def clean_words(text: str, CLEAN_CONTRACTIONS: dict) -> str:
    for k, v in CLEAN_CONTRACTIONS.items():
        text = text.replace(k, v)
    return text


def remove_consecutive_duplicates(text: str) -> str:
    return re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', text)


def cleaning_spoken_string(text: str, CLEAN_CONTRACTIONS: dict) -> str:
    text = clean_words(text, CLEAN_CONTRACTIONS)
    text = remove_consecutive_duplicates(text)
    return text

def fix_speech2text_error(text: str) -> str:
    """ Fix speech-to-text error """
    puncs = ',;!?'
    def _has_only_dots(txt):
        return not any([punc in txt for punc in puncs]) and ' . ' in txt

    if _has_only_dots(text):
        text = text.replace(' . ', ' point ')
    return text


def clean_spoken(df: pd.DataFrame,
                 dialogue_column: str,
                 CLEAN_CONTRACTIONS: dict,
                 **kwargs) -> pd.DataFrame:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue,
        this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    def _clean_spoken(row):
        f_dialogue = fix_speech2text_error(row[dialogue_column])
        row['clean_dialogue'] = cleaning_spoken_string(f_dialogue, CLEAN_CONTRACTIONS)
        return row

    df = df.apply(_clean_spoken, axis=1)

    return df




def change_pronouns(dialogue: List[Sentence],
                    PRONOUNS_CONTRACTIONS: Dict[str, str]):
    """
    Changing pronouns for sentences of patient.
    """
    nlp = get_spacy_model()
    for i in range(len(dialogue)):
        if dialogue[i].role == "[patient]":
            sent = dialogue[i].spacy_doc.text.lower()
            new_sent = sent
            for k, v in PRONOUNS_CONTRACTIONS.items():
                new_sent = re.sub(k, v, new_sent)
            if new_sent != sent:
                dialogue[i].spacy_doc = nlp(new_sent)

    return dialogue
