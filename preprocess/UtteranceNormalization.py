import re
from typing import List, Dict, Tuple, Any
import pandas as pd

from model_manager import get_spacy_model
from util import Sentence

def clean_words(text: str, CLEAN_CONTRACTIONS: dict) -> str:
    for k, v in CLEAN_CONTRACTIONS.items():
        text = text.replace(k, v)
    return text


def remove_consecutive_duplicates(text: str) -> str:
    return re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', text)


def cleaning_spoken_string(text: str, CLEAN_CONTRACTIONS: dict, PRONOUNS_CONTRACTIONS: dict) -> str:
    text = clean_words(text, CLEAN_CONTRACTIONS)
    text = remove_consecutive_duplicates(text)
    text = change_pronouns(text, PRONOUNS_CONTRACTIONS)  # Change pronouns
    return text

def fix_speech2text_error(text: str) -> str:
    """ Fix speech-to-text error """
    puncs = ',;!?'
    def _has_only_dots(txt):
        return not any([punc in txt for punc in puncs]) and ' . ' in txt

    if _has_only_dots(text):
        text = text.replace(' . ', ' point ')
    return text

#hàm chính của file này
def clean_spoken(df: pd.DataFrame,
                 dialogue_column: str,
                 CLEAN_CONTRACTIONS: dict,
                 PRONOUNS_CONTRACTIONS: dict,
                 **kwargs) -> pd.DataFrame:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue,
        this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    def _clean_spoken(row):
        f_dialogue = fix_speech2text_error(row[dialogue_column])
        row['clean_dialogue'] = cleaning_spoken_string(f_dialogue, CLEAN_CONTRACTIONS, PRONOUNS_CONTRACTIONS)
        return row

    df = df.apply(_clean_spoken, axis=1)

    return df




def change_pronouns(text: str,
                    PRONOUNS_CONTRACTIONS: Dict[str, str]) -> str:
    """
    Changing pronouns in a given text string.

    Args:
        text (str): The input text where pronouns need to be changed.
        PRONOUNS_CONTRACTIONS (Dict[str, str]): A dictionary mapping pronouns or contractions to their replacements.

    Returns:
        str: The modified text with pronouns replaced.
    """
    # Load the spaCy NLP model
    nlp = get_spacy_model()

    # Convert text to lowercase to make it case-insensitive
    new_text = text.lower()

    # Replace pronouns using the provided dictionary
    for k, v in PRONOUNS_CONTRACTIONS.items():
        new_text = re.sub(k, v, new_text)

    # Reprocess the modified text using the spaCy model
    new_spacy_doc = nlp(new_text)

    # Return the modified text
    return new_spacy_doc.text
