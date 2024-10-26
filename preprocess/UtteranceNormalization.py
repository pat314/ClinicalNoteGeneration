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
                    PRONOUNS_CONTRACTIONS: dict) -> str:
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

    # Split text into sentences based on newline
    sentences = text.split("\n")
    modified_sentences = []

    for sentence in sentences:
        # Check if the sentence is spoken by the patient
        if sentence.startswith("[patient]"):
            # Remove "[patient]" tag for processing
            content = sentence[len("[patient]"):].strip()

            # Process content with spaCy to split into individual sentence components
            spacy_doc = nlp(content)
            modified_content = []

            # Iterate over individual sentences in the patient's line
            for sent in spacy_doc.sents:
                sentence_text = sent.text.lower()  # Convert to lowercase for case-insensitive replacement

                # Replace pronouns using the provided dictionary
                for k, v in PRONOUNS_CONTRACTIONS.items():
                    sentence_text = re.sub(k, v, sentence_text)

                modified_content.append(sentence_text)

            # Combine modified content and re-add "[patient]" tag
            modified_sentences.append(f"[patient] {' '.join(modified_content)}")
        else:
            # Append sentences that are not spoken by the patient without modification
            modified_sentences.append(sentence)

    # Join modified sentences into a single text
    return "\n".join(modified_sentences)
