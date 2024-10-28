import re
from typing import List, Dict, Tuple, Any
import pandas as pd

from model_manager import get_spacy_model
from util import Sentence

def clean_words(text: str, CLEAN_CONTRACTIONS: dict) -> str:
    """
    Replaces contractions in a given text based on a specified dictionary.

    Args:
        text (str): The text to be cleaned by replacing contractions.
        CLEAN_CONTRACTIONS (dict): A dictionary where keys are contractions 
                                   and values are their expanded forms.

    Returns:
        str: The text with contractions replaced by their expanded forms.
    """
    # Iterate over each contraction and its expanded form in the dictionary
    for k, v in CLEAN_CONTRACTIONS.items():
        # Replace each contraction (key) in the text with its expanded form (value)
        text = text.replace(k, v)
        
    return text  # Return the cleaned text with all specified contractions replaced



def remove_consecutive_duplicates(text: str) -> str:
    """
    Removes consecutive duplicate words from a given text.

    Args:
        text (str): The text to be processed for consecutive duplicate words.

    Returns:
        str: The text with consecutive duplicate words removed.
    """
    # Use regular expression to match any word followed by itself, separated by optional non-word characters.
    # Replace these duplicate sequences with a single instance of the word.
    return re.sub(r'\b(\w+)(?:\W+\1\b)+', r'\1', text)



def cleaning_spoken_string(text: str, CLEAN_CONTRACTIONS: dict, PRONOUNS_CONTRACTIONS: dict) -> str:
    """
    Cleans a spoken text string by applying contraction replacement, removing consecutive duplicates, 
    and changing pronouns.

    Args:
        text (str): The spoken text to be cleaned.
        CLEAN_CONTRACTIONS (dict): Dictionary mapping contractions to their expanded forms.
        PRONOUNS_CONTRACTIONS (dict): Dictionary mapping pronouns to their preferred replacements.

    Returns:
        str: The cleaned and processed text.
    """
    # Step 1: Replace contractions in the text using the CLEAN_CONTRACTIONS dictionary
    text = clean_words(text, CLEAN_CONTRACTIONS)
    
    # Step 2: Remove consecutive duplicate words from the text
    text = remove_consecutive_duplicates(text)
    
    # Step 3: Replace specific pronouns based on the PRONOUNS_CONTRACTIONS dictionary
    text = change_pronouns(text, PRONOUNS_CONTRACTIONS)
    
    return text  # Return the fully cleaned and processed text


def fix_speech2text_error(text: str) -> str:
    """
    Fixes specific speech-to-text errors, particularly misinterpreted periods as dots in numeric contexts.

    Args:
        text (str): The text to be checked and corrected for speech-to-text errors.

    Returns:
        str: The corrected text with intended adjustments applied.
    """
    # Define punctuation characters to exclude from dot-only cases
    puncs = ',;!?'

    # Nested function to check if the text has only dot-based separators
    def _has_only_dots(txt):
        # Ensure no punctuation other than periods, and that periods appear with surrounding spaces
        return not any([punc in txt for punc in puncs]) and ' . ' in txt

    # Replace ' . ' with ' point ' if only dots are present without other punctuation
    if _has_only_dots(text):
        text = text.replace(' . ', ' point ')
    
    return text  # Return text with corrections for speech-to-text formatting errors










############################
# MAIN FUNCTION
############################
def clean_spoken(df: pd.DataFrame,
                 dialogue_column: str,
                 CLEAN_CONTRACTIONS: dict,
                 PRONOUNS_CONTRACTIONS: dict,
                 **kwargs) -> pd.DataFrame:
    """
    Cleans spoken dialogue data in a DataFrame by fixing speech-to-text errors and applying 
    contraction replacements and pronoun adjustments.

    Args:
        df (pd.DataFrame): DataFrame containing raw dialogue data.
        dialogue_column (str): Name of the column in `df` that holds the raw dialogue text.
        CLEAN_CONTRACTIONS (dict): Dictionary mapping contractions to their expanded forms.
        PRONOUNS_CONTRACTIONS (dict): Dictionary mapping pronouns to their preferred replacements.
        **kwargs: Additional arguments for future enhancements or modifications.

    Returns:
        pd.DataFrame: The DataFrame with an added column 'clean_dialogue' containing cleaned text.
    """
    # Ensure the specified dialogue column exists in the DataFrame
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    # Define a nested function to clean spoken text for each row
    def _clean_spoken(row):
        # Step 1: Fix any speech-to-text errors in the dialogue
        f_dialogue = fix_speech2text_error(row[dialogue_column])
        
        # Step 2: Clean the fixed dialogue by replacing contractions and adjusting pronouns
        row['clean_dialogue'] = cleaning_spoken_string(f_dialogue, CLEAN_CONTRACTIONS, PRONOUNS_CONTRACTIONS)
        
        return row  # Return the updated row with cleaned dialogue

    # Apply the cleaning function to each row of the DataFrame
    df = df.apply(_clean_spoken, axis=1)

    return df  # Return the updated DataFrame with cleaned dialogues



def change_pronouns(text: str,
                    PRONOUNS_CONTRACTIONS: dict) -> str:
    """
    Changes pronouns in a given text string based on a provided mapping.

    Args:
        text (str): The input text where pronouns need to be changed.
        PRONOUNS_CONTRACTIONS (Dict[str, str]): A dictionary mapping pronouns or contractions to their replacements.

    Returns:
        str: The modified text with pronouns replaced.
    """
    # Load the spaCy NLP model for natural language processing
    nlp = get_spacy_model()

    # Split the input text into individual sentences based on newline characters
    sentences = text.split("\n")
    modified_sentences = []  # Initialize a list to hold modified sentences

    for sentence in sentences:
        # Check if the sentence is spoken by the patient (identified by the "[patient]" tag)
        if sentence.startswith("[patient]"):
            # Remove the "[patient]" tag for further processing
            content = sentence[len("[patient]"):].strip()

            # Process the content with spaCy to analyze its structure
            spacy_doc = nlp(content)
            modified_content = []  # List to hold modified sentences for this patient's line

            # Iterate over individual sentences identified by spaCy
            for sent in spacy_doc.sents:
                sentence_text = sent.text.lower()  # Convert to lowercase for case-insensitive replacement

                # Replace pronouns using the provided dictionary of contractions
                for k, v in PRONOUNS_CONTRACTIONS.items():
                    sentence_text = re.sub(k, v, sentence_text)

                # Add the modified sentence to the list
                modified_content.append(sentence_text)

            # Combine modified sentences and re-add the "[patient]" tag
            modified_sentences.append(f"[patient] {' '.join(modified_content)}")
        else:
            # Append sentences that are not spoken by the patient without modification
            modified_sentences.append(sentence)

    # Join the modified sentences into a single string and return it
    return "\n".join(modified_sentences)

