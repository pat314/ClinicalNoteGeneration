from typing import List
import pandas as pd

from util import Utterance, init_from_str_dialogue


def fix_role_dialogue(dialogue: List[Utterance]):
    """
    Corrects any errors in the roles assigned to each dialogue utterance.

    Args:
        dialogue (List[Utterance]): A list of utterance objects, where each utterance has a role
                                    (e.g., '[doctor]' or other) and associated text.

    Returns:
        List[Utterance]: The dialogue list with corrected roles, if needed.
    """

    # Check the role for each utterance to identify the non-doctor role
    for utterance in dialogue:
        if utterance.role != "[doctor]":
            speaker2_role = utterance.role  # Stores the other role (non-doctor) for future reference

    # Initialize counters for questions asked by each role
    questions_doctor = 0
    questions_speaker2 = 0

    # Count the number of questions each role has asked
    for utterance in dialogue:
        if utterance.role == "[doctor]":
            questions_doctor += utterance.spacy_doc.text.count("?")
        else:
            questions_speaker2 += utterance.spacy_doc.text.count("?")

    # If the non-doctor role has asked more questions, swap roles to fix assignment errors
    if questions_doctor < questions_speaker2:
        for utterance in dialogue:
            if utterance.role == "[doctor]":
                utterance.role = speaker2_role  # Reassigns doctor utterances to the other role
            else:
                utterance.role = "[doctor]"  # Reassigns the other role utterances to doctor

    return dialogue  # Returns the corrected dialogue list



def fix_role(df: pd.DataFrame,
             dialogue_column: str,
             exceptions: List[List[str]] = None,
             **kwargs) -> pd.DataFrame:
    """
    Corrects roles in dialogue samples contained in a specific column of a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing dialogue samples.
        dialogue_column (str): Name of the column in `df` that holds dialogue text to be processed.
        exceptions (List[List[str]], optional): A list of exception rules for initializing dialogues.
        **kwargs: Additional keyword arguments for flexibility in future enhancements.

    Returns:
        pd.DataFrame: A modified DataFrame with an additional column ('fixed_role_dialogue')
                      where roles in the dialogue samples are corrected.
    """
    # Check that the specified dialogue column exists in the DataFrame
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the input DataFrame!"

    # Inner function to process each row by fixing dialogue roles
    def _fix_role_each(row):
        # Initialize a list of utterances from the dialogue text in the row, taking exceptions into account
        utterance_list = init_from_str_dialogue(row[dialogue_column],
                                                task='C',
                                                exceptions=exceptions)
        
        # Apply the `fix_role_dialogue` function to correct roles within the utterance list
        utterance_list = fix_role_dialogue(utterance_list)
        
        # Convert the fixed list of utterances back to a formatted dialogue string
        new_dialogue = '\n'.join(f'{u.role}{str(u.spacy_doc)}' for u in utterance_list)
        
        # Add the new dialogue text with fixed roles to a new column in the DataFrame row
        row['fixed_role_dialogue'] = new_dialogue
        return row

    # Apply the `_fix_role_each` function to each row in the DataFrame
    df = df.apply(_fix_role_each, axis=1)

    return df  # Return the modified DataFrame with the additional 'fixed_role_dialogue' column

