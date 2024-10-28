import pandas as pd
from tqdm import tqdm

from model_manager import get_punc_restore_model
from util import init_utterance_text_list

def has_punc(text: str) -> bool:
    """
    Checks if a given text contains any punctuation marks: ?, ;, or !.

    Args:
        text (str): The text to be checked for punctuation.

    Returns:
        bool: True if any specified punctuation mark is found in the text, False otherwise.
    """
    # Check each character in the string '?;!' to see if it appears in `text`
    return any([punc in text for punc in '?,;!'])



def restore_punctuation_row(row, task: str, dialogue_column: str,
                            enforce_punc: bool = True, verbose: tqdm = None):
    """
    Restores punctuation in dialogue text for a specific DataFrame row, based on the task type.

    Args:
        row (pd.Series): A row from a DataFrame containing dialogue text.
        task (str): Task type, used to determine how the role and text are split.
        dialogue_column (str): The name of the column containing dialogue text to process.
        enforce_punc (bool, optional): If True, punctuation restoration will be enforced
                                       even if punctuation is already present. Defaults to True.
        verbose (tqdm, optional): A tqdm progress bar for tracking progress, if provided.

    Returns:
        pd.Series: The modified row with a new column ('restore_punctuation_dialogue') 
                   containing the dialogue with restored punctuation.
    """

    # Retrieve the dialogue text from the specified column
    cleaned_spoken = row[dialogue_column]
    
    # Check if punctuation is missing, or if punctuation enforcement is enabled
    if not has_punc(cleaned_spoken) or enforce_punc:
        # Split text into list of (role, utterance) pairs according to the task type
        role_w_utterance = init_utterance_text_list(text=cleaned_spoken, task=task)
        
        # Define a function for restoring punctuation in an utterance
        restore_punctuation = lambda text: get_punc_restore_model().restore_punctuation(text)
        
        # Set task-specific separator between role and utterance
        task_splitter = ': ' if task != "C" else " "

        # Restore punctuation for each utterance and combine it with the role
        restored_texts = []
        for role, utterance in role_w_utterance:
            restored_texts.append(f'{role}{task_splitter}{restore_punctuation(utterance)}')

        # Join restored texts into a single string with newlines between entries
        row['restore_punctuation_dialogue'] = '\n'.join(restored_texts)
    else:
        # If punctuation is present and not enforced, keep the original text
        row['restore_punctuation_dialogue'] = cleaned_spoken

    # Update progress bar if provided
    if isinstance(verbose, tqdm):
        verbose.update(1)

    return row  # Return the modified row with punctuation restored in the specified column



def restore_punc(df: pd.DataFrame,
                 dialogue_column: str,
                 enforce_punc: bool = True,
                 verbose=True,
                 **kwargs) -> pd.DataFrame:
    """
    Restores punctuation for dialogue data in a specified DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame containing raw dialogue data.
        dialogue_column (str): Column name with the dialogue data to be processed.
        enforce_punc (bool, optional): If True, punctuation restoration is applied 
                                       even if punctuation is already present. Defaults to True.
        verbose (bool, optional): If True, displays a tqdm progress bar for tracking progress.
                                  Defaults to True.
        **kwargs: Additional keyword arguments for flexibility in future enhancements.

    Returns:
        pd.DataFrame: A DataFrame with an additional column containing dialogues with restored punctuation.
    """
    # Ensure the specified dialogue column exists in the DataFrame
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    # Initialize tqdm progress bar if verbose is enabled
    if verbose:
        from tqdm import tqdm
        verbose = tqdm(range(len(df[dialogue_column])), desc="Restoring punctuation...")

    # Apply `restore_punctuation_row` to each row to restore punctuation in the dialogue column
    df = df.apply(lambda row: restore_punctuation_row(row, "C",
                                                      enforce_punc=enforce_punc,
                                                      verbose=verbose,
                                                      dialogue_column=dialogue_column),
                  axis=1)

    return df  # Return DataFrame with restored punctuation in dialogue data
