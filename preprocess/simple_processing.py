import pandas as pd

from ConversationRoleNormalization import fix_role
from PunctuationRestoration import restore_punc
from UtteranceNormalization import clean_spoken

CLEAN_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "I'm": "I am",
    "isn't": "is not",
    "aren't": "are not",
    "it's": "it is",
    "I'll": "I will",
    "we'll": "we will",
    "don't": "do not",
    "didn't": "did not",
    "ca n't": "can not",
    "wo not": "will not",
    "n't": "not",
    "could've": "could have",
    "i'm": "i am",
    "i've": "i have",
    "might've": "might have",
    "must've": "must have",
    "shan't": "shall not",
    "should've": "should have",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we're": "we are",
    "we've": "we have",
    "what're": "what are",
    "what've": "what have",
    "who're": "who are",
    "who've": "who have",
    "would've": "would have",
    "you're": "you are",
    "you've": "you have",
    "gonna": "going to",
    "gon'na": "going to",
    "gon na": "going to",
    "gotta": "got to",
    "wanna": "want to",
    "wan'na": "want to",
    "wan na": "want to",
    "hafta": "have to",
    "hadta": "had to",
    "shoulda": "should have",
    "woulda": "would have",
    "coulda": "could have",
    "mighta": "might have",
    "musta": "must have",
    "oughta": "ought to",
    "dont": "do not",
    "doesnt": "does not",
    "uh": "",
    "um": "",
    "er": "",
    "ah": "",
    "well": "",
    "you know": "",
    "I mean": "",
    "like": "",
    "hmm": "",
}

PRONOUNS_CONTRACTIONS = {
      ", you know ?": ",",
      ", you know ,": ",",
      ", you know,": ",",
      "you know ,": "",
      "you know,": "",
    r"\bi have\b": "the patient has",
    r"\bi've\b": "the patient has",
    r"\bi am\b": "the patient is",
    r"\bi'm\b": "the patient is",
    r"\bi'd\b": "the patient would",
    r"\bi\b": "the patient",
    r"\bme\b": "the patient",
    r"\bmy\b": "his / her"
    }

input_file_path = "TaskC-TrainingSet.csv"
output_file_path = "valid_fix_role_output_file.csv"
dialogue_column = "dialogue"

def preprocessing(df: pd.DataFrame,
                  dialogue_column: str,
                  **kwargs) -> pd.DataFrame:
    """
    Preprocesses dialogue data by applying cleaning, punctuation restoration, and role fixing.

    Args:
        df (pd.DataFrame): DataFrame containing raw dialogue data.
        dialogue_column (str): Name of the column in `df` that holds the raw dialogue text.
        **kwargs: Additional arguments for cleaning and punctuation restoration options.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with cleaned dialogue, restored punctuation, 
                      and corrected roles in dialogue text.
    """
    # Ensure the specified dialogue column exists in the DataFrame
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    # Step 1: Clean the spoken dialogue by applying text cleaning functions
    df_cleaned = clean_spoken(df, dialogue_column=dialogue_column, **kwargs)
    
    # Step 2: Restore punctuation in the cleaned dialogue
    df_punc = restore_punc(df_cleaned, dialogue_column='clean_dialogue', **kwargs)
    
    # Step 3: Fix the roles in the dialogue after punctuation restoration
    df_role = fix_role(df_punc, dialogue_column='restore_punctuation_dialogue', **kwargs)

    return df_role  # Return the fully processed DataFrame

if __name__ == "__main__":
    # Read dataset from a CSV file
    df = pd.read_csv(input_file_path)

    # Preprocess the dataset using specified contraction and pronoun options
    df_processed = preprocessing(df, dialogue_column, 
                                 CLEAN_CONTRACTIONS=CLEAN_CONTRACTIONS, 
                                 PRONOUNS_CONTRACTIONS=PRONOUNS_CONTRACTIONS)

    # Save the processed DataFrame to a CSV file (optional)
    df_processed.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}")

