import pandas as pd

from ConversationRoleNormalization import fix_role
from DialogueChunking import chunk_dialogues_from_df
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
    "hmm": ""

}

input_file_path = "TaskC-TrainingSet.csv"
output_file_path = "[Final]valid_chunked_output_file.csv"
dialogue_column = "dialogue"

def preprocessing(df: pd.DataFrame,
                  dialogue_column: str,
                  **kwargs) -> pd.DataFrame:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    df_cleaned = clean_spoken(df, dialogue_column=dialogue_column, **kwargs)
    df_punc = restore_punc(df_cleaned, dialogue_column='clean_dialogue', **kwargs)
    df_role = fix_role(df_punc, dialogue_column='restore_punctuation_dialogue', **kwargs)
    df_chunked = chunk_dialogues_from_df(df_role, dialogue_column='fixed_role_dialogue', **kwargs)

    return df_chunked

if __name__ == "__main__":
    # Read dataset
    df = pd.read_csv(input_file_path)

    # Preprocessing dataset
    df_processed = preprocessing(df, dialogue_column, CLEAN_CONTRACTIONS=CLEAN_CONTRACTIONS)

    # Save DataFrame after processing to .csv file (optional)
    df_processed.to_csv(output_file_path, index=False)
    print(f"Processed dataset saved to {output_file_path}")
