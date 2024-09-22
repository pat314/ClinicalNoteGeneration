import re
from typing import *

import pandas as pd


def term_replace(text: str, contractions: Dict) -> str:
    for k, v in contractions.items():
        text = re.sub(k, v, text, flags=re.IGNORECASE)
    return text


def postprocessing(df: pd.DataFrame,
                   dialogue_column: str,
                   FULL_CONTRACTIONS: Dict,
                   **kwargs) -> dict:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"

    def _postprocess(row):
        text = row[dialogue_column]
        _divisions = ["subjective", "objective_exam", "objective_results", "assessment_and_plan"]
        for division in _divisions:
            if division in row:
                row[division] = term_replace(row[division], FULL_CONTRACTIONS)
        row['note'] = term_replace(text, FULL_CONTRACTIONS)
        return row

    post_process_df = df.apply(_postprocess, axis=1)

    return {
        "df": post_process_df
    }
