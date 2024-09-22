from typing import List

import pandas as pd

from ..util import Utterance, init_from_str_dialogue


def fix_role_dialogue(dialogue: List[Utterance]):
    """
    Check for role error and fix if there is error
    """
    for utterance in dialogue:
        if utterance.role != "[doctor]":
            speaker2_role = utterance.role

    questions_doctor = 0
    questions_speaker2 = 0

    for utterance in dialogue:
        if utterance.role == "[doctor]":
            questions_doctor += utterance.spacy_doc.text.count("?")
        else:
            questions_speaker2 += utterance.spacy_doc.text.count("?")

    if questions_doctor < questions_speaker2:
        for utterance in dialogue:
            if utterance.role == "[doctor]":
                utterance.role = speaker2_role
            else:
                utterance.role = "[doctor]"

    return dialogue


def fix_role(df: pd.DataFrame,
             dialogue_column: str,
             exceptions: List[List[str]] = None,
             **kwargs) -> pd.DataFrame:
    """
    Return a new dataframe where all sample in `dialogue_column` are fixed roles
    """
    assert dialogue_column in df.columns, f"Expect {dialogue_column} in the dataframe input!"

    def _fix_role_each(row):
        utterance_list = init_from_str_dialogue(row[dialogue_column],
                                                task='C',
                                                exceptions=exceptions)
        utterance_list = fix_role_dialogue(utterance_list)
        new_dialogue = '\n'.join(f'{u.role}{str(u.spacy_doc)}' for u in utterance_list)
        row['fixed_role_dialogue'] = new_dialogue
        return row
    df = df.apply(_fix_role_each, axis=1)

    return df
