from typing import List, Dict

import pandas as pd

from .clean_spoken import clean_spoken
from .group_dialogue import group_dialogue
from .check_role import fix_role
from .restore_punctuation import restore_punc
from ..model_manager import get_spacy_model
from ..util import Utterance, init_from_str_dialogue


def statistic_each_dialogue(dialogue: str, task: str):
    """
    Describe a dialogue based on:
        - Total length in word; sentence; uterance
        - uterances/role
        - words/uterance/role; sentences/uterance/role
        - NER/uterance/role
    """
    figures = {}
    uterances = init_from_str_dialogue(dialogue, task=task)
    figures['Dialogue length in uterance'] = len(uterances)
    figures['Dialogue length in word'] = sum([len(each.spacy_doc) for each in uterances])
    figures['Dialogue length in sentence'] = sum([len(list(each.spacy_doc.sents)) for each in uterances])
    to_merge_figures = {}
    roles = {}
    for each in uterances:
        if each.role not in roles:
            roles[each.role] = 1
        else:
            roles[each.role] += 1
        if f'Uterance length in word of {each.role}' not in to_merge_figures:
            to_merge_figures[f'Uterance length in word of {each.role}'] = []
        to_merge_figures[f'Uterance length in word of {each.role}'].append(len(each.spacy_doc))
        if f'Uterance length in sentence of {each.role}' not in to_merge_figures:
            to_merge_figures[f'Uterance length in sentence of {each.role}'] = []
        to_merge_figures[f'Uterance length in sentence of {each.role}'].append(len(list(each.spacy_doc.sents)))
        if f'Number of named entities in uterance of {each.role}' not in to_merge_figures:
            to_merge_figures[f'Number of named entities in uterance of {each.role}'] = []
        to_merge_figures[f'Number of named entities in uterance of {each.role}'].append(len(list(each.spacy_doc.ents)))

    for role, num in roles.items():
        figures[f'Dialogue length in uterance of {role}'] = num
    return figures, to_merge_figures, uterances


def describe_text(text):
    """
    Describe a text
    """
    nlp = get_spacy_model()
    spacy_doc = nlp(text)
    figures = {}
    figures['Number of words'] = len(spacy_doc)
    figures['Number of sentences'] = len(list(spacy_doc.sents))
    figures['Number of entities'] = len(list(spacy_doc.ents))
    to_merge = {}
    to_merge['Sentence length in word'] = [len(sent) for sent in spacy_doc.sents]
    return figures, to_merge


def preprocessing(dialogue_list: List[List[Utterance]],
                  df: pd.DataFrame,
                  dialogue_column: str,
                  **kwargs) -> Dict:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue, this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    df_cleaned = clean_spoken(df, dialogue_column=dialogue_column, **kwargs)
    df_punc = restore_punc(df_cleaned, dialogue_column='clean_dialogue', **kwargs)
    df_fix_role = fix_role(df_punc, dialogue_column='restore_punctuation_dialogue', **kwargs)
    df_group = group_dialogue(df_fix_role, dialogue_column='fixed_role_dialogue', **kwargs)

    return {
        "dialogue_list": dialogue_list, "df": df_group
    }
