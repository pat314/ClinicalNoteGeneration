import re
from typing import List, Dict, Any, Tuple, Union

import pandas as pd

from .clean_spoken import change_pronouns
from model_manager import get_spacy_model
from util import Sentence, get_formatted_dialogue


def token_segmentor(text: Sentence):
    """
    Receive a Sentence, return a list of its tokens
    """
    nlp = get_spacy_model()
    text = text.spacy_doc.text.lower()
    doc = nlp(text)
    return [token for token in doc]


def is_question(sent: Sentence, question_patterns: tuple):
    """
    Detect if a sentence is a question.
    """
    text = sent.spacy_doc.text
    pattern1, pattern2 = question_patterns

    if re.search(pattern1, text) and not re.search(pattern2, text):
        return True
    else:
        return False


def is_end(sent: Sentence, end_patterns: tuple):
    """
    Detect if a sentence is the end of a group.
    """
    text = sent.spacy_doc.text
    pattern1 = end_patterns[0]

    if re.search(pattern1, text):
        return True
    else:
        return False


def is_start(sent: Sentence, question_patterns: tuple, start_patterns: tuple) -> Union[bool, str]:
    """
    Detect if a sentence is the start of a group.
    """
    text = sent.spacy_doc.text
    pattern1, pattern2, title = start_patterns

    if re.search(title, text):
        return "title"
    elif re.search(pattern1, text) or re.search(pattern2, text):
        return True
    else:
        return False


def not_end(sent: Sentence, notend_patterns: tuple):
    """
    Detect if a sentence can not be the end of a group.
    """
    text = sent.spacy_doc.text
    pattern = notend_patterns[0]

    if re.search(pattern, text):
        return True
    else:
        return False


def group_label(dialogue: List[Sentence],
                question_patterns: Tuple[str],
                start_patterns: Tuple[str],
                end_patterns: Tuple[str],
                notend_patterns: Tuple[str]):
    """
    Get a list of labelled grouping information for each sentence with a rule-based approach.
    A sentence could be labelled: start of a group, ending of a group, none.
    This information is used for grouping.
    """
    switched = True
    list = ["none", "none"]
    i = 2
    while i < len(dialogue) - 1:
        if dialogue[i].role != "[doctor]":
            switched = True
            list.append("none")
        else:
            if is_start(dialogue[i], question_patterns, start_patterns) == "title" and switched == True:
                list.append("start")
                switched = False
            elif is_question(dialogue[i], question_patterns) and dialogue[i + 1].role != "[doctor]" and switched == True:
                list.append("start")
                switched = False
            elif list[-1] == "end" and is_end(dialogue[i], end_patterns):
                list[-1] = "none"
                list.append("end")
            elif is_start(dialogue[i - 1], question_patterns, start_patterns):
                list.append("none")
            elif is_end(dialogue[i], end_patterns) and switched == True:
                switched = False
                list.append("end")
            elif is_start(dialogue[i], question_patterns, start_patterns) and not not_end(dialogue[i - 1], notend_patterns) and switched == True:
                switched = False
                list.append("start")
            else:
                list.append("none")
        i += 1
    list.append("none")
    return list


def print_dialogue(dialogue: List[Sentence],
                   PRONOUNS_CONTRACTIONS: Dict[str, str],
                   question_patterns: Tuple[str],
                   start_patterns: Tuple[str],
                   end_patterns: Tuple[str],
                   notend_patterns: Tuple[str]) -> str:
    """
    Print a dialogue, adding <GROUP> tokens.
    """
    tmp = 0
    dialogue = change_pronouns(dialogue, PRONOUNS_CONTRACTIONS)
    group_list = group_label(dialogue,
                             question_patterns=question_patterns,
                             start_patterns=start_patterns,
                             end_patterns=end_patterns,
                             notend_patterns=notend_patterns)
    i = 1
    text = dialogue[0].role + " " + dialogue[0].spacy_doc.text
    while i < len(dialogue):
        if group_list[i] == "start" or group_list[i - 1] == "end":
            text += "\n<GROUP>\n" + dialogue[i].role + " " + dialogue[i].spacy_doc.text
            tmp += 1
        else:
            if dialogue[i - 1].role == dialogue[i].role:
                text += " " + dialogue[i].spacy_doc.text
            else:
                text += "\n" + dialogue[i].role + " " + dialogue[i].spacy_doc.text
        i += 1
    return text


def group_dialogue(df: pd.DataFrame,
                   dialogue_column: str,
                   PRONOUNS_CONTRACTIONS: dict,
                    question_patterns: Tuple[str],
                   start_patterns: Tuple[str],
                   end_patterns: Tuple[str],
                   notend_patterns: Tuple[str],
                   verbose: bool = True,
                   **kwargs) -> pd.DataFrame:
    """
    Preprocessing function.
    Receive dialogue data (raw) of a list of dialogue,
        this function return a tuple of pre-processed dialogue, other meta data, etc.
    """
    assert dialogue_column in df.columns, f"Expected column '{dialogue_column}' in the dataframe"
    from tqdm import tqdm
    if verbose:
        verbose = tqdm(desc="Grouping dialogue...")
    else:
        verbose = None

    def _group(row: dict, _verbose: tqdm = None) -> dict:
        formatted_dialogue = get_formatted_dialogue(row[dialogue_column], "C")
        row['group_dialogue'] = print_dialogue(formatted_dialogue,
                                               PRONOUNS_CONTRACTIONS=PRONOUNS_CONTRACTIONS,
                                               question_patterns=question_patterns,
                                               start_patterns=start_patterns,
                                               end_patterns=end_patterns,
                                               notend_patterns=notend_patterns)
        if _verbose is not None:
            _verbose.update(1)
        return row

    df = df.apply(lambda r: _group(r, verbose), axis=1)

    return df
