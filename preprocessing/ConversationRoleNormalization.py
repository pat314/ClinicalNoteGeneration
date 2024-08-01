import pandas as pd
from dataclasses import dataclass
import spacy
from typing import *
from spacy.tokens import Doc
import re

_SPACY_MODEL = None


@dataclass
class Utterance:
    role: str = None
    spacy_doc: Doc = None


def get_spacy_model() -> spacy.language.Language:
    """
    Load SpaCy model one time
    """

    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        _SPACY_MODEL = spacy.load("en_core_web_sm")
    return _SPACY_MODEL


def initialize_utterance_list(text: str) -> List[Utterance]:
    """
    Convert a text into a list of Utterance objects with roles and spacy_doc.
    """
    role_pattern = r'\[.*?\] '
    utterances_with_role = text.split('\n')
    results = []

    roles = []
    utterances = []
    for entry in utterances_with_role:
        if not entry:
            continue

        role_matches = re.findall(role_pattern, entry)
        if not role_matches:
            continue

        role = role_matches[0].strip()
        utterance = entry.replace(role, '', 1)
        roles.append(role)
        utterances.append(utterance)

    nlp = get_spacy_model()
    spacy_docs = nlp.pipe(utterances)
    for spacy_doc, role in zip(spacy_docs, roles):
        results.append(Utterance(role=role, spacy_doc=spacy_doc))

    return results


def check_for_questions(dialogue: List[Utterance]):
    """
    Count the questions each role asked
    """

    questions_doctor = 0
    questions_speaker2 = 0

    for utterance in dialogue:
        if utterance.role == "[doctor]":
            questions_doctor += utterance.spacy_doc.text.count("?")
        else:
            questions_speaker2 += utterance.spacy_doc.text.count("?")
    return questions_doctor, questions_speaker2


def swap_roles(dialogue: List[Utterance], speaker2_role):
    """
    Swap roles between doctor and the second speaker in the dialogue.
    """
    for utterance in dialogue:
        if utterance.role == "[doctor]":
            utterance.role = speaker2_role
        else:
            utterance.role = "[doctor]"


def fix_role_dialogue(dialogue: List[Utterance]):
    """
    Use rule-based approach to correct role errors in the dialogue.
    """
    for utterance in dialogue:
        if utterance.role != "[doctor]":
            speaker2_role = utterance.role

    if dialogue[0].role == "[patient]":
        swap_roles(dialogue, speaker2_role)
    else:
        questions_by_doctor, questions_by_speaker2 = check_for_questions(dialogue)
        if questions_by_doctor < questions_by_speaker2:
            swap_roles(dialogue, speaker2_role)

    return dialogue


def fix_role(df: pd.DataFrame, dialogue_column: str) -> pd.DataFrame:
    """
    Return a new dataframe where all samples in `dialogue_column` have fixed roles.
    """

    def _fix_all_role(row):
        utterance_list = initialize_utterance_list(row[dialogue_column])
        utterance_list = fix_role_dialogue(utterance_list)
        row['fixed_role_dialogue'] = '\n'.join(f'{u.role} {u.spacy_doc.text}' for u in utterance_list)
        return row

    df = df.apply(_fix_all_role, axis=1)

    return df
