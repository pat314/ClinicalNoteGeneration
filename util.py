import re
from dataclasses import dataclass
from typing import *

import nltk
import pandas as pd
from spacy.tokens import Doc

from model_manager import get_spacy_model

@dataclass
class Utterance:
    role: str = None
    spacy_doc: Doc = None


@dataclass
class RetrievalResult:
    texts: List[str]
    scores: List[float]
    detailed_texts: List[str] = None
    detailed_scores: Union[List[float], List[List[float]]] = None


@dataclass
class Sentence:
    role: str = None
    spacy_doc: Doc = None


def init_utterance_text_list(text: str, task: str,
                             exceptions: List[List[str]] = None) -> List[Tuple[str]]:
    results = []
    if task == "A" or task == "B":
        role_pattern = '^[A-Z].*?:'
        # handle format exception
        if exceptions is not None:
            for x, y in exceptions:
                text = text.replace(x, y)
        Utterances_with_role = text.split('\r\n')
        roles = []
        utterances = []
        for each in Utterances_with_role:
            if not each:
                continue
            role = re.findall(role_pattern, each)
            if len(role) == 0:
                continue
            assert len(role) == 1, f"Invalid Utterance!, found {len(role)} roles in {each}"
            role = role[0].replace(':', '')
            roles.append(role)
            utterance = re.sub(role_pattern, '', each)
            utterances.append(utterance)
        results = list(zip(roles, utterances))
    elif task == "C":
        role_pattern = '\[.*?\] '
        # handle format exception
        if exceptions is not None:
            for x, y in exceptions:
                text = text.replace(x, y)
        Utterances_with_role = text.split('\n')
        roles = []
        utterances = []
        for each in Utterances_with_role:
            if not each:
                continue
            role = re.findall(role_pattern, each)
            if len(role) == 0:
                continue
            assert len(role) == 1, f"Invalid Utterance!, found {len(role)} roles in {each}"
            role = role[0].strip()
            roles.append(role)
            utterance = each.replace(role, '', 1)
            utterances.append(utterance)
        results = list(zip(roles, utterances))
    return results


def init_from_str_dialogue(text: str, task: str,
                           exceptions: List[List[str]] = None) -> List[Utterance]:
    """
    Split a text (utf-8) into a list of Utterances
    """

    text_results = init_utterance_text_list(text, task, exceptions=exceptions)
    nlp = get_spacy_model()
    spacy_docs = nlp.pipe([x[-1] for x in text_results])
    roles = [x[0] for x in text_results]
    results = []
    for spacy_doc, role in zip(spacy_docs, roles):
        results.append(Utterance(role=role, spacy_doc=spacy_doc))
    return results


def read_dataframe(input_file_path,
                   dialogue_column: str,
                   index_column: str, **kwargs):
    df = pd.read_csv(input_file_path)
    assert index_column in df, f"Column {index_column} not found in the csv file! Available columns are {df.columns}"
    assert dialogue_column in df, f"Column {dialogue_column} not found in the csv file! Available columns are {df.columns}"
    return {"df": df}


def read_dataset(input_file_path,
                 dialogue_column: str,
                 index_column: str,
                 task: str = "C",
                 exceptions: List[List[str]] = None) -> Dict:
    df = pd.read_csv(input_file_path)
    # assert index_column in df, f"Column {index_column} not found in the csv file! Available columns are {df.columns}"
    assert dialogue_column in df, f"Column {dialogue_column} not found in the csv file! Available columns are {df.columns}"
    dialogue_list = []
    for sample in df[dialogue_column]:
        dialogue_list.append(init_from_str_dialogue(sample, task=task, exceptions=exceptions))
    return {"dialogue_list": dialogue_list, "df": df}



def save_df(df: pd.DataFrame, output_file_path: str, **kwargs) -> bool:
    df.to_csv(output_file_path, index=False)
    return True


def get_sent_from_utterance(utterance_list: List[Utterance]) -> List[Sentence]:
    """
    Sentence segmentation of a dialogue
     (recommend spacy.Doc)
    Param: List of utterances
    Return: List of sentences
    """
    result = []
    nlp = get_spacy_model()
    for utterance in utterance_list:
        # for sent in utterance.spacy_doc.sents:
        #     result.append(Sentence(role=utterance.role, spacy_doc=sent.as_doc()))
        if re.search(r"^(\.|\,)", utterance.spacy_doc.text):
            new_sent = utterance.spacy_doc.text[2:]
        else:
            new_sent = utterance.spacy_doc.text
        sents = nltk.sent_tokenize(new_sent)
        for sent in sents:
            result.append(Sentence(role=utterance.role, spacy_doc=nlp(sent)))

    return result


def get_formatted_dialogue(text, task) -> List[Sentence]:
    """
    Take a spoken-cleaned dialogue with punctuations from dataset,
    convert it to list of fully preprocessed Sentences ready for next tasks.
    """
    nlp = get_spacy_model()
    utterance_list = init_from_str_dialogue(text, task)
    sent_list = get_sent_from_utterance(utterance_list)
    for i in range(len(sent_list)):
        sent_list[i].spacy_doc = nlp(sent_list[i].spacy_doc.text)
    return sent_list


def is_meaning_full_sentence(text):
    words = text.split()
    words = [w for w in words if len(w) > 0 and w[0] != '[' and w[-1] != ']']
    return len(words) >= 3


def split_sentence(text,
                   mark_noisy_sentence: bool = False,
                   marker: str = None,
                   simple: bool = True) -> List[str]:
    if simple:
        sents = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    else:
        nlp = get_spacy_model()
        spacy_doc = nlp(text)
        sents = [str(sent) for sent in spacy_doc.sents]

    if mark_noisy_sentence:
        for idx, sent in enumerate(sents):
            if not is_meaning_full_sentence(sent):
                sents[idx] = marker + sent
    return sents