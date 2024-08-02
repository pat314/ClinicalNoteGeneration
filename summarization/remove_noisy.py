!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from typing import Union, Dict, Tuple, List, Any
import re
import torch

_MEANINGLESS_TEXT = '<--NOISY-->'

def cosine_matrix(a: np.array, b: np.array) -> np.array:

    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_normalized = a / a_norm
    b_normalized = b / b_norm

    similarity = np.matmul(a_normalized, b_normalized.T)
    return similarity

def build_markov(text,
                 query,
                 model: str = 'paraphrase-MiniLM-L6-v2',
                 device = 'cpu'):
    model = SentenceTransformer(model, device='cpu')
    device = torch.device(device)

    text_encode = model.encode(text,
                               device = device,
                               show_progress_bar = False,
                               convert_to_numpy = True)

    if text_encode.ndim == 1:
        text_encode = text_encode.reshape(1, -1)

    query_encode = model.encode(query,
                                 device = device,
                                 show_progress_bar = False,
                                 convert_to_numpy = True)
    if query_encode.ndim == 1:
        query_encode = query_encode.reshape(1, -1)



    markov = cosine_matrix(query_encode, text_encode)

    return markov

def is_meaningfull_sentence(text):
    words = text.split()
    words = [word for word in words if len(word) > 0 and word[0] != '[' and word[-1] != ']']
    return len(words) >= 3

def split_sentence(text,
                   mark_noisy_sentence: bool = False,
                   marker: str = None,
                   simple: bool = True) -> List[str]:
    if simple:
        texts = re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
    else:
        nlp = spacy.load("en_core_web_sm")
        spacy_doc = nlp(text)
        texts = [str(sent) for sent in spacy_doc.sents]

    if mark_noisy_sentence:
        for idx, sent in enumerate(texts):
            if not is_meaningfull_sentence(sent):
                texts[idx] = marker + sent
    return texts

def remove_roles(text: str) -> str:
    ans = re.sub(r"\[(\w+)\]", '', text)
    return ans

def soft_cutoff(text: str,
                spoken_terms: List[str] = None,
                wanted_like_sentences: List[str] = None,
                pos_thresh: float = None,
                unwanted_like_sentences: List[str] = None,
                neg_thresh: float = None,
                model: str = 'paraphrase-MiniLM-L6-v2',
                device: str = 'cpu'):
    sents = split_sentence(text = text,
                           mark_noisy_sentence = True,
                           marker = _MEANINGLESS_TEXT,
                           simple = False)
    sents = [sent for sent in sents if sent[:len(_MEANINGLESS_TEXT)] != _MEANINGLESS_TEXT]

    if wanted_like_sentences and sents:
        a = build_markov(text = sents,
                         query = wanted_like_sentences,
                         model = model,
                         device = device)
        scores = a.max(axis = 0)

        sents = [sents[i] for i in np.argwhere(scores >= pos_thresh).reshape(-1)]
    if unwanted_like_sentences and sents:
        a = build_markov(text = sents,
                         query = unwanted_like_sentences,
                         model = model,
                         device = device)
        scores = a.max(axis = 0)
        sents = [sents[i] for i in np.argwhere(scores < neg_thresh).reshape(-1)]
    text = ' '.join(sents)
    if spoken_terms:
        pattern = f"\\b({'|'.join(spoken_terms)})" + r'[\s!"\#\$%\&\'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}\~]'
        text = re.sub(pattern, "", text, flags=re.I)

    return text

def remove_noisy_sentence(text: str,
                          suffix: str = None,
                          use_soft_cutoff: bool = False,
                          drop_roles: Union[str, List[str]] = None,
                          spoken_terms: List[str] = None,
                          wanted_like_sentences: List[str] = None,
                          pos_thresh: float = 0.1,
                          unwanted_like_sentences: List[str] = None,
                          neg_thresh: float = 0.7,
                          model: str = 'paraphrase-MiniLM-L6-v2',
                          device: str = 'cpu') -> str:

    if drop_roles and isinstance(drop_roles, str):
        drop_roles = [drop_roles]
    if drop_roles:
        texts = []
        to_check = []
        for role in drop_roles:
            to_check.append(f'[{role}]')
        for utter in text.split('\n'):
            if all(utter.strip()[:len(i)] != i for i in to_check):
                texts.append(utter)
            text = '\n'.join(texts)
    sents = split_sentence(text,
                           mark_noisy_sentence = True,
                           marker = _MEANINGLESS_TEXT,
                           simple = False)
    text = [sent for sent in sents if sent[:len(_MEANINGLESS_TEXT)] != _MEANINGLESS_TEXT]
    text = ' '.join(text)
    text = remove_roles(text)

    if use_soft_cutoff:
        text = soft_cutoff(text=text,
                           spoken_terms=spoken_terms,
                           wanted_like_sentences=wanted_like_sentences,
                           pos_thresh=pos_thresh,
                           unwanted_like_sentences=unwanted_like_sentences,
                           neg_thresh=neg_thresh,
                           model=model,
                           device=device)

    if suffix is not None:
        text += suffix
    return text