import logging
import re
from typing import Tuple, Union

import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import (
    BartForConditionalGeneration,
    pipeline,
    PreTrainedModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Pipeline,
    QuestionAnsweringPipeline,
    BartTokenizer)

logger = logging.getLogger(__name__)

_SPACY_MODEL = None
_PUNCTUATION_MODEL = None
_SBERT = {}
_QAPIPE = {}
_BART_GEN = {}


class PunctuationModel():
    """ Copied from `punctuationmodel.PunctuationModel`.
    This customization aims at auto-device navigation"""

    def __init__(self, model="oliverguhr/fullstop-punctuation-multilang-large") -> None:
        _name = model
        model = AutoModelForTokenClassification.from_pretrained(_name)
        tokenizer = AutoTokenizer.from_pretrained(_name)
        self.target_device = auto_detect_device(model)
        self.pipe = pipeline("ner",
                             model=model,
                             tokenizer=tokenizer,
                             grouped_entities=False, device=self.target_device)

    def preprocess(self, text):
        # remove markers except for markers in numbers
        text = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "", text)
        text = text.split()
        return text

    def restore_punctuation(self, text):
        result = self.predict(self.preprocess(text))
        return self.prediction_to_text(result)

    def overlap_chunks(self, lst, n, stride=0):
        """Yield successive n-sized chunks from lst with stride length of overlap."""
        for i in range(0, len(lst), n - stride):
            yield lst[i:i + n]

    def predict(self, words):
        overlap = 5
        chunk_size = 230
        if len(words) <= chunk_size:
            overlap = 0

        batches = list(self.overlap_chunks(words, chunk_size, overlap))

        # if the last batch is smaller than the overlap,
        # we can just remove it (in condition that batches have elements)
        # batches doesn't have any elements when there's a space line in the dataset (noise)
        if len(batches) > 0 and len(batches[-1]) <= overlap:
            batches.pop()

        tagged_words = []
        for batch in batches:
            # use last batch completely
            if batch == batches[-1]:
                overlap = 0
            text = " ".join(batch)
            result = self.pipe(text)
            assert len(text) == result[-1]["end"], "chunk size too large, text got clipped"

            char_index = 0
            result_index = 0
            for word in batch[:len(batch) - overlap]:
                char_index += len(word) + 1
                # if any subtoken of an word is labled as sentence end
                # we label the whole word as sentence end
                label = 0
                while result_index < len(result) and char_index > result[result_index]["end"]:
                    label = result[result_index]['entity']
                    score = result[result_index]['score']
                    result_index += 1
                tagged_words.append([word, label, score])

        assert len(tagged_words) == len(words)
        return tagged_words

    def prediction_to_text(self, prediction):
        result = ""
        for word, label, _ in prediction:
            result += word
            if label == "0":
                result += " "
            if label in ".,?-:":
                result += label + " "
        return result.strip()


def get_bart_gen_model(model_name,
                       use_auth_token) -> Tuple[BartForConditionalGeneration, BartTokenizer]:
    global _BART_GEN
    if model_name not in _BART_GEN:
        model = BartForConditionalGeneration.from_pretrained(model_name, use_auth_token=use_auth_token)
        tokenizer = BartTokenizer.from_pretrained(model_name, use_auth_token=use_auth_token)
        _BART_GEN[model_name] = (model, tokenizer)
    return _BART_GEN[model_name]


def get_spacy_model() -> spacy.language.Language:
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        _SPACY_MODEL = spacy.load("en_core_web_sm")
    return _SPACY_MODEL


def get_punc_restore_model() -> PunctuationModel:
    global _PUNCTUATION_MODEL
    if _PUNCTUATION_MODEL is None:
        _PUNCTUATION_MODEL = PunctuationModel()
    return _PUNCTUATION_MODEL


def get_sbert_model(name: str) -> SentenceTransformer:
    global _SBERT
    if name not in _SBERT:
        _SBERT[name] = SentenceTransformer(name, device='cpu')
    return _SBERT[name]


def get_question_answering_pipeline(model_name: str) -> QuestionAnsweringPipeline:
    global _QAPIPE
    if model_name not in _QAPIPE:
        _QAPIPE[model_name] = pipeline("question-answering", model=model_name, tokenizer=model_name)
    return _QAPIPE[model_name]


def auto_detect_device(model: Union[PreTrainedModel, Pipeline, SentenceTransformer]) -> Union[torch.device, str]:
    logger.info('auto_detect_device...')
    scale = 2.5
    if isinstance(model, Pipeline):
        scale = 4
    elif isinstance(model, SentenceTransformer):
        scale = 25  # sbert use very large batch size,
    device_count = torch.cuda.device_count()

    if isinstance(model, Pipeline):
        model_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
    else:
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    logger.info(f'model-size {model_size / 1024 / 1024} Mb')

    # Check if there are CUDA devices available
    if device_count > 0:
        for i in range(device_count):
            device = torch.device(f"cuda:{i}")

            available_mem, used = torch.cuda.mem_get_info(device)

            logger.info(f'cuda:{i}: available_mem = {available_mem / 1024 / 1024} Mb')

            if model_size * scale < available_mem:
                return f"cuda:{i}"

    return 'cpu'
