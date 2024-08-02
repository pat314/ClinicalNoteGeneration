import pandas as pd
import torch
import re
from typing import Union, Dict, Tuple, List

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, PreTrainedModel, PreTrainedTokenizer, pipeline, BartForConditionalGeneration, BartTokenizer

class RetrievalResult:
    texts: List[str]
    scores: List[float]
    detailed_texts: List[str] = None
    detailed_scores: Union[List[float], List[List[float]]] = None
    def __init__(self, texts: List[str], scores: List[float], detailed_texts: List[str] = None, detailed_scores: Union[List[float], List[List[float]]] = None):
        self.texts = texts
        self.scores = scores
        self.detailed_texts = detailed_texts
        self.detailed_scores = detailed_scores

def dialogue_normalize(text: Union[List[str],str]) -> Union[List[str],str]:
    pattern = r"\[(\w+)\]"
    if isinstance(text, str):  # single question
        normalized_text = re.sub(pattern, r"\1:", text)
    else:
        normalized_text = []
        for item in text:
            normalized_text.append(re.sub(pattern, r"\1:", item))
    return normalized_text

def replace_pronouns(dialogue):
    new_dialogue = []
    pattern = r"\[(\w+)\]"
    roles = [re.sub(pattern, r"\1", m) for m in re.findall(pattern, dialogue)]
    for utt in dialogue.split('\n'):
        matches = re.findall(pattern, utt)

        if len(matches) > 0:
            normalized_role = re.sub(pattern, r"\1", matches[0])
            other_roles = [r for r in roles if r != normalized_role]
            has_other = len(other_roles) > 0
            i = normalized_role
            utt = re.sub(r"\bI\b", i, utt, flags=re.IGNORECASE)
            if has_other:
                utt = re.sub(r"\byou\b", other_roles[0], utt, flags=re.IGNORECASE)
        new_dialogue.append(utt)
    return '\n'.join(new_dialogue)


def question_extract_then_fill_text(text: str,
                                    question: Union[List[str], str],
                                    prefix: Union[List[str], str] = None,
                                    model: str = "distilbert-base-cased-distilled-squad",
                                    device: str = 'cpu',
                                    return_score: bool = False,
                                    normalize: bool = True,
                                    replace_pronoun = False) -> Union[Tuple[str, List[str]], str]:
    _model = AutoModelForQuestionAnswering.from_pretrained(model)
    _tokenizer = AutoTokenizer.from_pretrained(model)
    pipe = pipeline("question-answering", model = _model, tokenizer = _tokenizer)

    assert (normalize ^ replace_pronoun) or (not (normalize or replace_pronoun)), "Only either 'replace_pronoun' or 'normalize', or none of them is turned on"
    if normalize:
        text = dialogue_normalize(text)
    if replace_pronouns:
        text = replace_pronouns(text)
    if isinstance(question, str):
        answer = pipe(question = question, context = text, device = device)
        text_ans = answer['answer']
        scores = answer['score']
        output = text_ans
    else:
        scores = []
        text_ans = []
        output = []
        for item in question:
            answer = pipe(question = item, context = text, device = device)
            scores.append(answer['score'])
            text_ans.append(answer['answer'])
        for prefix_item, text_ans_item in zip(prefix, text_ans):
            output.append(prefix_item + text_ans_item)


    if return_score:
        to_return = (output, scores)
    else:
        to_return = output

    return to_return

def question_extract_then_fill(retrieval_result: RetrievalResult,
                               question: str,
                               prefix: str,
                               model: str = "distilbert-base-cased-distilled-squad",
                               device: str = 'cpu',
                               return_score: bool = False,
                               normalize: bool = True,
                               replace_pronoun: bool = False,
                               **kwargs) -> Union[Tuple[str, List[str]], str]:
    text = '\n'.join(retrieval_result.texts)
    if not text.strip():
        return ''
    return question_extract_then_fill_text(text,
                                           question=question,
                                           prefix=prefix,
                                           model=model,
                                           device=device,
                                           return_score=return_score,
                                           normalize=normalize,
                                           replace_pronoun=replace_pronoun,
                                           **kwargs)

def naive_extract_after_terms(text: str,
                              suffix: str = None,
                              exceptions: List[str] = None) -> str:
    substring1 = [
        'assessment', 'my impression', ' plan'
    ]

    substring2 = "-year-old"

    text1, text2 = "", ""
    r1, r2 = "", ""

    for s in substring1:
        if text.find(s) != -1:
            index = text.find(s)

            if exceptions:
                if any([result[:len(i)] == i for i in exceptions]):
                    continue
            result = "\n".join(result.split("\n<GROUP>\n"))
            r1 = result
            result = re.sub("(\\[doctor\\]|\\[patient\\])", "", result)
            text1 = result

    new_text = text.split(".")
    for sent in new_text:
        if re.search(substring2, sent):
            result = sent

            result = "\n".join(result.split("\n<GROUP>\n"))

            r2 = result
            result = re.sub("(\\[doctor\\] |\\[patient\\] |\n<GROUP>\n)", "", result)
            text2 = result

    to_return = text2 + "\n" + text1
    if suffix is not None:
        to_return += suffix
    return to_return

def summarize_text(text: str,
                    model: Union[PreTrainedModel, str] = "philschmid/bart-large-cnn-samsum",
                    tokenizer: Union[PreTrainedTokenizer, str] = None,
                    prompt: str = None,
                    device: str = 'cpu',
                    normalize: bool = True,
                    replace_pronoun = False,
                    num_beams: int = None,
                    return_best_only = True,
                    **kwargs) -> Union[str, List[Tuple[str, float]]]:
    if isinstance(model, str):
        model_name = model
        model = BartForConditionalGeneration.from_pretrained(model_name)
        tokenizer = BartTokenizer.from_pretrained(model_name)
    model = model.to(device)
    assert (normalize ^ replace_pronoun) or (not (normalize or replace_pronoun)), "Only either 'replace_pronoun' or 'normalize', or none of them is turned on"
    if normalize:
        text = dialogue_normalize(text)
    if replace_pronouns:
        text = replace_pronouns(text)
    encoder_inputs = tokenizer(text, truncation = True, return_tensors='pt')

    inputs = encoder_inputs
    if prompt:
        with tokenizer.as_target_tokenizer():
            decoder_inputs = tokenizer(prompt,
                                       truncation = True,
                                       return_tensors = 'pt',
                                       add_special_tokens = False)
            decoder_inputs = {f'decoder_{k}': v for k, v in decoder_inputs.items()}
        inputs.update(decoder_inputs)
    if num_beams is None:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        gen = model.generate(**inputs,
                                return_dict_in_generate = True,
                                output_scores = True,
                                **kwargs)

        gen_texts = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)

        text_and_score = [(txt, sc.cpu().numpy().tolist()) for txt, sc in zip(gen_texts, gen.sequences_scores)]
        to_return = text_and_score
    else:
        num_return_sequences = num_beams
        gen = model.generate(**inputs,
                             return_dict_in_generate = True,
                             num_return_sequences = num_return_sequences,
                             output_scores = True,
                             num_beams = num_beams,
                             **kwargs)
        gen_texts = tokenizer.batch_decode(gen.sequences, skip_special_tokens=True)
    text_and_score = [(txt, sc.cpu().numpy().tolist()) for txt, sc in zip(gen_texts, gen.sequences_scores)]
    to_return = text_and_score

    if return_best_only:
        return to_return[0]
    return to_return

def summarize(retrieval_result: RetrievalResult,
              model: Union[PreTrainedModel, str] = "philschmid/bart-large-cnn-samsum",
              tokenizer: Union[PreTrainedTokenizer, str] = None,
              prompt: str = None,
              device: str = 'cpu',
              normalize: bool = True,
              replace_pronoun = False,
              num_beams: int = None,
              remove_prompt: bool = False,
              prefix: str = None,
              **kwargs) -> str:
    text = '\n'.join(retrieval_result.texts)
    if not text.strip():
        return ''

    text_sum = summarize_text(text,
                              model = model,
                              tokenizer = tokenizer,
                              prompt = prompt,
                              device = device,
                              normalize = normalize,
                              replace_pronoun = replace_pronoun,
                              return_best_only = True,
                              num_beams = num_beams,
                              **kwargs)
    if prompt is not None:
        non_prompt_sum = text_sum[len(prompt):]
    else:
        non_prompt_sum = text_sum
    if not remove_prompt:
        return text_sum
    else:
        return prefix + non_prompt_sum

