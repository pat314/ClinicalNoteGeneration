import pandas as pd
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# default
# model_name = "oliverguhr/fullstop-punctuation-multilang-large"
# model = AutoModelForTokenClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

def text_chunk(words, overlap = 5, chunk_size = 230):
    if len(words) < chunk_size:
        overlap = 0
    for i in range(0, len(words), chunk_size - overlap):
        yield words[i:i+chunk_size]

def text_split(text):
  tmp_text = re.sub(r"(?<!\d)[.,;:!?](?!\d)", "", text)
  words = tmp_text.split()
  return words

def get_role_utterance(text):
  role_pattern = '\[.*?\] '
  utterances_with_role = text.split('\n')
  roles = []
  utterances = []
  for item in utterances_with_role:
    if not item:
      continue
    role = re.findall(role_pattern, item)
    if len(role) == 0:
      continue
    assert len(role) == 1, f"Invalid Utterance!, found {role} {len(role)} roles in {item}"
    role = role[0].replace(':', '')
    roles.append(role)
    utterance = re.sub(role_pattern, '', item)
    utterances.append(utterance)
  return list(zip(roles, utterances))


def predict_punctuation(text: str, model, tokenizer, pipe):
  
    overlap = 5
    chunk_size = 230

    words = text_split(text)
    chunks = list(text_chunk(words, overlap, chunk_size))
    tagged_words = []
    for item in chunks:

        if item == chunks[-1]:
            overlap = 0
        tmp_text = " ".join(item)
        result = pipe(tmp_text)

        char_index = 0
        result_index = 0
        for word in item[:len(item) - overlap]:
            entity = 0
            char_index += len(word) + 1

            while result_index < len(result) and char_index > result[result_index]["end"]:
                entity = result[result_index]['entity']
                score = result[result_index]['score']

                result_index += 1
            tagged_words.append([word, entity, score])
            
    return tagged_words

def get_prediction(tagged_words):
    result = ""
    for word, label, score in tagged_words:
        supple = ""
        if label in ".,;:!?":
            supple = label
        tmp_text = word + supple + ' '
        result += tmp_text
    return result

def restore_punctuation(df : pd.DataFrame, dialogue_column : str, 
            model = AutoModelForTokenClassification.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large"), 
            tokenizer = AutoTokenizer.from_pretrained("oliverguhr/fullstop-punctuation-multilang-large")) -> pd.DataFrame:
    pipe = pipeline("ner", model = model, tokenizer = tokenizer, device='cpu', grouped_entities=False)
    text = list(df[dialogue_column])
    role_utters = []
    results = []
    for each in text:
      role_utter = get_role_utterance(each)
      role_utters.append(role_utter) 
      result = []
      for role, utterance in role_utter:
        tagged_words = predict_punctuation(utterance, model, tokenizer, pipe)
        predict_text = get_prediction(tagged_words)
        result.append(f'{role}{": "}{predict_text}')
      results.append(result)
    df['restore_punctuation_dialogue'] = results
    return df
