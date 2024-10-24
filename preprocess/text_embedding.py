# -*- coding: utf-8 -*-

"""
Package: preprocess
Writer: Hoang-Thuy-Duong Vu
File: utterance-encoder.py
Project: CNG - Clinical Note Generation Ver.3
Src: emilyalsentzer/Bio_ClinicalBERT
---
Current version: written on 22 sept. 2024
Note: possible to be reprlaced with sentence-transformers/all-MiniLM-L6-v2 if too slow
"""

# ---------------------------
# UTTERANCE ENCODER - BioClinicalBERT
# ---------------------------

def bcbert_encoder(sentences): 
  """
  Implement SBERT from pre-trained model on HuggingFace Hub
  """
  # Import necessary libraries
  from transformers import AutoTokenizer, AutoModel
  import torch
  import torch.nn.functional as F

  # Predefined function
  def mean_pooling(model_output, attention_mask):
    """
    Obj: Take attention mask into account for correct averaging
    """
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
  model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")  

  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)

  # Perform pooling
  sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  # Normalize embeddings
  sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

  return sentence_embeddings


def sim_score(sentence_embeddings): 
  pass