# -*- coding: utf-8 -*-

"""
Package: preprocess
Writer: Hoang-Thuy-Duong Vu
File: text_embedding.py
Project: CNG - Clinical Note Generation Ver.3
Src: emilyalsentzer/Bio_ClinicalBERT
---
Current version: written on 22 sept. 2024
Note: possible to be reprlaced with sentence-transformers/all-MiniLM-L6-v2 if too slow
"""

# ---------------------------
# UTTERANCE ENCODER - BioClinicalBERT
# ---------------------------

# Import necessary libraries
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
# Suppress a specific warning type
warnings.filterwarnings("ignore", category=DeprecationWarning)




def bcbert_encoder(Utt : str) -> torch.Tensor: 
  """
  Implement BioClinicalBERT to embed an input utterance (Utt) as a normalized vector embedding.
  """
  # Import necessary libraries
  from transformers import AutoTokenizer, AutoModel
  import torch.nn.functional as F

  def mean_pooling(model_output : torch.Tensor, attention_mask : torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on the token embeddings to obtain sentence embeddings, taking the attention mask into account.
    
    Args:
        model_output (torch.Tensor): The BERT model output tensor of shape (batch_size, seq_length, hidden_dim).
        attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, seq_length).
    
    Returns:
        torch.Tensor: Mean-pooled embeddings of shape (batch_size, hidden_dim).
    """
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]  # Shape: (batch_size, seq_length, hidden_dim)

    # Expand the attention mask to match the token embeddings' dimensions
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Perform mean pooling, taking the attention mask into account
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)  # Avoid division by zero
    mean_pooled_embeddings = sum_embeddings / sum_mask  # Shape: (batch_size, hidden_dim)
    
    return mean_pooled_embeddings


  # Load model from HuggingFace Hub
  tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
  model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")  

  sentences = Utt.split("\n")

  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt',max_length=1024)

  # Compute token embeddings
  with torch.no_grad():
    model_output = model(**encoded_input)

  # Perform pooling
  cluster_center = mean_pooling(model_output, encoded_input['attention_mask'])

  # Normalize embeddings
  cluster_center = F.normalize(cluster_center, p=2, dim=1)

  return cluster_center
