# -*- coding: utf-8 -*-

"""
Package: retrieval
Writer: Hoang-Thuy-Duong Vu
File: similarity_calc.py
Project: CNG - Clinical Note Generation Ver.3
Src: L2 norm
---
Current version: written on 22 sept. 2024
"""



# ---------------------------
# SIMILARITY MATRIX - COSINE SIMILARITY SCORE
# ---------------------------

def cossim_score(embedded_sentence): 
  """
  Input:
  embedded_sentence -- tensor: embedded sentence from encoder
  Obj: calculate the similarity matrix using Cosine Similarity Score

  ---------------------------
  Ouput:
  df_normalized -- pd.DataFrame: normalized dataframe
  """
  # Import global necessary libraries
  import numpy as np

  # ---------------------------
  # Data Normalization
  # ---------------------------
  def _normalization(df) : 
    """
    Input:
    df -- pd.DataFrame: input data to normalize

    ---------------------------
    Ouput:
    df_normalized -- pd.DataFrame: normalized dataframe
    """
    # define matrix
    mat = []

    # iteration
    for i in range(len(df)): 
      min_val = df[i].min()
      max_val = df[i].max()

      mat.append((df[i] - min_val) / (max_val - min_val))
    return np.array(mat)

  # ---------------------------
  # L2 norm
  # ---------------------------
  def _L2_norm(array): 
    """
    Input:
    array -- np.ndarray: array of data
    Hypothesis: L2 norm

    ---------------------------
    Ouput:
    res -- int: L2 norm of `array`
    """
    return np.sqrt(np.sum(array**2))

  # ---------------------------
  # Cosine Sim calculation
  # ---------------------------
  def _cossim(sentence): 
    """
    Input:
    df -- pd.DataFrame: input data to normalize

    ---------------------------
    Ouput:
    df_normalized -- pd.DataFrame: normalized dataframe
    """
    sentence = _normalization(sentence)     # Normalize embedded sentence
    mat = np.zeros((len(sentence), len(sentence)))
    
    for i in range(len(sentence)) : 
      for j in range(len(sentence)) : 
        if i == j: 
          mat[i][j] = 1
        else: 
          A = np.array(sentence[i])
          B = np.array(sentence[j])
          mat[i][j] = np.dot(A, B) / (_L2_norm(A) * _L2_norm(B))
    return mat
    
  return _cossim(embedded_sentence)

