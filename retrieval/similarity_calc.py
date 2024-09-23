# -*- coding: utf-8 -*-

"""
Package: retrieval
Writer: Hoang-Thuy-Duong Vu
File: similarity_calc.py
Project: CNG - Clinical Note Generation Ver.3
Src: L1 norm
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
    for i in range(len(s)): 
      min_val = s[i].min()
      max_val = s[i].max()

      mat.append((s[0] - min_val) / (max_val - min_val))
    
    return np.array(mat)

    # ---------------------------
    # L1 norm
    # ---------------------------
    def _L1_norm(array): 
      """
      Input:
      array -- np.ndarray: array of data
      Hypothesis: L1 norm

      ---------------------------
      Ouput:
      res -- int: L1 norm of `array`
      """
      return np.sqrt(np.sum(array))

    # ---------------------------
    # Cosine Sim calculation
    # ---------------------------
    def cossim(embedded_sentence): 
      """
      Input:
      df -- pd.DataFrame: input data to normalize

      ---------------------------
      Ouput:
      df_normalized -- pd.DataFrame: normalized dataframe
      """
      embedded_sentence = _normalization(embedded_sentence)     # Normalize embedded sentence
      mat = np.zeros((len(embedded_sentence), len(embedded_sentence)))
      
      for i in range(len(embedded_sentence)) : 
        for j in range(len(embedded_sentence)) : 
          if i == j: 
            mat[i][j] = 1
          else: 
            A = np.array(embedded_sentence[i])
            B = np.array(embedded_sentence[j])
            mat[i][j] = np.dot(A, B) / (_L1_norm(A) * _L1_norm(B))
      return mat
    return cossim(embedded_sentence)

