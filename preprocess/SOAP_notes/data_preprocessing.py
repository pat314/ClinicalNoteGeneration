# -*- coding: utf-8 -*-

"""
Package: preprocess/SOAP_notes
Writer: Hoang-Thuy-Duong Vu
File: data_preprocessing.py
Project: CNG - Clinical Note Generation Ver.3
---
Current version: written on 6 oct. 2024
"""

# ---------------------------
# DATA PREPROCESSING
# ---------------------------

# Import necessary libraries
import numpy as np 
import pandas as pd 



# TRAIN SET
df = pd.read_csv("train.csv")

assert len(df["ID"]) == len(df)
assert len(np.unique(df["Label"])) == 4

#df["Processed text"] = [df["Text"][i].split(" . ") for i in range(len(df))]   # Split text
#df.to_csv("train.csv", index=False)




# TEST SET
df = pd.read_csv("test.csv")

assert len(df["ID"]) == len(df)
assert len(np.unique(df["Label"])) == 4

# Drop columns where the fill rate < 50%
for col in df.columns : 
  if type(df[col][0])!=str and len([i for i in range(len(df)) if np.isnan(df[col][i])==True]) > len(df)//2: 
    df.drop(col,axis=1,inplace=True)

#df["Processed text"] = [df["Text"][i].split(" . ") for i in range(len(df))]   # Split text

df.to_csv("test.csv", index=False)

