# -*- coding: utf-8 -*-

"""
Package: preprocess/tools
Writer: Hoang-Thuy-Duong Vu
File: side.py
Project: CNG - Clinical Note Generation Ver.3
---
Current version: written on 6 oct. 2024
"""

# Import necessary libraries
import numpy as np

def mean_sample(sample): 
  res = []
  for i in range(len(sample[0])) : 
    tmp = np.sum([sample[j][i] for j in range(len(sample))])
    res.append(tmp)
  return res
