# -*- coding: utf-8 -*-

"""
Package: preprocess
Writer: Hoang-Thuy-Duong Vu
File: classifier.py
Project: CNG - Clinical Note Generation Ver.3
---
Current version: written on 07 oct. 2024
"""

# Import necessary global libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import classifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import string
punc = string.punctuation

from stop_words import get_stop_words
stop_words = get_stop_words('en')

# CLASS: CLASSIFIER
class Classifier: 

  def __init__(self, method, x_train, y_train): 
    self.method = method
    self.x_train = x_train
    self.y_train = y_train


  def predict(self, x_test):
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer=string_to_list)), # messages to weighted TFIDF score
        ('classifier', self.method())                    # train on TFIDF vectors with Naive Bayes
    ])
    pipeline.fit(self.x_train, self.y_train)
    predictions = pipeline.predict(x_test)
    return predictions


def string_to_list(corpus) : 
  simplified = remove_punctuation(corpus).split(" ")
  results = remove_stop_words(simplified)

  final_res = []
  for i in results : 
    if len(i)>0 : 
      final_res.append(i)
  return results

def remove_punctuation(corpus) : 
  chain = ""
  for i in range(len(corpus)) : 
    chain += corpus[i] if corpus[i] not in punc else " "
  return chain

def remove_stop_words(corpus) : 
  results = []
  for text in corpus : 
    tmp = text.split(' ')
    for stop_word in stop_words:
      if stop_word in tmp:
        tmp.remove(stop_word)
    results.append(" ".join(tmp))
        
  return results


# Test
import pan