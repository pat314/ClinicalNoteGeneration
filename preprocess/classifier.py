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
import pandas as pd 
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

# String punctuation
import string
punc = string.punctuation

from stop_words import get_stop_words
stop_words = get_stop_words('en')




# CLASS: CLASSIFIER
class ClassifierMultiOAA: 

  """
  Best accuracy: Perception and SGDClassifier
  
  ClassifierMultiOOA is a flexible text classifier that allows users to choose from various machine learning algorithms 
  for classification tasks, integrated with a TF-IDF vectorizer for text preprocessing.

  Supported Classifiers:
  - Logistic Regression
  - Perceptron
  - SGD Classifier
  - K-Nearest Neighbors (KNN)
  - Multinomial Naive Bayes
  - Decision Tree Classifier
  - Support Vector Classifier (SVC)
  - Multi-layer Perceptron (MLP)
  """

  def __init__(self, 
              method: callable, 
              x_train: pd.core.series.Series, 
              y_train: pd.core.series.Series,
              alpha: float = 0.0001, 
              max_iter: int = 1000,
              epsilon: float = 0.1,
              learning_rate: str = 'optimal',
              loss: str = 'hinge',
              penalty: str = 'l2',
              n_neighbors: int = 5,
              weights: str = 'uniform',
              activation: str = 'relu',
              learning_rate_init: float = 0.001,learning_rate_MLP: str = "adaptive"): 
    """
    Initializes the classifier with:
    
    Input:
    - method: Callable (A classification algorithm function, e.g., LogisticRegression, SVC, etc.).
    - x_train: List[str] (A list of strings containing the training data, i.e., textual data).
    - y_train: List[Any] (A list containing the corresponding labels for the training data).

    Output:
    - None (the function initializes the instance variables for later use).
    """
    # Initialize x_train and y_train
    self.x_train = x_train
    self.y_train = y_train

    # Initialize method
    if method=="LogisticRegression": 
      self.method = LogisticRegression()
      
    elif method=="Perceptron": 
      self.method = Perceptron(alpha=alpha, max_iter=max_iter)

    elif method=="SGDClassifier": 
      self.method = SGDClassifier(
        loss=loss,penalty=penalty, 
        alpha=alpha, max_iter=max_iter,
        epsilon=epsilon,learning_rate=learning_rate)

    elif method=="KNeighborsClassifier":
      self.method = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    elif method=="MultinomialNB":
      self.method = MultinomialNB(alpha=alpha)

    elif method=="DecisionTreeClassifier":
      self.method = DecisionTreeClassifier()

    elif method=="SVC":
      self.method = SVC()

    else: 
      self.method = MLPClassifier(activation=activation,learning_rate=learning_rate_MLP,alpha=alpha,learning_rate_init=learning_rate_init,max_iter=max_iter,epsilon=epsilon)


  def predict(self, x_test: pd.core.series.Series):
    """
    Predicts the labels for the test dataset.
    
    Input:
    - x_test: List[str] (A list of strings containing the test data, i.e., textual data to classify).
    
    Output:
    - predictions: List[Any] (A list of predicted labels for the test data).
    
    The method uses a pipeline with two stages:
    1. TF-IDF Vectorizer for converting text into a numerical format (using the 'string_to_list' function to preprocess).
    2. The classification method provided during initialization, which is trained on the x_train and y_train data.
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer=string_to_list)), 
        ('classifier', self.method)
    ])
    pipeline.fit(self.x_train, self.y_train)
    predictions = pipeline.predict(x_test)
    return predictions

  def accuracy(self, x_test: pd.core.series.Series, y_test: pd.core.series.Series):
    """
    Calculates the accuracy of the classifier on the test dataset.
    
    Input:
    - x_test: pd.core.series.Series (Pandas Series containing the test data, which are textual inputs).
    - y_test: pd.core.series.Series (Pandas Series containing the true labels for the test data).
    
    Output:
    - accuracy: float (The ratio of correctly predicted labels to the total number of test instances).
    """
    pred = [self.predict([x_test[i]]) for i in range(len(x_test))]
    good_rate = 0
    for i in range(len(x_test)) : 
      if pred[i] == y_test[i] :
        good_rate += 1
    return good_rate / len(x_test)




# SIDE FUNCTION
def string_to_list(corpus) : 
  """
  Converts input text into a list of simplified words by removing punctuation and stop words.
  
  Input:
  - corpus: str (A string containing the text data, i.e., a single document or sentence).
  
  Output:
  - final_res: List[str] (A list of cleaned tokens (words) after removing punctuation and stop words).
  """
  simplified = remove_punctuation(corpus).split(" ")
  results = remove_stop_words(simplified)

  final_res = []
  for i in results : 
    if len(i)>0 : 
      final_res.append(i)
  return results

def remove_punctuation(corpus) : 
  """
  Removes punctuation from the input text.
  
  Input:
  - corpus: str (A string containing the text data).
  
  Output:
  - chain: str (A string with all punctuation replaced by spaces).
  """
  chain = ""
  for i in range(len(corpus)) : 
    chain += corpus[i] if corpus[i] not in punc else " "
  return chain

def remove_stop_words(corpus) : 
  """
  Removes stop words from the input list of words.
  
  Input:
  - corpus: List[str] (A list of words (tokens) from which stop words need to be removed).
  
  Output:
  - results: List[str] (A list of strings with stop words removed (each string corresponds to a cleaned sentence)).
  """
  results = []
  for text in corpus : 
    tmp = text.split(' ')
    for stop_word in stop_words:
      if stop_word in tmp:
        tmp.remove(stop_word)
    results.append(" ".join(tmp))
        
  return results