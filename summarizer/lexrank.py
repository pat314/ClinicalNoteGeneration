import numpy as np
import math
from collections import defaultdict
from typing import *


class LexRank:
    threshold = 0.01
    epsilon = 0.01

    def __init__(self, documents: List[str]):
        self.documents = documents
        self.idf_score = self.calculate_idf(documents)

    @staticmethod
    def tokenize(document: str) -> List[str]:
        return document.split()

    def calculate_tf(self, document: str) -> Dict[str, float]:
        tokens = self.tokenize(document)
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        total_tokens = len(tokens)
        return {word: count / total_tokens for word, count in tf.items()}

    def calculate_idf(self, documents: List[str]) -> Dict[str, float]:
        N = len(documents)
        idf = defaultdict(int)
        for doc in documents:
            tokens = set(self.tokenize(doc))
            for token in tokens:
                idf[token] += 1
        return {word: math.log(N / (count + 1)) for word, count in idf.items()}

    def idf_modified_cosine_similarity(self, tf_idf_x: Dict[str, float], tf_idf_y: Dict[str, float]) -> float:
        common_words = set(tf_idf_x.keys()) & set(tf_idf_y.keys())

        nominator = 0
        for word in common_words:
            idf = self.idf_score[word]
            nominator += tf_idf_x[word] * tf_idf_y[word] * idf ** 2

        if math.isclose(nominator, 0):
            return 0

        denominator_x = sum((tf_idf_x[word] * self.idf_score[word]) ** 2 for word in tf_idf_x)
        denominator_y = sum((tf_idf_y[word] * self.idf_score[word]) ** 2 for word in tf_idf_y)

        similarity = nominator / math.sqrt(denominator_x * denominator_y)

        return similarity

    def calculate_similarity_graph(self):
        length = len(self.documents)
        matrix = np.zeros((length, length))
        degrees = np.zeros((length), )

        for i in range(length):
            for j in range(i, length):
                tf_i = self.calculate_tf(self.documents[i])
                tf_j = self.calculate_tf(self.documents[j])
                similarity = self.idf_modified_cosine_similarity(tf_i, tf_j)
                matrix[i, j] = similarity

                if matrix[i, j] > self.threshold:
                    matrix[i, j] = 1.0
                    degrees[i] += 1
                else:
                    matrix[i, j] = 0.0

        for i in range(length):
            for j in range(length):
                if degrees[i] == 0:
                    degrees[i] = 1
                matrix[i][j] = matrix[i][j] / degrees[i]
        return matrix

    @staticmethod
    def power_method(matrix, epsilon: float):
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = np.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0

        while lambda_val > epsilon:
            next_p = np.dot(transposed_matrix, p_vector)
            lambda_val = np.linalg.norm(next_p - p_vector)
            p_vector = next_p

        return p_vector

    def summarize(self, top_n: int = 2) -> List[str]:
        similarity_matrix = self.calculate_similarity_graph()
        scores = self.power_method(similarity_matrix, self.epsilon)
        ranked_sentences = sorted(((score, sentence) for score, sentence in zip(scores, self.documents)), reverse=True)
        summary = [sentence for _, sentence in ranked_sentences[:top_n]]

        return summary
