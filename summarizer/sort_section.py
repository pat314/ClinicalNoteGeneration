# import os
# from os import listdir
# parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))  # Get the parent directory
# os.chdir(parent_dir)  # Change the working directory to the parent directory
# os.sys.path.append(parent_dir)  # Append the parent directory to the system path
# # listdir()

import numpy as np 
import pandas as pd 

from preprocess.text_embedding import bcbert_encoder


def cossim(A, B) : 
    return np.dot(A,B).sum() / (np.sqrt((A**2).sum()) * np.sqrt((B**2).sum()))

# def embed(point, Utt) : 
#     embedded_pt = bcbert_encoder(point)[0]
#     embedded_Utt = bcbert_encoder(Utt)[0]
#     return embedded_pt, embedded_Utt

def local_sort(file_path, dialogue_column, idx, nodes,thres) : 
    df = pd.read_csv(file_path)
    col = df.columns.lower()

    node = nodes[idx]

    results = [[] for _ in range(len(col))]

    for i in range(len(node)) : 
        l = np.array([cossim(bcbert_encoder(col[j])[0],node[i].embeddings[0]) for j in range(len(col))])

        index = [i for i in range(len(l)) if l[i] >= thres]
        for j in range(len(index)) : 
            results[j].append(node[i])

    return results

def sort(file_path, dialogue_column,nodes,thres) : 
    df = pd.read_csv(file_path)
    col = df.columns.lower()

    utt = df[dialogue_column][id]
    sentences = utt.split("\n")

    results = [local_sort(file_path, dialogue_column, i, nodes,thres) for i in range(len(nodes))]

    return results

def cluster_center(cluster) : 
    return np.mean(cluster)

    


    # for i in range(len(col)) : 
    #     embedded_col = bcbert_encoder(col[i])

    #     l = [cossim(embedded_pt,embedded_Utt) for i in range(len(list_section))]

