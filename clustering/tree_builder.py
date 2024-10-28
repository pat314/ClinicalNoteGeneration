import pandas as pd
from typing import List
from tree_structures import Node
from preprocess.text_embedding import bcbert_encoder


def create_node(
        file_path: str, dialogue_column: str, task='C', exceptions: List[List[str]] = None
) -> List[List[Node]]:
    """
    Create List of List of Nodes in which each List represents all Nodes of a single sample
    """
    # Load CSV file
    df = pd.read_csv(file_path)

    results = []

    for _, row in df.iterrows():
        text = row[dialogue_column].replace('\r', '')

        # Apply exception replacements, if provided
        if exceptions:
            for old_text, new_text in exceptions:
                text = text.replace(old_text, new_text)

        # Split processed text into utterances
        utterances = text.split('\n')

        # Convert each utterance to a Node with embedding
        nodes = [Node(text=utter, embeddings=bcbert_encoder(utter)) for utter in utterances]

        # Append the list of Nodes for the current row
        results.append(nodes)

    return results

