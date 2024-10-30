import pandas as pd
from typing import List
from clustering.tree_structures import Node
from preprocess.text_embedding import bcbert_encoder


def create_node(
        file_path: str, 
        dialogue_column: str, 
        task='C', 
        exceptions: List[List[str]] = None
) -> List[List[Node]]:
    """
    Create a list of lists of Nodes, where each list represents all Nodes of a single sample.

    Args:
        file_path (str): Path to the CSV file containing dialogue data.
        dialogue_column (str): The name of the column in the DataFrame that contains the dialogue text.
        task (str): A task identifier; default is 'C'.
        exceptions (List[List[str]]): Optional list of [old_text, new_text] pairs for text replacements.

    Returns:
        List[List[Node]]: A nested list where each inner list contains Node objects for a single sample.
    """
    # Load CSV file into a DataFrame
    df = pd.read_csv(file_path)

    results = []

    for _, row in df.iterrows():
        # Extract and clean text from the specified dialogue column
        text = row[dialogue_column].replace('\r', '')

        # Apply exception replacements, if provided
        if exceptions:
            for old_text, new_text in exceptions:
                text = text.replace(old_text, new_text)

        # Split the processed text into utterances based on newlines
        utterances = text.split('\n')

        # Convert each utterance into a Node with its corresponding embedding
        nodes = [
            Node(text=utter, index=i, embeddings=bcbert_encoder(utter))
            for i, utter in enumerate(utterances)
        ]

        # Append the list of Nodes for the current row to the results
        results.append(nodes)

    return results


