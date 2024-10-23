from typing import Dict, List, Set
from util import Utterance


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(
            self, text: Utterance, index: int, embeddings
    ) -> None:
        self.text = text
        self.index = index
        self.embeddings = embeddings


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
            self, all_nodes, root_nodes, leaf_nodes, num_layers, layer_to_nodes
    ) -> None:
        self.all_nodes = all_nodes
        self.root_nodes = root_nodes
        self.leaf_nodes = leaf_nodes
        self.num_layers = num_layers
        self.layer_to_nodes = layer_to_nodes
