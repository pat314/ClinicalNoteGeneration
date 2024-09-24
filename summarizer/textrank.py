# -*- coding: utf-8 -*-

"""
Package: summarizer
Writer: Hoang-Thuy-Duong Vu
File: textrank.py
Project: CNG - Clinical Note Generation Ver.3
---
Current version: written on 23 sept. 2024
"""

# Import necessary global libraries
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
import matplotlib.pyplot as plt

class Graph: 
  """ 
  Purpose: Class for Graph initialization and construction
  """
  def __init__(self, sim_matrix):
    """
    Input:
    sim_matrix -- np.ndarray: similarity matrix calculated from ../retrieval/
    Obj: Initiate a blank graph with n=len(sim_mnatrix) nodes

    ---------------------------
    Ouput:
    None
    """
    self.G = nx.complete_graph(len(sim_matrix))
    self.sim_matrix = sim_matrix

  def _construct_from_sim(self): 
    """
    Input:
    None
    Obj: Add edges with weight corresponded to data from self.sim_matrix

    ---------------------------
    Ouput:
    None
    """
    thres = 0.3
    for i in range(len(self.sim_matrix)) : 
      for j in range(i, len(self.sim_matrix)) : 
        if self.sim_matrix[i][j] > thres: 
          self.G.add_edge(i+1, j+1, weight=self.sim_matrix[i][j])

  def _draw_graph(self):
    """
    Obj: Draw graph
    """
    G = self.G

    pos = nx.spring_layout(G, seed=7)  # positions for all nodes - seed for reproducibility

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # edges
    nx.draw_networkx_edges(
        G, pos, width=3, alpha=0.5, edge_color="b", style="dashed"
    )

    # node labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    # edge weight labels
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()