import numpy as np
import logging
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import umap
from sentence_transformers import SentenceTransformer
from abc import ABC, abstractmethod
from typing import List, Optional
from RAPTOR_util import spacy_tokenize
from tree_structures import Node

RANDOM_SEED = 224
random.seed(RANDOM_SEED)

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    reduced_embeddings = umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def local_cluster_embeddings(
        embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    reduced_embeddings = umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)
    return reduced_embeddings


def get_optimal_clusters(
        embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
    optimal_clusters = n_clusters[np.argmin(bics)]
    return optimal_clusters


def GMM_cluster(
        embeddings: np.ndarray,
        threshold: float,
        n_clusters: Optional[int] = None,
        initial_means: Optional[np.ndarray] = None,
        random_state: int = RANDOM_SEED,
):
    if n_clusters is None:
        n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)

    # Set initial parameters if provided
    if initial_means is not None:
        gm.means_init = initial_means

    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]

    # Return GMM parameters
    return labels, n_clusters, gm


def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
        initial_means: Optional[np.ndarray] = None,
        initial_covariances: Optional[np.ndarray] = None,
        initial_weights: Optional[np.ndarray] = None,
        verbose: bool = False,
):
    reduced_embeddings_global = global_cluster_embeddings(
        embeddings, min(dim, len(embeddings) - 2)
    )

    # Pass initial GMM parameters
    global_clusters, n_global_clusters, gm_global = GMM_cluster(
        reduced_embeddings_global,
        threshold,
        n_clusters=len(initial_means) if initial_means is not None else None,
        initial_means=initial_means,
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0
    gmm_parameters = []  # To store GMM parameters for local clusters

    for i in range(n_global_clusters):
        # Get the boolean array indicating which embeddings belong to the current global cluster
        global_cluster_mask = np.array([i in gc for gc in global_clusters])
        # Get the actual indices of the nodes in this global cluster
        global_cluster_indices = np.where(global_cluster_mask)[0]
        global_cluster_embeddings_ = embeddings[global_cluster_indices]
        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )
        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
            gm_local = None  # No GMM for this small cluster
        else:
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            # Perform local clustering without initial parameters
            local_clusters, n_local_clusters, gm_local = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        if verbose:
            logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")

        # Store GMM parameters for local clusters
        gmm_parameters.append((gm_local, global_cluster_indices))

        # Assign clusters
        for j in range(n_local_clusters):
            local_cluster_indices = np.array([j in lc for lc in local_clusters])
            indices = np.where(global_cluster_indices)[0][local_cluster_indices]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters, gm_global, gmm_parameters


class ClusteringAlgorithm(ABC):
    @abstractmethod
    def perform_clustering(self, embeddings: np.ndarray, **kwargs) -> List[List[int]]:
        pass


class Layer_Clustering(ClusteringAlgorithm):
    def perform_clustering(
            self,
            nodes: List[Node],
            embedding_model_name: str,
            max_length_in_cluster: int = 3500,
            tokenizer=spacy_tokenize,
            reduction_dimension: int = 10,
            threshold: float = 0.1,
            initial_means: Optional[np.ndarray] = None,
            verbose: bool = False,
    ) -> List[List[Node]]:
        # Get the embeddings from the nodes
        embeddings = np.array(
            [node.embeddings[embedding_model_name] for node in nodes]
        )

        # Perform the clustering
        clusters, gm_global, gmm_parameters = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            initial_means=initial_means,
            verbose=verbose,
        )

        # Initialize an empty list to store the clusters of nodes
        node_clusters = []

        # Iterate over each unique label in the clusters
        for idx, label in enumerate(np.unique(np.concatenate(clusters))):
            # Get the indices of the nodes that belong to this cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]

            # Add the corresponding nodes to the node_clusters list
            cluster_nodes = [nodes[i] for i in indices]

            # Base case: if the cluster only has one node, do not attempt to recluster it
            if len(cluster_nodes) == 1:
                node_clusters.append(cluster_nodes)
                continue

            # Calculate the total length of the text in the nodes
            total_length = sum(
                [len(tokenizer(node.text)) for node in cluster_nodes]  # Tokenize using spaCy
            )

            # If the total length exceeds the maximum allowed length, recluster this cluster
            if total_length > max_length_in_cluster:
                if verbose:
                    logging.info(
                        f"reclustering cluster with {len(cluster_nodes)} nodes"
                    )
                node_clusters.extend(
                    self.perform_clustering(
                        cluster_nodes, embedding_model_name, max_length_in_cluster
                    )
                )
            else:
                node_clusters.append(cluster_nodes)

        return node_clusters
