import numpy as np
import logging
import random
from sklearn.mixture import GaussianMixture
import umap
from typing import List, Optional
from RAPTOR_util import spacy_tokenize
from tree_structures import Node

RANDOM_SEED = 224
random.seed(RANDOM_SEED)

# Initialize logging
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


def get_optimal_clusters(
        embeddings: np.ndarray, 
        max_clusters: int = 50, 
        random_state: int = RANDOM_SEED
) -> int:
    """
    Determines the optimal number of clusters for Gaussian Mixture Model (GMM) clustering using the Bayesian Information Criterion (BIC).

    Args:
        embeddings (np.ndarray): Array of embeddings for which clusters need to be identified.
        max_clusters (int): Maximum number of clusters to test. Defaults to 50.
        random_state (int): Random seed for reproducibility of results. Defaults to a predefined constant RANDOM_SEED.

    Returns:
        int: The optimal number of clusters determined by the BIC criterion.
    """
    # Ensure the maximum number of clusters does not exceed the number of embeddings
    max_clusters = min(max_clusters, len(embeddings))

    # Create an array of possible cluster numbers from 1 to max_clusters - 1
    n_clusters = np.arange(1, max_clusters)

    bics = []  # List to store BIC values for each cluster count
    for n in n_clusters:
        # Initialize Gaussian Mixture Model with n components
        gm = GaussianMixture(n_components=n, random_state=random_state)
        
        # Fit the model to the embeddings
        gm.fit(embeddings)
        
        # Calculate and store the BIC for the current model
        bics.append(gm.bic(embeddings))

    # Determine the optimal number of clusters as the one with the lowest BIC
    optimal_clusters = n_clusters[np.argmin(bics)]
    
    return optimal_clusters  # Return the optimal number of clusters



def GMM_cluster(
        embeddings: np.ndarray,
        threshold: float,
        n_clusters: Optional[int] = None,
        initial_means: Optional[np.ndarray] = None,
        random_state: int = RANDOM_SEED,
):
    """
    Performs clustering on the provided embeddings using Gaussian Mixture Model (GMM).

    Args:
        embeddings (np.ndarray): Array of embeddings to be clustered.
        threshold (float): Probability threshold for assigning labels to clusters.
        n_clusters (Optional[int]): Number of clusters to form. If None, optimal clusters will be determined.
        initial_means (Optional[np.ndarray]): Initial means for the clusters, if provided.
        random_state (int): Random seed for reproducibility of results. Defaults to a predefined constant RANDOM_SEED.

    Returns:
        Tuple[List[np.ndarray], int]: A tuple containing:
            - List of labels for each embedding based on the probability threshold.
            - The number of clusters used in the GMM.
    """
    # Determine the optimal number of clusters if not provided
    if n_clusters is None:
        n_clusters = get_optimal_clusters(embeddings)

    # Initialize Gaussian Mixture Model with the specified number of components
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)

    # Set initial cluster means if provided
    if initial_means is not None:
        gm.means_init = initial_means

    # Fit the GMM to the embeddings
    gm.fit(embeddings)

    # Get the predicted probabilities of each embedding belonging to each cluster
    probs = gm.predict_proba(embeddings)

    # Create labels based on the provided threshold
    labels = [np.where(prob > threshold)[0] for prob in probs]

    # Return the labels and the number of clusters used
    return labels, n_clusters



def perform_clustering(
        embeddings: np.ndarray,
        dim: int,
        threshold: float,
        initial_means: Optional[np.ndarray] = None,
        verbose: bool = False,
) -> List[np.ndarray]:
    """
    Perform clustering on the provided embeddings using Gaussian Mixture Model (GMM).

    Args:
        embeddings (np.ndarray): Array of embeddings to be clustered.
        dim (int): The dimensionality of the data, used to determine clustering feasibility.
        threshold (float): Probability threshold for assigning labels to clusters.
        initial_means (Optional[np.ndarray]): Initial means for the clusters, if provided.
        verbose (bool): Flag to enable verbose logging for debugging purposes.

    Returns:
        List[np.ndarray]: A list of arrays where each array contains the cluster indices for the corresponding embedding.
    """
    # Check if the number of data points is less than or equal to the dimension + 1
    # If true, clustering is not effective, so return a list of zeros
    if len(embeddings) <= dim + 1:
        return [np.array([0]) for _ in range(len(embeddings))]

    # Perform clustering using GMM with the provided embeddings and parameters
    global_clusters, n_global_clusters = GMM_cluster(
        embeddings,
        threshold,
        n_clusters=len(initial_means) if initial_means is not None else None,
        initial_means=initial_means,
    )

    if verbose:
        logging.info(f"Global Clusters: {n_global_clusters}")

    # Initialize a list to hold clusters for all embeddings
    all_clusters = [np.array([]) for _ in range(len(embeddings))]

    # Iterate through each global cluster to gather embeddings
    for i in range(n_global_clusters):
        # Extract all embeddings that belong to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if verbose:
            logging.info(
                f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
            )

        # Find the indices of the original embeddings that belong to the current global cluster
        indices = np.where(
            (embeddings == global_cluster_embeddings_[:, None]).all(-1)
        )[1]

        # Mark each embedding with the corresponding global cluster index
        for idx in indices:
            all_clusters[idx] = np.append(
                all_clusters[idx], i
            )
    
    return all_clusters



def perform_full_clustering(
        samples: List[List[Node]],
        max_length_in_cluster: int = 3500,
        tokenizer=spacy_tokenize,
        reduction_dimension: int = 10,
        threshold: float = 0.4,
        initial_means: Optional[np.ndarray] = None,
        verbose: bool = False,
        prev_length=None,
) -> List[List[List[Node]]]:
    """
    Perform clustering on a list of samples, where each sample contains a list of nodes.

    Args:
        samples (List[List[Node]]): A list of samples, each containing a list of Node objects.
        max_length_in_cluster (int): Maximum length allowed for clusters (not currently used in the implementation).
        tokenizer: Tokenizer function to process text data (default is spacy_tokenize).
        reduction_dimension (int): Dimensionality reduction parameter for clustering.
        threshold (float): Probability threshold for assigning labels to clusters.
        initial_means (Optional[np.ndarray]): Initial means for the clusters, if provided.
        verbose (bool): Flag to enable verbose logging for debugging purposes.
        prev_length: Previous length (not currently used in the implementation).

    Returns:
        List[List[List[Node]]]: A nested list where each element contains the clustered nodes for each sample.
    """
    sample_clusters = []

    # Process each sample independently
    for sample in samples:
        # Get embeddings for nodes in the current sample
        embeddings = np.array([node.embeddings for node in sample])

        # Fix the shape of embeddings to ensure it has the correct dimensions
        if embeddings.ndim == 3 and embeddings.shape[1] == 1:
            embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[2])

        # Perform global clustering on the current sample's embeddings
        clusters = perform_clustering(
            embeddings,
            dim=reduction_dimension,
            threshold=threshold,
            initial_means=initial_means,
            verbose=verbose
        )

        # Collect clusters of nodes for the current sample
        node_clusters = []
        # Iterate over each unique label in the clusters
        for label in np.unique(np.concatenate(clusters)):
            # Get indices of nodes in the current sample that belong to this global cluster
            indices = [i for i, cluster in enumerate(clusters) if label in cluster]
            cluster_nodes = [sample[i] for i in indices]
            node_clusters.append(cluster_nodes)

        # Add clusters for this sample to the final result
        sample_clusters.append(node_clusters)

    return sample_clusters



# def global_cluster_embeddings(
#         embeddings: np.ndarray,
#         dim: int,
#         n_neighbors: Optional[int] = None,
#         metric: str = "cosine",
# ) -> np.ndarray:
#     if n_neighbors is None:
#         n_neighbors = int((len(embeddings) - 1) ** 0.5)
#     reduced_embeddings = umap.UMAP(
#         n_neighbors=n_neighbors, n_components=dim, metric=metric
#     ).fit_transform(embeddings)
#     return reduced_embeddings

# def local_cluster_embeddings(
#         embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
# ) -> np.ndarray:
#     reduced_embeddings = umap.UMAP(
#         n_neighbors=num_neighbors, n_components=dim, metric=metric
#     ).fit_transform(embeddings)
#     return reduced_embeddings

# def perform_clustering(
#         embeddings: np.ndarray,
#         dim: int,
#         threshold: float,
#         initial_means: Optional[np.ndarray] = None,
#         verbose: bool = False,
# ) -> List[np.ndarray]:
#     # if data points are smaller than its dimension then clustering is not effective
#     if len(embeddings) <= dim + 1:
#         return [np.array([0]) for _ in range(len(embeddings))]
#
#     reduced_embeddings_global = global_cluster_embeddings(
#         embeddings, min(dim, len(embeddings) - 2)
#     )
#
#     # Pass initial GMM parameters
#     global_clusters, n_global_clusters = GMM_cluster(
#         reduced_embeddings_global,
#         threshold,
#         n_clusters=len(initial_means) if initial_means is not None else None,
#         initial_means=initial_means,
#     )
#
#     if verbose:
#         logging.info(f"Global Clusters: {n_global_clusters}")
#
#     all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
#     total_clusters = 0
#
#     for i in range(n_global_clusters):
#         global_cluster_embeddings_ = embeddings[
#             np.array([i in gc for gc in global_clusters])
#         ]
#
#         if verbose:
#             logging.info(
#                 f"Nodes in Global Cluster {i}: {len(global_cluster_embeddings_)}"
#             )
#         if len(global_cluster_embeddings_) == 0:
#             continue
#         if len(global_cluster_embeddings_) <= dim + 1:
#             local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
#             n_local_clusters = 1
#         else:
#             reduced_embeddings_local = local_cluster_embeddings(
#                 global_cluster_embeddings_, dim
#             )
#             # Perform local clustering without initial parameters
#             local_clusters, n_local_clusters = GMM_cluster(
#                 reduced_embeddings_local, threshold
#             )
#
#         if verbose:
#             logging.info(f"Local Clusters in Global Cluster {i}: {n_local_clusters}")
#
#         for j in range(n_local_clusters):
#             local_cluster_embeddings_ = global_cluster_embeddings_[
#                 np.array([j in lc for lc in local_clusters])
#             ]
#             indices = np.where(
#                 (embeddings == local_cluster_embeddings_[:, None]).all(-1)
#             )[1]
#             for idx in indices:
#                 all_local_clusters[idx] = np.append(
#                     all_local_clusters[idx], j + total_clusters
#                 )
#
#         total_clusters += n_local_clusters
#
#         if verbose:
#             logging.info(f"Total Clusters: {total_clusters}")
#         return all_local_clusters
#
#     def perform_full_clustering(
#             nodes: List[Node],
#             embedding_model_name: str,
#             max_length_in_cluster: int = 3500,
#             tokenizer=spacy_tokenize,
#             reduction_dimension: int = 10,
#             threshold: float = 0.1,
#             initial_means: Optional[np.ndarray] = None,
#             verbose: bool = False,
#             prev_length=None,
#     ) -> List[List[Node]]:
#
#         # Get the embeddings from the nodes
#         embeddings = np.array(
#             [node.embeddings[embedding_model_name] for node in nodes]
#         )
#
#         # Perform the clustering
#         clusters, gm_global, gmm_parameters = perform_clustering(
#             embeddings,
#             dim=reduction_dimension,
#             threshold=threshold,
#             initial_means=initial_means,
#             verbose=verbose,
#         )
#
#         # Initialize an empty list to store the clusters of nodes
#         node_clusters = []
#
#         # Iterate over each unique label in the clusters
#         for label in np.unique(np.concatenate(clusters)):
#             # Get the indices of the nodes that belong to this cluster
#             indices = [i for i, cluster in enumerate(clusters) if label in cluster]
#
#             # Add the corresponding nodes to the node_clusters list
#             cluster_nodes = [nodes[i] for i in indices]
#
#             # Base case: if the cluster only has one node, do not attempt to recluster it
#             if len(cluster_nodes) == 1:
#                 node_clusters.append(cluster_nodes)
#                 continue
#
#             # Calculate the total length of the text in the nodes
#             total_length = sum(
#                 [len(tokenizer(node.text)) for node in cluster_nodes]  # Tokenize using spaCy
#             )
#
#             # If the total length exceeds the maximum allowed length, recluster this cluster
#             # If total length did not change, then do not recluster
#             if total_length > max_length_in_cluster and (prev_length is None or total_length < prev_length):
#                 if verbose:
#                     logging.info(
#                         f"reclustering cluster with {len(cluster_nodes)} nodes"
#                     )
#                 node_clusters.extend(
#                     perform_full_clustering(
#                         cluster_nodes,
#                         embedding_model_name,
#                         max_length_in_cluster,
#                         tokenizer=tokenizer,
#                         reduction_dimension=reduction_dimension,
#                         threshold=threshold,
#                         prev_length=total_length
#                     )
#                 )
#             else:
#                 node_clusters.append(cluster_nodes)
#
#         return node_clusters
