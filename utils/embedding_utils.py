"""
Embedding Utilities: Helper functions for working with embeddings
"""

import numpy as np
from typing import List, Tuple, Optional
import pickle


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    L2 normalize embedding to unit vector

    Args:
        embedding: Input embedding

    Returns:
        Normalized embedding
    """
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2 normalize multiple embeddings

    Args:
        embeddings: Input embeddings (N, D)

    Returns:
        Normalized embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return embeddings / norms


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Similarity score (-1 to 1)
    """
    # Normalize
    emb1_norm = normalize_embedding(emb1)
    emb2_norm = normalize_embedding(emb2)

    # Dot product
    return float(np.dot(emb1_norm, emb2_norm))


def cosine_similarity_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)

    Returns:
        Similarity matrix (N, M)
    """
    # Normalize
    emb1_norm = normalize_embeddings(embeddings1)
    emb2_norm = normalize_embeddings(embeddings2)

    # Matrix multiplication
    return np.dot(emb1_norm, emb2_norm.T)


def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute Euclidean distance between two embeddings

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Distance
    """
    return float(np.linalg.norm(emb1 - emb2))


def euclidean_distance_matrix(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)

    Returns:
        Distance matrix (N, M)
    """
    # Expand dimensions for broadcasting
    emb1_expanded = embeddings1[:, np.newaxis, :]  # (N, 1, D)
    emb2_expanded = embeddings2[np.newaxis, :, :]  # (1, M, D)

    # Compute distances
    distances = np.sqrt(np.sum((emb1_expanded - emb2_expanded) ** 2, axis=2))

    return distances


def find_nearest_neighbors(query_embedding: np.ndarray,
                          candidate_embeddings: np.ndarray,
                          k: int = 5,
                          metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors

    Args:
        query_embedding: Query embedding (D,)
        candidate_embeddings: Candidate embeddings (N, D)
        k: Number of neighbors
        metric: Distance metric ("cosine" or "euclidean")

    Returns:
        Tuple of (indices, scores)
    """
    if metric == "cosine":
        # Compute similarities
        similarities = cosine_similarity_matrix(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]

    elif metric == "euclidean":
        # Compute distances
        distances = euclidean_distance_matrix(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]

        # Get top-k (smallest distances)
        top_k_indices = np.argsort(distances)[:k]
        top_k_scores = distances[top_k_indices]

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return top_k_indices, top_k_scores


def filter_by_threshold(embeddings: np.ndarray,
                       query_embedding: np.ndarray,
                       threshold: float,
                       metric: str = "cosine") -> List[int]:
    """
    Filter embeddings by similarity/distance threshold

    Args:
        embeddings: Candidate embeddings (N, D)
        query_embedding: Query embedding (D,)
        threshold: Threshold value
        metric: Distance metric

    Returns:
        List of indices that pass threshold
    """
    if metric == "cosine":
        similarities = cosine_similarity_matrix(
            query_embedding.reshape(1, -1),
            embeddings
        )[0]

        indices = np.where(similarities >= threshold)[0].tolist()

    elif metric == "euclidean":
        distances = euclidean_distance_matrix(
            query_embedding.reshape(1, -1),
            embeddings
        )[0]

        indices = np.where(distances <= threshold)[0].tolist()

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return indices


def average_embeddings(embeddings: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute weighted average of embeddings

    Args:
        embeddings: Input embeddings (N, D)
        weights: Optional weights (N,)

    Returns:
        Average embedding (D,)
    """
    if weights is None:
        return np.mean(embeddings, axis=0)
    else:
        weights = weights / np.sum(weights)  # Normalize weights
        return np.average(embeddings, axis=0, weights=weights)


def concatenate_embeddings(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate multiple embeddings

    Args:
        embeddings_list: List of embeddings

    Returns:
        Concatenated embedding
    """
    return np.concatenate(embeddings_list, axis=-1)


def save_embeddings(embeddings: np.ndarray, filepath: str):
    """
    Save embeddings to file

    Args:
        embeddings: Embeddings to save
        filepath: Output file path
    """
    try:
        np.save(filepath, embeddings)
    except Exception as e:
        print(f"Error saving embeddings: {e}")


def load_embeddings(filepath: str) -> Optional[np.ndarray]:
    """
    Load embeddings from file

    Args:
        filepath: Input file path

    Returns:
        Loaded embeddings or None
    """
    try:
        return np.load(filepath)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return None


def compute_centroid(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute centroid of embeddings

    Args:
        embeddings: Input embeddings (N, D)

    Returns:
        Centroid embedding (D,)
    """
    return np.mean(embeddings, axis=0)


def compute_variance(embeddings: np.ndarray) -> float:
    """
    Compute variance of embeddings

    Args:
        embeddings: Input embeddings (N, D)

    Returns:
        Variance score
    """
    centroid = compute_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)
    return float(np.var(distances))


def remove_outliers(embeddings: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, List[int]]:
    """
    Remove outlier embeddings based on distance from centroid

    Args:
        embeddings: Input embeddings (N, D)
        threshold: Z-score threshold

    Returns:
        Tuple of (filtered_embeddings, kept_indices)
    """
    centroid = compute_centroid(embeddings)
    distances = np.linalg.norm(embeddings - centroid, axis=1)

    # Compute z-scores
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if std_dist == 0:
        return embeddings, list(range(len(embeddings)))

    z_scores = (distances - mean_dist) / std_dist

    # Keep embeddings within threshold
    kept_indices = np.where(np.abs(z_scores) <= threshold)[0].tolist()
    filtered_embeddings = embeddings[kept_indices]

    return filtered_embeddings, kept_indices


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """
    Simple K-means clustering of embeddings

    Args:
        embeddings: Input embeddings (N, D)
        n_clusters: Number of clusters

    Returns:
        Cluster labels (N,)
    """
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels
    except ImportError:
        print("sklearn not installed, cannot perform clustering")
        return np.zeros(len(embeddings), dtype=int)


def reduce_dimensionality(embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Reduce dimensionality using PCA

    Args:
        embeddings: Input embeddings (N, D)
        n_components: Target dimensions

    Returns:
        Reduced embeddings (N, n_components)
    """
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        return reduced
    except ImportError:
        print("sklearn not installed, cannot perform dimensionality reduction")
        return embeddings
