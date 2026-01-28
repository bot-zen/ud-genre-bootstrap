"""Evaluation metrics for clustering and bootstrapping."""

import logging
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


class ClusterQualityMetrics:
    """Compute unsupervised cluster quality metrics."""

    @staticmethod
    def compute_silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score.

        Higher is better (range: [-1, 1]).

        Args:
            embeddings: Sentence embeddings
            labels: Cluster labels

        Returns:
            Silhouette score
        """
        if len(np.unique(labels)) < 2:
            logger.warning("Need at least 2 clusters for silhouette score")
            return 0.0

        return float(silhouette_score(embeddings, labels))

    @staticmethod
    def compute_calinski_harabasz(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute Calinski-Harabasz score.

        Higher is better (no fixed range).

        Args:
            embeddings: Sentence embeddings
            labels: Cluster labels

        Returns:
            Calinski-Harabasz score
        """
        if len(np.unique(labels)) < 2:
            logger.warning("Need at least 2 clusters for Calinski-Harabasz score")
            return 0.0

        return float(calinski_harabasz_score(embeddings, labels))

    @staticmethod
    def compute_davies_bouldin(embeddings: np.ndarray, labels: np.ndarray) -> float:
        """Compute Davies-Bouldin index.

        Lower is better (>= 0).

        Args:
            embeddings: Sentence embeddings
            labels: Cluster labels

        Returns:
            Davies-Bouldin index
        """
        if len(np.unique(labels)) < 2:
            logger.warning("Need at least 2 clusters for Davies-Bouldin index")
            return float("inf")

        return float(davies_bouldin_score(embeddings, labels))

    @staticmethod
    def compute_pairwise_centroid_distances(
        embeddings: np.ndarray, labels: np.ndarray, metric: str = "cosine"
    ) -> Dict:
        """Compute pairwise distances between cluster centroids.

        This shows how well-separated different clusters are from each other.
        For example, "how far is the 'news' cluster from the 'social' cluster?"

        Args:
            embeddings: Sentence embeddings
            labels: Cluster labels
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Dictionary with:
                - pairwise_matrix: 2D list of distances [cluster_i][cluster_j]
                - cluster_ids: List of cluster IDs
                - mean_distance: Average pairwise distance
                - min_distance: Closest pair of clusters
                - max_distance: Furthest pair of clusters
        """
        from scipy.spatial.distance import cosine, euclidean

        unique_labels = sorted(np.unique(labels))
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            return {
                "pairwise_matrix": [],
                "cluster_ids": unique_labels.tolist(),
                "mean_distance": 0.0,
                "min_distance": 0.0,
                "max_distance": 0.0,
            }

        # Compute cluster centroids
        centroids = []
        for label in unique_labels:
            mask = labels == label
            centroid = embeddings[mask].mean(axis=0)
            centroids.append(centroid)

        # Compute pairwise distances
        distance_matrix = np.zeros((n_clusters, n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    if metric == "cosine":
                        dist = cosine(centroids[i], centroids[j])
                    elif metric == "euclidean":
                        dist = euclidean(centroids[i], centroids[j])
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    distance_matrix[i, j] = dist

        # Get upper triangle (excluding diagonal) for statistics
        upper_triangle = distance_matrix[np.triu_indices(n_clusters, k=1)]

        return {
            "pairwise_matrix": distance_matrix.tolist(),
            "cluster_ids": [int(label) for label in unique_labels],
            "mean_distance": float(upper_triangle.mean()) if len(upper_triangle) > 0 else 0.0,
            "min_distance": float(upper_triangle.min()) if len(upper_triangle) > 0 else 0.0,
            "max_distance": float(upper_triangle.max()) if len(upper_triangle) > 0 else 0.0,
            "metric": metric,
        }

    @classmethod
    def compute_all(
        cls, embeddings: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute all cluster quality metrics.

        Args:
            embeddings: Sentence embeddings
            labels: Cluster labels

        Returns:
            Dictionary of metric scores
        """
        metrics = {
            "silhouette": cls.compute_silhouette(embeddings, labels),
            "calinski_harabasz": cls.compute_calinski_harabasz(embeddings, labels),
            "davies_bouldin": cls.compute_davies_bouldin(embeddings, labels),
        }

        # Add pairwise centroid distances
        pairwise = cls.compute_pairwise_centroid_distances(embeddings, labels)
        metrics["pairwise_distances"] = pairwise

        return metrics


class GenreSeparationMetrics:
    """Compute genre-level separation metrics across treebanks."""

    @staticmethod
    def compute_genre_centroid_distances(
        genre_embeddings: Dict[str, np.ndarray], metric: str = "cosine"
    ) -> Dict:
        """Compute pairwise distances between genre centroids.

        Shows how separable different genres are in the embedding space.
        For example, "how far apart are 'news' and 'social' across all treebanks?"

        Args:
            genre_embeddings: Dictionary mapping genre names to their mean embeddings
            metric: Distance metric ('cosine' or 'euclidean')

        Returns:
            Dictionary with:
                - pairwise_matrix: 2D list of distances [genre_i][genre_j]
                - genres: List of genre names
                - mean_distance: Average pairwise distance
                - min_pair: Closest pair of genres
                - max_pair: Furthest pair of genres
        """
        from scipy.spatial.distance import cosine, euclidean

        genres = sorted(genre_embeddings.keys())
        n_genres = len(genres)

        if n_genres < 2:
            return {
                "pairwise_matrix": [],
                "genres": genres,
                "mean_distance": 0.0,
                "min_pair": None,
                "max_pair": None,
            }

        # Compute pairwise distances
        distance_matrix = np.zeros((n_genres, n_genres))
        for i, genre_i in enumerate(genres):
            for j, genre_j in enumerate(genres):
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    if metric == "cosine":
                        dist = cosine(genre_embeddings[genre_i], genre_embeddings[genre_j])
                    elif metric == "euclidean":
                        dist = euclidean(genre_embeddings[genre_i], genre_embeddings[genre_j])
                    else:
                        raise ValueError(f"Unknown metric: {metric}")
                    distance_matrix[i, j] = dist

        # Find min/max pairs (excluding diagonal)
        upper_triangle_indices = np.triu_indices(n_genres, k=1)
        upper_triangle = distance_matrix[upper_triangle_indices]

        min_idx = np.argmin(upper_triangle)
        max_idx = np.argmax(upper_triangle)

        # Convert flat index to 2D indices
        min_i, min_j = upper_triangle_indices[0][min_idx], upper_triangle_indices[1][min_idx]
        max_i, max_j = upper_triangle_indices[0][max_idx], upper_triangle_indices[1][max_idx]

        return {
            "pairwise_matrix": distance_matrix.tolist(),
            "genres": genres,
            "mean_distance": float(upper_triangle.mean()) if len(upper_triangle) > 0 else 0.0,
            "min_pair": (genres[min_i], genres[min_j], float(distance_matrix[min_i, min_j])),
            "max_pair": (genres[max_i], genres[max_j], float(distance_matrix[max_i, max_j])),
            "metric": metric,
        }


class BootstrapMetrics:
    """Compute bootstrap-specific metrics."""

    @staticmethod
    def resolution_rate(
        total_sentences: int,
        resolved_sentences: int,
    ) -> float:
        """Compute percentage of sentences that received genre labels.

        Args:
            total_sentences: Total number of sentences
            resolved_sentences: Number with genre labels

        Returns:
            Resolution rate [0, 1]
        """
        if total_sentences == 0:
            return 0.0

        return resolved_sentences / total_sentences

    @staticmethod
    def confidence_distribution(confidences: List[float]) -> Dict[str, float]:
        """Compute statistics of confidence scores.

        Args:
            confidences: List of confidence scores

        Returns:
            Dictionary with mean, median, std, min, max
        """
        if not confidences:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        arr = np.array(confidences)
        return {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }

    @staticmethod
    def method_distribution(methods: List[str]) -> Dict[str, int]:
        """Count how many sentences were labeled by each method.

        Args:
            methods: List of method labels

        Returns:
            Dictionary: {method: count}
        """
        from collections import Counter

        return dict(Counter(methods))

    @staticmethod
    def convergence_stats(
        schedule_length: int,
        disjunct_combinations: List,
        resolvable_genres: int,
        total_genres: int,
    ) -> Dict:
        """Compute bootstrap convergence statistics.

        Args:
            schedule_length: Number of environments in schedule
            disjunct_combinations: Unresolved genre combinations
            resolvable_genres: Number of genres that could be resolved
            total_genres: Total number of unique genres

        Returns:
            Dictionary with convergence statistics
        """
        return {
            "schedule_length": schedule_length,
            "num_disjunct": len(disjunct_combinations),
            "disjunct_combinations": [list(combo) for combo in disjunct_combinations],
            "resolvable_genres": resolvable_genres,
            "total_genres": total_genres,
            "genre_coverage": resolvable_genres / total_genres if total_genres > 0 else 0.0,
        }
