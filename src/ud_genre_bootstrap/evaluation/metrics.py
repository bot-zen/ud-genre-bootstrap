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
        return {
            "silhouette": cls.compute_silhouette(embeddings, labels),
            "calinski_harabasz": cls.compute_calinski_harabasz(embeddings, labels),
            "davies_bouldin": cls.compute_davies_bouldin(embeddings, labels),
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
