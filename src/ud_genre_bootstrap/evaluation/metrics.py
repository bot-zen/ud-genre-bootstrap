"""Evaluation metrics for clustering and bootstrapping."""

import logging
from collections import Counter, defaultdict
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
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


class ClusteringEvaluationMetrics:
    """Compute paper-aligned metrics for clustering evaluation."""

    @staticmethod
    def _validate_lengths(
        true_genres: Sequence[str],
        pred_genres: Sequence[str],
        treebank_keys: Optional[Sequence[Hashable]] = None,
    ) -> None:
        if len(true_genres) != len(pred_genres):
            raise ValueError(
                f"Length mismatch: {len(true_genres)} true labels vs {len(pred_genres)} predictions"
            )
        if treebank_keys is not None and len(treebank_keys) != len(true_genres):
            raise ValueError(
                "Length mismatch: treebank keys must match number of labels "
                f"({len(treebank_keys)} vs {len(true_genres)})"
            )

    @staticmethod
    def _format_treebank_key(treebank_key: Hashable) -> str:
        if isinstance(treebank_key, tuple) and len(treebank_key) == 2:
            tb_code, split_name = treebank_key
            return f"{tb_code}:{split_name}"
        return str(treebank_key)

    @staticmethod
    def _distribution(labels: Sequence[str]) -> Dict[str, float]:
        if not labels:
            return {}
        counts = Counter(labels)
        total = float(sum(counts.values()))
        return {label: count / total for label, count in counts.items()}

    @staticmethod
    def _bhattacharyya_coefficient(
        p_dist: Dict[str, float],
        q_dist: Dict[str, float],
    ) -> float:
        labels = set(p_dist.keys()) | set(q_dist.keys())
        if not labels:
            return 0.0
        return float(sum(np.sqrt(p_dist.get(label, 0.0) * q_dist.get(label, 0.0)) for label in labels))

    @staticmethod
    def _majority_label(labels: Sequence[str]) -> Optional[str]:
        if not labels:
            return None
        counts = Counter(labels)
        # Deterministic tie-break for stable tests/output.
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    @classmethod
    def compute_purity(
        cls,
        true_genres: Sequence[str],
        pred_genres: Sequence[str],
    ) -> float:
        """Standard cluster purity on predicted-label groups."""
        cls._validate_lengths(true_genres, pred_genres)
        if not true_genres:
            return 0.0

        contingency: Dict[str, Counter] = defaultdict(Counter)
        for true_label, pred_label in zip(true_genres, pred_genres):
            contingency[pred_label][true_label] += 1

        dominant_total = sum(max(counter.values()) for counter in contingency.values() if counter)
        return float(dominant_total / len(true_genres))

    @classmethod
    def compute_overlap_error(
        cls,
        true_genres: Sequence[str],
        pred_genres: Sequence[str],
        treebank_keys: Sequence[Hashable],
    ) -> Dict[str, object]:
        """Compute inverse Bhattacharyya overlap (ΔBC) per treebank and aggregated."""
        cls._validate_lengths(true_genres, pred_genres, treebank_keys)
        if not true_genres:
            return {
                "overlap_error": 0.0,
                "overlap_error_weighted": 0.0,
                "overlap_error_by_treebank": {},
            }

        per_treebank_indices: Dict[Hashable, List[int]] = defaultdict(list)
        for idx, treebank_key in enumerate(treebank_keys):
            per_treebank_indices[treebank_key].append(idx)

        overlap_errors: List[float] = []
        weighted_numer = 0.0
        weighted_denom = 0
        by_treebank: Dict[str, float] = {}

        for treebank_key, indices in per_treebank_indices.items():
            tb_true = [true_genres[i] for i in indices]
            tb_pred = [pred_genres[i] for i in indices]
            true_dist = cls._distribution(tb_true)
            pred_dist = cls._distribution(tb_pred)
            bc = cls._bhattacharyya_coefficient(true_dist, pred_dist)
            overlap_error = float(1.0 - bc)
            overlap_errors.append(overlap_error)
            weighted_numer += overlap_error * len(indices)
            weighted_denom += len(indices)
            by_treebank[cls._format_treebank_key(treebank_key)] = overlap_error

        return {
            "overlap_error": float(np.mean(overlap_errors)) if overlap_errors else 0.0,
            "overlap_error_weighted": float(weighted_numer / weighted_denom) if weighted_denom else 0.0,
            "overlap_error_by_treebank": by_treebank,
        }

    @classmethod
    def compute_agreement(
        cls,
        true_genres: Sequence[str],
        pred_genres: Sequence[str],
        treebank_keys: Sequence[Hashable],
    ) -> Dict[str, object]:
        """Compute cross-treebank label agreement (AGR)."""
        cls._validate_lengths(true_genres, pred_genres, treebank_keys)
        if not true_genres:
            return {"agreement": 0.0, "agreement_by_genre": {}}

        grouped: Dict[Hashable, List[Tuple[str, str]]] = defaultdict(list)
        for treebank_key, true_label, pred_label in zip(treebank_keys, true_genres, pred_genres):
            grouped[treebank_key].append((true_label, pred_label))

        per_genre_majorities: Dict[str, List[str]] = defaultdict(list)
        for samples in grouped.values():
            preds_by_true_genre: Dict[str, List[str]] = defaultdict(list)
            for true_label, pred_label in samples:
                preds_by_true_genre[true_label].append(pred_label)
            for true_label, predictions in preds_by_true_genre.items():
                majority = cls._majority_label(predictions)
                if majority is not None:
                    per_genre_majorities[true_label].append(majority)

        agreement_by_genre: Dict[str, float] = {}
        weighted_numer = 0.0
        weighted_denom = 0

        for true_label, majority_labels in per_genre_majorities.items():
            label_counts = Counter(majority_labels)
            agreement = max(label_counts.values()) / len(majority_labels)
            agreement_by_genre[true_label] = float(agreement)
            weighted_numer += agreement * len(majority_labels)
            weighted_denom += len(majority_labels)

        return {
            "agreement": float(weighted_numer / weighted_denom) if weighted_denom else 0.0,
            "agreement_by_genre": agreement_by_genre,
        }

    @classmethod
    def compute_all(
        cls,
        true_genres: Sequence[str],
        pred_genres: Sequence[str],
        treebank_keys: Sequence[Hashable],
    ) -> Dict[str, object]:
        """Compute full clustering evaluation metric bundle."""
        cls._validate_lengths(true_genres, pred_genres, treebank_keys)
        if not true_genres:
            return {
                "purity": 0.0,
                "agreement": 0.0,
                "agreement_by_genre": {},
                "overlap_error": 0.0,
                "overlap_error_weighted": 0.0,
                "overlap_error_by_treebank": {},
                "micro_f1_instance": 0.0,
                "macro_f1_instance": 0.0,
                "instance_labeled_treebanks": 0,
            }

        overlap = cls.compute_overlap_error(true_genres, pred_genres, treebank_keys)
        agreement = cls.compute_agreement(true_genres, pred_genres, treebank_keys)

        return {
            "purity": cls.compute_purity(true_genres, pred_genres),
            "agreement": agreement["agreement"],
            "agreement_by_genre": agreement["agreement_by_genre"],
            "overlap_error": overlap["overlap_error"],
            "overlap_error_weighted": overlap["overlap_error_weighted"],
            "overlap_error_by_treebank": overlap["overlap_error_by_treebank"],
            "micro_f1_instance": float(
                f1_score(true_genres, pred_genres, average="micro", zero_division=0)
            ),
            "macro_f1_instance": float(
                f1_score(true_genres, pred_genres, average="macro", zero_division=0)
            ),
            "instance_labeled_treebanks": len(set(treebank_keys)),
        }


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
