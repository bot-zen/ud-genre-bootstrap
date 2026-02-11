"""Shared clustering operations for production and evaluation pipelines."""

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance

logger = logging.getLogger(__name__)


class ClusteringOperations:
    """Shared clustering operations used by both production and evaluation.

    This class provides common operations for:
    - Grouping treebank splits
    - Combining embeddings across splits
    - Creating virtual splits from sentence metadata
    - Computing cluster centroids
    - Building reference embeddings
    - Labeling clusters with confidence thresholds

    By sharing this code, production and evaluation are guaranteed to stay consistent.
    """

    def __init__(self, min_confidence: float = 0.8):
        """Initialize clustering operations.

        Args:
            min_confidence: Minimum confidence threshold for high-confidence labeling
        """
        self.min_confidence = min_confidence

    def group_splits_by_treebank(
        self,
        treebank_keys: List[Tuple[str, str]],
        embeddings_by_tb: Dict,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """Group treebank splits by treebank code.

        Combines train/dev/test splits of the same treebank together.

        Args:
            treebank_keys: List of (treebank_code, split) tuples
            embeddings_by_tb: Dict mapping (treebank_code, split) -> embedding data

        Returns:
            Dict mapping treebank_code -> List of (treebank_code, split) tuples

        Example:
            Input: [(cs_pdtc, train), (cs_pdtc, dev), (de_pud, test)]
            Output: {
                'cs_pdtc': [(cs_pdtc, train), (cs_pdtc, dev)],
                'de_pud': [(de_pud, test)]
            }
        """
        treebank_groups = defaultdict(list)
        for tb_key in treebank_keys:
            if tb_key in embeddings_by_tb:
                tb_code = tb_key[0]
                treebank_groups[tb_code].append(tb_key)
        return dict(treebank_groups)

    def combine_treebank_splits(
        self,
        tb_keys: List[Tuple[str, str]],
        embeddings_by_tb: Dict,
    ) -> Tuple[np.ndarray, List[str], Dict[str, str]]:
        """Combine embeddings from all splits of a treebank.

        This is the core operation for treebank-level clustering.

        Args:
            tb_keys: List of (treebank_code, split) tuples for this treebank
            embeddings_by_tb: Dict mapping (treebank_code, split) -> embedding data

        Returns:
            Tuple of:
                - combined_embeddings: Stacked embeddings from all splits
                - all_sent_ids: List of all sentence IDs
                - sent_id_to_split: Dict mapping sentence ID -> split name
        """
        all_embeddings = []
        all_sent_ids = []
        sent_id_to_split = {}

        for tb_key in tb_keys:
            if tb_key not in embeddings_by_tb:
                logger.warning(f"No embeddings found for {tb_key[0]}:{tb_key[1]}")
                continue

            emb_data = embeddings_by_tb[tb_key]
            all_embeddings.append(emb_data['embedding'])
            all_sent_ids.extend(emb_data['sent_id'])

            for sent_id in emb_data['sent_id']:
                sent_id_to_split[sent_id] = tb_key[1]

        if len(all_embeddings) == 0:
            raise ValueError(f"No embeddings found for any splits of {tb_keys[0][0]}")

        combined_embeddings = np.vstack(all_embeddings)
        return combined_embeddings, all_sent_ids, sent_id_to_split

    def create_virtual_splits(
        self,
        tb_code: str,
        combined_embeddings: np.ndarray,
        all_sent_ids: List[str],
        sent_id_to_split: Dict[str, str],
        sentence_metadata: Dict[Tuple[str, str, str], str],
    ) -> Dict[str, Dict]:
        """Create virtual splits from combined embeddings using sentence metadata.

        Groups sentences by genre to create single-genre virtual splits.

        Args:
            tb_code: Treebank code
            combined_embeddings: Combined embeddings from all splits
            all_sent_ids: All sentence IDs
            sent_id_to_split: Mapping of sentence ID to split name
            sentence_metadata: Dict mapping (tb_code, split, sent_id) -> genre

        Returns:
            Dict mapping genre -> {
                'embeddings': np.ndarray of genre embeddings,
                'sent_ids': List of sentence IDs,
                'indices': List of indices in combined_embeddings,
                'split_distribution': Dict mapping split -> count
            }
        """
        genre_indices = defaultdict(list)
        genre_sent_ids = defaultdict(list)

        # Group sentences by genre
        for sent_idx, sent_id in enumerate(all_sent_ids):
            split_name = sent_id_to_split[sent_id]
            key = (tb_code, split_name, sent_id)
            if key in sentence_metadata:
                genre = sentence_metadata[key]
                genre_indices[genre].append(sent_idx)
                genre_sent_ids[genre].append(sent_id)

        # Build virtual split data structures
        virtual_splits = {}
        for genre, indices in genre_indices.items():
            if len(indices) > 0:
                genre_embeddings = combined_embeddings[indices]
                sent_ids = genre_sent_ids[genre]

                # Track distribution across splits
                split_distribution = defaultdict(int)
                for sent_id in sent_ids:
                    split_distribution[sent_id_to_split[sent_id]] += 1

                virtual_splits[genre] = {
                    'embeddings': genre_embeddings,
                    'sent_ids': sent_ids,
                    'indices': indices,
                    'split_distribution': dict(split_distribution),
                }

        return virtual_splits

    def compute_cluster_centroids(
        self,
        cluster_ids: np.ndarray,
        embeddings: np.ndarray,
        n_clusters: int,
    ) -> Dict[int, np.ndarray]:
        """Compute centroid for each cluster.

        Args:
            cluster_ids: Array of cluster assignments for each sentence
            embeddings: Sentence embeddings
            n_clusters: Number of clusters

        Returns:
            Dict mapping cluster_id -> centroid embedding
        """
        cluster_centroids = {}
        for cluster_id in range(n_clusters):
            mask = cluster_ids == cluster_id
            if mask.sum() > 0:
                cluster_centroids[cluster_id] = embeddings[mask].mean(axis=0)
        return cluster_centroids

    def build_reference_embeddings_from_virtual_splits(
        self,
        virtual_splits_by_treebank: Dict[str, Dict[str, Dict]],
    ) -> Dict[str, np.ndarray]:
        """Build genre reference embeddings from virtual splits.

        Computes cluster centroids for each virtual split, then computes
        sentence-count weighted averages across treebanks for each genre.

        This is the core reference construction used by both production and evaluation.

        Args:
            virtual_splits_by_treebank: Dict mapping treebank_code -> genre -> virtual_split_data
                where virtual_split_data has 'embeddings' key

        Returns:
            Dict mapping genre -> reference embedding
        """
        reference_genre_embeddings = defaultdict(list)
        reference_genre_weights = defaultdict(list)

        for tb_code, genres_dict in virtual_splits_by_treebank.items():
            for genre, split_data in genres_dict.items():
                # Compute centroid for this virtual split
                centroid = np.mean(split_data['embeddings'], axis=0)
                reference_genre_embeddings[genre].append(centroid)
                split_size = len(split_data.get('sent_ids', []))
                reference_genre_weights[genre].append(float(split_size if split_size > 0 else 1))

                splits_info = split_data.get('split_distribution', {})
                if splits_info:
                    splits_list = list(splits_info.keys())
                    logger.debug(
                        f"    Virtual split {tb_code}:{genre} "
                        f"({len(split_data['sent_ids'])} sentences across {len(splits_list)} splits) "
                        f"→ cluster centroid"
                    )

        # Sentence-count weighted average centroids per genre
        known_genre_embeddings = {}
        for genre, centroids in reference_genre_embeddings.items():
            if len(centroids) > 0:
                known_genre_embeddings[genre] = np.average(
                    np.stack(centroids),
                    axis=0,
                    weights=np.array(reference_genre_weights[genre]),
                )
                logger.debug(
                    f"    Built reference for '{genre}' from {len(centroids)} cluster centroid(s), "
                    f"weighted by {int(sum(reference_genre_weights[genre]))} sentence(s)"
                )

        return known_genre_embeddings

    def label_clusters(
        self,
        cluster_centroids: Dict[int, np.ndarray],
        reference_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[Dict[int, Tuple[str, float, str]], int, int]:
        """Label clusters by comparing to reference embeddings.

        Uses cosine similarity to find best matching genre for each cluster.
        Applies confidence threshold to distinguish high vs low confidence assignments.

        Args:
            cluster_centroids: Dict mapping cluster_id -> centroid embedding
            reference_embeddings: Dict mapping genre -> reference embedding

        Returns:
            Tuple of:
                - cluster_labels: Dict mapping cluster_id -> (genre, confidence, method)
                  where method is 'bootstrap-labeled' (high conf) or 'bootstrap-inferred' (low conf)
                - high_conf_count: Number of high confidence assignments
                - low_conf_count: Number of low confidence assignments
        """
        cluster_labels = {}
        high_conf_count = 0
        low_conf_count = 0

        for cluster_id, centroid in cluster_centroids.items():
            best_genre = None
            best_similarity = -1

            # Find best matching genre
            for genre, genre_emb in reference_embeddings.items():
                similarity = 1 - distance.cosine(centroid, genre_emb)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_genre = genre

            confidence = best_similarity

            # Apply confidence threshold
            if confidence >= self.min_confidence:
                method = "bootstrap-labeled"
                high_conf_count += 1
            else:
                method = "bootstrap-inferred"
                low_conf_count += 1

            cluster_labels[cluster_id] = (best_genre, confidence, method)

        return cluster_labels, high_conf_count, low_conf_count

    def check_virtual_split_coverage(
        self,
        combined_embeddings: np.ndarray,
        all_sent_ids: List[str],
        sent_id_to_split: Dict[str, str],
        sentence_metadata: Dict[Tuple[str, str, str], str],
        tb_code: str,
        coverage_threshold: float = 0.8,
        min_genre_sentences: int = 1,
    ) -> Tuple[bool, set]:
        """Check if a treebank has sufficient sentence-level metadata for virtual splits.

        Args:
            combined_embeddings: Combined embeddings from all splits
            all_sent_ids: All sentence IDs
            sent_id_to_split: Mapping of sentence ID to split name
            sentence_metadata: Dict mapping (tb_code, split, sent_id) -> genre
            tb_code: Treebank code
            coverage_threshold: Minimum fraction of sentences with metadata (default: 0.8)
            min_genre_sentences: Minimum sentence count required per genre to keep
                a virtual split (default: 1)

        Returns:
            Tuple of:
                - can_create_virtual_splits: True if coverage >= threshold
                - genres: Set of genres meeting min_genre_sentences
        """
        genre_counts = defaultdict(int)
        sentences_with_metadata = 0

        for sent_id in all_sent_ids:
            split_name = sent_id_to_split[sent_id]
            key = (tb_code, split_name, sent_id)
            if key in sentence_metadata:
                genre = sentence_metadata[key]
                genre_counts[genre] += 1
                sentences_with_metadata += 1

        coverage = sentences_with_metadata / len(all_sent_ids) if len(all_sent_ids) > 0 else 0
        eligible_genres = {
            genre for genre, count in genre_counts.items() if count >= min_genre_sentences
        }
        can_create = len(eligible_genres) >= 2 and coverage >= coverage_threshold

        return can_create, eligible_genres
