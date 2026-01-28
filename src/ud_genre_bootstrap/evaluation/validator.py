"""Cross-validation for bootstrap evaluation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ud_genre_bootstrap.clustering.clustering_utils import ClusteringOperations

logger = logging.getLogger(__name__)


class CrossValidator:
    """Perform cross-validation to evaluate bootstrap quality."""

    def __init__(
        self,
        n_folds: int = 5,
        stratify_by: str = "genre",
        group_by: Optional[str] = "language",
        random_state: int = 42,
    ):
        """Initialize cross-validator.

        Args:
            n_folds: Number of folds for cross-validation
            stratify_by: Variable to stratify on ('genre')
            group_by: Variable to group by to avoid data leakage ('language')
            random_state: Random seed
        """
        self.n_folds = n_folds
        self.stratify_by = stratify_by
        self.group_by = group_by
        self.random_state = random_state

    def k_fold_validate(
        self,
        treebank_data: List[Dict],
        bootstrapper_fn,
    ) -> Dict:
        """Perform k-fold cross-validation.

        Args:
            treebank_data: List of treebank metadata dicts with genres
            bootstrapper_fn: Function that runs bootstrap given visible treebanks

        Returns:
            Dictionary with cross-validation results
        """
        logger.info(f"Starting {self.n_folds}-fold cross-validation")

        # Extract single-genre treebanks for CV
        single_genre_treebanks = [
            tb for tb in treebank_data if len(tb["genres"]) == 1
        ]

        if len(single_genre_treebanks) < self.n_folds:
            logger.error(
                f"Not enough single-genre treebanks ({len(single_genre_treebanks)}) "
                f"for {self.n_folds}-fold CV"
            )
            raise ValueError("Insufficient data for cross-validation")

        # Prepare data for stratified k-fold
        treebank_ids = [tb["id"] for tb in single_genre_treebanks]
        genres = [tb["genres"][0] for tb in single_genre_treebanks]  # Single genre
        languages = [tb["language"] for tb in single_genre_treebanks]

        # Group by language if specified
        if self.group_by == "language":
            fold_results = self._grouped_k_fold(
                treebank_ids,
                genres,
                languages,
                bootstrapper_fn,
            )
        else:
            fold_results = self._stratified_k_fold(
                treebank_ids,
                genres,
                bootstrapper_fn,
            )

        # Aggregate results
        return self._aggregate_fold_results(fold_results)

    def _stratified_k_fold(
        self,
        treebank_ids: List[str],
        genres: List[str],
        bootstrapper_fn,
    ) -> List[Dict]:
        """Perform stratified k-fold (may split same language across folds).

        Args:
            treebank_ids: List of treebank IDs
            genres: List of genre labels
            bootstrapper_fn: Bootstrap function

        Returns:
            List of fold results
        """
        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(treebank_ids, genres)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            # Hide metadata for test fold
            hidden_treebanks = [treebank_ids[i] for i in test_idx]
            visible_treebanks = [treebank_ids[i] for i in train_idx]

            # Run bootstrap
            predictions = bootstrapper_fn(visible_treebanks)

            # Evaluate on hidden treebanks
            fold_result = self._evaluate_fold(
                hidden_treebanks,
                [genres[i] for i in test_idx],
                predictions,
            )

            fold_results.append(fold_result)

        return fold_results

    def _grouped_k_fold(
        self,
        treebank_ids: List[str],
        genres: List[str],
        languages: List[str],
        bootstrapper_fn,
    ) -> List[Dict]:
        """Perform grouped k-fold (keeps same group together).

        Args:
            treebank_ids: List of treebank IDs
            genres: List of genre labels
            languages: List of language labels
            bootstrapper_fn: Bootstrap function

        Returns:
            List of fold results
        """
        from sklearn.model_selection import GroupKFold

        # Determine grouping variable
        if self.group_by == "treebank":
            # Each treebank is its own group (treebank-level CV)
            groups = treebank_ids
        elif self.group_by == "language":
            # Group by language (language-level CV)
            groups = languages
        else:
            # No grouping (sample-level CV) - use stratified instead
            return self._stratified_k_fold(treebank_ids, genres, bootstrapper_fn)

        # Use GroupKFold to ensure same group stays together
        gkf = GroupKFold(n_splits=self.n_folds)

        fold_results = []

        # Convert to numpy arrays for sklearn
        X = np.arange(len(treebank_ids))  # Dummy features
        y = np.array(genres)  # Not used for splitting, but needed for API

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            # Hide metadata for test fold
            hidden_treebanks = [treebank_ids[i] for i in test_idx]
            visible_treebanks = [treebank_ids[i] for i in train_idx]

            logger.info(
                f"  Training on {len(visible_treebanks)} treebanks, "
                f"testing on {len(hidden_treebanks)} treebanks"
            )

            # Run bootstrap
            predictions = bootstrapper_fn(visible_treebanks)

            # Evaluate on hidden treebanks
            fold_result = self._evaluate_fold(
                hidden_treebanks,
                [genres[i] for i in test_idx],
                predictions,
            )

            fold_results.append(fold_result)

        return fold_results

    def _evaluate_fold(
        self,
        test_treebanks: List[str],
        true_genres: List[str],
        predictions: Dict[str, str],
    ) -> Dict:
        """Evaluate predictions for a single fold.

        Args:
            test_treebanks: Treebank IDs in test set
            true_genres: Ground truth genres
            predictions: Predicted genres {treebank_id: genre}

        Returns:
            Fold evaluation results
        """
        # Extract predictions for test treebanks
        pred_genres = [predictions.get(tb, None) for tb in test_treebanks]

        # Filter out None predictions
        valid_mask = [p is not None for p in pred_genres]
        true_filtered = [g for g, v in zip(true_genres, valid_mask) if v]
        pred_filtered = [p for p, v in zip(pred_genres, valid_mask) if v]

        if len(pred_filtered) == 0:
            logger.warning("No predictions for this fold")
            return {
                "accuracy": 0.0,
                "num_test": len(test_treebanks),
                "num_predicted": 0,
            }

        # Compute metrics
        accuracy = accuracy_score(true_filtered, pred_filtered)

        return {
            "accuracy": float(accuracy),
            "num_test": len(test_treebanks),
            "num_predicted": len(pred_filtered),
            "true_genres": true_filtered,
            "pred_genres": pred_filtered,
        }

    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds.

        Args:
            fold_results: List of per-fold results

        Returns:
            Aggregated results
        """
        # Collect all predictions
        all_true = []
        all_pred = []
        accuracies = []

        for result in fold_results:
            all_true.extend(result["true_genres"])
            all_pred.extend(result["pred_genres"])
            accuracies.append(result["accuracy"])

        # Compute overall metrics
        overall_accuracy = accuracy_score(all_true, all_pred)
        conf_matrix = confusion_matrix(all_true, all_pred)
        class_report = classification_report(all_true, all_pred, output_dict=True)

        # Get unique genre labels in sorted order for confusion matrix axes
        genre_labels = sorted(set(all_true))

        return {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "overall_accuracy": float(overall_accuracy),
            "confusion_matrix": conf_matrix.tolist(),
            "genre_labels": genre_labels,
            "classification_report": class_report,
            "fold_accuracies": accuracies,
            "num_folds": len(fold_results),
        }

    def holdout_validate(
        self,
        treebank_data: List[Dict],
        test_fraction: float,
        bootstrapper_fn,
    ) -> Dict:
        """Perform holdout validation.

        Args:
            treebank_data: List of treebank metadata
            test_fraction: Fraction to hold out
            bootstrapper_fn: Bootstrap function

        Returns:
            Validation results
        """
        from sklearn.model_selection import train_test_split

        # Extract single-genre treebanks
        single_genre_treebanks = [
            tb for tb in treebank_data if len(tb["genres"]) == 1
        ]

        if len(single_genre_treebanks) < 2:
            logger.error("Need at least 2 single-genre treebanks for holdout validation")
            raise ValueError("Insufficient data for holdout validation")

        # Prepare data
        treebank_ids = [tb["id"] for tb in single_genre_treebanks]
        genres = [tb["genres"][0] for tb in single_genre_treebanks]
        languages = [tb["language"] for tb in single_genre_treebanks]

        # Stratified train/test split
        if self.group_by == "language":
            # Group by language - use first occurrence of each language
            unique_langs = {}
            for i, lang in enumerate(languages):
                if lang not in unique_langs:
                    unique_langs[lang] = []
                unique_langs[lang].append(i)

            # Split languages into train/test
            lang_list = list(unique_langs.keys())
            train_langs, test_langs = train_test_split(
                lang_list,
                test_size=test_fraction,
                random_state=self.random_state,
            )

            train_idx = [i for lang in train_langs for i in unique_langs[lang]]
            test_idx = [i for lang in test_langs for i in unique_langs[lang]]
        else:
            # Simple stratified split
            train_idx, test_idx = train_test_split(
                range(len(treebank_ids)),
                test_size=test_fraction,
                stratify=genres,
                random_state=self.random_state,
            )

        # Get train/test sets
        train_treebanks = [treebank_ids[i] for i in train_idx]
        test_treebanks = [treebank_ids[i] for i in test_idx]
        test_genres = [genres[i] for i in test_idx]

        logger.info(
            f"Holdout split: {len(train_treebanks)} train, {len(test_treebanks)} test"
        )

        # Run bootstrap on training set
        predictions = bootstrapper_fn(train_treebanks)

        # Evaluate on test set
        result = self._evaluate_fold(test_treebanks, test_genres, predictions)

        # Add overall accuracy
        result["test_fraction"] = test_fraction
        result["train_size"] = len(train_treebanks)
        result["test_size"] = len(test_treebanks)

        return result


class ClusteringEvaluator:
    """Evaluate clustering + labeling on multi-genre treebanks.

    Tests the actual problem the framework solves: clustering mixed sentences
    and assigning genres, rather than classifying pre-separated virtual splits.
    """

    def __init__(
        self,
        n_folds: int = 5,
        group_by: Optional[str] = "language",
        random_state: int = 42,
        min_confidence: float = 0.8,
    ):
        """Initialize clustering evaluator.

        Args:
            n_folds: Number of folds for cross-validation
            group_by: Variable to group by to avoid data leakage ('language', 'treebank', or None)
            random_state: Random seed
            min_confidence: Minimum confidence threshold for cluster labeling (matches production)
        """
        self.n_folds = n_folds
        self.group_by = group_by
        self.random_state = random_state
        self.min_confidence = min_confidence

        # Initialize shared clustering operations
        self.clustering_ops = ClusteringOperations(min_confidence=min_confidence)

    def k_fold_validate(
        self,
        multi_genre_treebanks: List[Dict],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
    ) -> Dict:
        """Perform k-fold cross-validation on clustering + labeling.

        Args:
            multi_genre_treebanks: List of multi-genre treebank metadata
                Each dict has: {
                    'treebank': str,  # e.g., 'de_pud'
                    'split': str,     # e.g., 'test'
                    'genres': List[str],  # e.g., ['news', 'wiki']
                    'language': str,  # e.g., 'German'
                }
            sentence_metadata: Dict mapping (treebank, split, sent_id) -> genre
            embeddings_by_tb: Dict mapping (treebank, split) -> {'embedding': ndarray, 'sent_id': list}
            clusterer: Clustering algorithm instance (GMM or K-Means)

        Returns:
            Dictionary with cross-validation results including sentence-level accuracy
        """
        logger.info(f"Starting {self.n_folds}-fold clustering evaluation")
        logger.info(f"  Evaluating {len(multi_genre_treebanks)} multi-genre treebanks")

        if len(multi_genre_treebanks) < self.n_folds:
            logger.error(
                f"Not enough multi-genre treebanks ({len(multi_genre_treebanks)}) "
                f"for {self.n_folds}-fold CV"
            )
            raise ValueError("Insufficient data for cross-validation")

        # Extract data for grouping
        treebank_keys = [(tb['treebank'], tb['split']) for tb in multi_genre_treebanks]
        languages = [tb['language'] for tb in multi_genre_treebanks]

        # Determine grouping variable
        if self.group_by == "treebank":
            groups = [f"{tb['treebank']}" for tb in multi_genre_treebanks]
        elif self.group_by == "language":
            groups = languages
        else:
            groups = None

        # Use GroupKFold if grouping specified, otherwise StratifiedKFold
        if groups:
            from sklearn.model_selection import GroupKFold
            kf = GroupKFold(n_splits=self.n_folds)
            X = np.arange(len(treebank_keys))
            y = np.zeros(len(treebank_keys))  # Dummy target
            splits = list(kf.split(X, y, groups=groups))
        else:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(range(len(treebank_keys))))

        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")

            # Get train/test treebanks
            train_treebanks = [treebank_keys[i] for i in train_idx]
            test_treebanks = [multi_genre_treebanks[i] for i in test_idx]

            logger.info(
                f"  Training on {len(train_treebanks)} treebanks, "
                f"testing on {len(test_treebanks)} treebanks"
            )

            # For each test treebank, cluster and evaluate
            fold_result = self._evaluate_fold(
                test_treebanks,
                train_treebanks,
                sentence_metadata,
                embeddings_by_tb,
                clusterer,
            )

            fold_results.append(fold_result)

        # Aggregate results across folds
        return self._aggregate_fold_results(fold_results)

    def _evaluate_fold(
        self,
        test_treebanks: List[Dict],
        train_treebanks: List[Tuple[str, str]],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
    ) -> Dict:
        """Evaluate clustering on test treebanks for one fold.

        Args:
            test_treebanks: List of test treebank metadata dicts
            train_treebanks: List of (treebank, split) tuples for training
            sentence_metadata: Sentence-level genre metadata
            embeddings_by_tb: Precomputed embeddings
            clusterer: Clustering algorithm

        Returns:
            Fold results with sentence-level predictions
        """

        # Build reference genre embeddings from training treebanks
        # Use shared operations to mirror production exactly

        # Use shared operation: Group training treebanks by treebank code
        train_treebank_groups = self.clustering_ops.group_splits_by_treebank(
            train_treebanks, embeddings_by_tb
        )

        virtual_splits_by_treebank = {}

        for tb_code, tb_keys in train_treebank_groups.items():
            # Use shared operation: Combine embeddings from all splits
            try:
                combined_train_embeddings, all_train_sent_ids, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )
            except ValueError:
                # No embeddings for this treebank
                continue

            # Use shared operation: Create virtual splits from combined data
            virtual_splits = self.clustering_ops.create_virtual_splits(
                tb_code,
                combined_train_embeddings,
                all_train_sent_ids,
                sent_id_to_split,
                sentence_metadata,
            )

            if len(virtual_splits) > 0:
                virtual_splits_by_treebank[tb_code] = virtual_splits

        # Use shared operation: Build reference embeddings from virtual splits
        known_genre_embeddings = self.clustering_ops.build_reference_embeddings_from_virtual_splits(
            virtual_splits_by_treebank
        )

        logger.info(f"  Reference genres from training: {list(known_genre_embeddings.keys())}")

        # Use shared operation: Group test treebanks by treebank code
        test_treebank_keys = [(tb['treebank'], tb['split']) for tb in test_treebanks]
        test_treebank_groups = self.clustering_ops.group_splits_by_treebank(
            test_treebank_keys, embeddings_by_tb
        )

        # Map treebank code back to test metadata
        treebank_metadata_map = {}
        for test_tb in test_treebanks:
            tb_code = test_tb['treebank']
            if tb_code not in treebank_metadata_map:
                treebank_metadata_map[tb_code] = test_tb

        # Evaluate each test treebank (combining all splits)
        all_true = []
        all_pred = []
        all_sent_ids = []

        for tb_code, tb_keys in test_treebank_groups.items():
            # Use shared operation: Combine embeddings from all splits
            try:
                combined_embeddings, all_sent_ids_list, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )
            except ValueError:
                logger.warning(f"  No embeddings for any splits of {tb_code}, skipping")
                continue

            # Get expected genres (should be same across all splits)
            test_metadata = treebank_metadata_map[tb_code]
            expected_genres = test_metadata['genres']
            n_genres = len(expected_genres)

            splits_list = [tk[1] for tk in tb_keys]
            logger.info(
                f"  Clustering {tb_code} "
                f"({len(combined_embeddings)} sentences across {len(splits_list)} splits into {n_genres} clusters)"
            )

            # Cluster the combined sentences (treebank-level clustering)
            cluster_result = clusterer.cluster_treebank(
                embeddings=combined_embeddings,
                sent_ids=all_sent_ids_list,
                n_genres=n_genres,
            )

            cluster_ids = cluster_result['cluster_ids']
            clusters = cluster_result['clusters']

            # Use shared operation: Compute cluster centroids
            cluster_centroids = self.clustering_ops.compute_cluster_centroids(
                cluster_ids, combined_embeddings, n_genres
            )

            # Use shared operation: Label clusters with confidence threshold
            cluster_labels, high_conf_count, low_conf_count = self.clustering_ops.label_clusters(
                cluster_centroids, known_genre_embeddings
            )

            logger.info(
                f"    Labeled {len(cluster_labels)} clusters "
                f"({high_conf_count} high conf, {low_conf_count} low conf)"
            )

            # Get sentence-level predictions
            for i, sent_id in enumerate(all_sent_ids_list):
                cluster_id = cluster_ids[i]
                if cluster_id in cluster_labels:
                    pred_genre, confidence, method = cluster_labels[cluster_id]
                else:
                    pred_genre = None

                split_name = sent_id_to_split[sent_id]

                # Get true genre from metadata
                meta_key = (tb_code, split_name, sent_id)
                true_genre = sentence_metadata.get(meta_key, None)

                if pred_genre and true_genre:
                    all_true.append(true_genre)
                    all_pred.append(pred_genre)
                    all_sent_ids.append(f"{tb_code}:{split_name}:{sent_id}")

        if len(all_pred) == 0:
            logger.warning("No predictions for this fold")
            return {
                "accuracy": 0.0,
                "num_test": len(test_treebanks),
                "num_sentences": 0,
            }

        # Compute accuracy
        accuracy = accuracy_score(all_true, all_pred)

        logger.info(f"  Fold accuracy: {accuracy:.3f} ({len(all_pred)} sentences)")

        return {
            "accuracy": float(accuracy),
            "num_test": len(test_treebanks),
            "num_sentences": len(all_pred),
            "true_genres": all_true,
            "pred_genres": all_pred,
            "sent_ids": all_sent_ids,
        }

    def _aggregate_fold_results(self, fold_results: List[Dict]) -> Dict:
        """Aggregate results across folds.

        Args:
            fold_results: List of per-fold results

        Returns:
            Aggregated results with sentence-level metrics
        """
        # Collect all predictions
        all_true = []
        all_pred = []
        all_sent_ids = []
        accuracies = []
        total_sentences = []

        for result in fold_results:
            all_true.extend(result["true_genres"])
            all_pred.extend(result["pred_genres"])
            all_sent_ids.extend(result["sent_ids"])
            accuracies.append(result["accuracy"])
            total_sentences.append(result["num_sentences"])

        # Compute overall metrics
        overall_accuracy = accuracy_score(all_true, all_pred)
        conf_matrix = confusion_matrix(all_true, all_pred)
        class_report = classification_report(all_true, all_pred, output_dict=True)

        # Get unique genre labels in sorted order for confusion matrix axes
        genre_labels = sorted(set(all_true))

        logger.info(f"Overall accuracy: {overall_accuracy:.3f} ({sum(total_sentences)} sentences)")

        return {
            "mean_accuracy": float(np.mean(accuracies)),
            "std_accuracy": float(np.std(accuracies)),
            "overall_accuracy": float(overall_accuracy),
            "confusion_matrix": conf_matrix.tolist(),
            "genre_labels": genre_labels,
            "classification_report": class_report,
            "fold_accuracies": accuracies,
            "num_folds": len(fold_results),
            "total_sentences": sum(total_sentences),
            "num_sentences_per_fold": total_sentences,
        }
