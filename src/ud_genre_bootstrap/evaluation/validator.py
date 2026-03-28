"""Cross-validation for bootstrap evaluation."""

from collections import defaultdict
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from ud_genre_bootstrap.bootstrapping.scheduler import BootstrapScheduler
from ud_genre_bootstrap.clustering.clustering_utils import ClusteringOperations
from ud_genre_bootstrap.evaluation.metrics import ClusteringEvaluationMetrics

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
        class_report = classification_report(
            all_true,
            all_pred,
            output_dict=True,
            zero_division=0,
        )

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
        min_margin: float = 0.05,
        max_iterations: int = 10,
        anchor_mode: str = "strict",
        anchor_pool_policy: str = "auto",
        reference_weighting: str = "sentence_count",
        protocol: str = "generalization",
    ):
        """Initialize clustering evaluator.

        Args:
            n_folds: Number of folds for cross-validation
            group_by: Variable to group by to avoid data leakage ('language', 'treebank', or None)
            random_state: Random seed
            min_confidence: Minimum top-1 similarity threshold for high-confidence labeling
            min_margin: Minimum top1-top2 similarity gap for high-confidence labeling
            max_iterations: Maximum bootstrap schedule iterations
            anchor_mode: Anchor source mode ('strict' or 'parity')
            anchor_pool_policy: Anchor source policy ('auto', 'train_virtual',
                'single_genre', or 'combined')
            reference_weighting: Reference centroid aggregation strategy
                ('sentence_count' or 'uniform')
            protocol: Evaluation protocol ('generalization' or 'paper_parity')
        """
        self.n_folds = n_folds
        self.group_by = group_by
        self.random_state = random_state
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.scheduler = BootstrapScheduler(max_iterations=max_iterations)
        self.protocol = (protocol or "generalization").strip().lower()
        if self.protocol not in {"generalization", "paper_parity"}:
            raise ValueError(
                f"Invalid protocol '{protocol}'. Use 'generalization' or 'paper_parity'."
            )
        self.anchor_mode = (anchor_mode or "strict").strip().lower()
        if self.anchor_mode not in {"strict", "parity"}:
            raise ValueError(f"Invalid anchor_mode '{anchor_mode}'. Use 'strict' or 'parity'.")
        self.anchor_pool_policy = self._normalize_anchor_pool_policy(
            anchor_pool_policy,
            anchor_mode=self.anchor_mode,
        )
        self.reference_weighting = (reference_weighting or "sentence_count").strip().lower()
        if self.reference_weighting not in {"sentence_count", "uniform"}:
            raise ValueError(
                f"Invalid reference_weighting '{reference_weighting}'. Use 'sentence_count' or 'uniform'."
            )

        # Initialize shared clustering operations
        self.clustering_ops = ClusteringOperations(
            min_confidence=min_confidence,
            min_margin=min_margin,
            reference_weighting=self.reference_weighting,
        )

    @staticmethod
    def _normalize_anchor_pool_policy(policy: Optional[str], *, anchor_mode: str) -> str:
        """Normalize anchor-pool policy and resolve `auto` by anchor mode."""
        normalized = (policy or "auto").strip().lower()
        alias_map = {
            "train_virtual_only": "train_virtual",
            "single_genre_only": "single_genre",
            "virtual_only": "train_virtual",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized == "auto":
            return "combined" if anchor_mode == "parity" else "train_virtual"
        if normalized not in {"train_virtual", "single_genre", "combined"}:
            raise ValueError(
                f"Invalid anchor_pool_policy '{policy}'. "
                "Use 'auto', 'train_virtual', 'single_genre', or 'combined'."
            )
        return normalized

    @staticmethod
    def _qualify_sentence_ref(tb_code: str, split_name: str, sent_id) -> Tuple[str, str, str]:
        """Build an evaluation-local sentence ref that is unique across splits/treebanks."""
        if (
            isinstance(sent_id, tuple)
            and len(sent_id) == 3
            and sent_id[0] == tb_code
            and sent_id[1] == split_name
        ):
            return (str(sent_id[0]), str(sent_id[1]), str(sent_id[2]))
        return (str(tb_code), str(split_name), str(sent_id))

    @classmethod
    def _qualify_evaluation_inputs(
        cls,
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
    ) -> Tuple[Dict, Dict]:
        """Rewrite evaluation inputs to use split-qualified sentence refs internally."""
        qualified_sentence_metadata = {}
        for (tb_code, split_name, sent_id), genre in sentence_metadata.items():
            sent_ref = cls._qualify_sentence_ref(tb_code, split_name, sent_id)
            qualified_sentence_metadata[(tb_code, split_name, sent_ref)] = genre

        qualified_embeddings_by_tb = {}
        for (tb_code, split_name), emb_data in embeddings_by_tb.items():
            qualified_sent_ids = [
                cls._qualify_sentence_ref(tb_code, split_name, sent_id)
                for sent_id in emb_data.get("sent_id", [])
            ]
            qualified_emb_data = dict(emb_data)
            qualified_emb_data["sent_id"] = qualified_sent_ids
            qualified_embeddings_by_tb[(tb_code, split_name)] = qualified_emb_data

        return qualified_sentence_metadata, qualified_embeddings_by_tb

    @staticmethod
    def _extract_sentence_ref_parts(
        sent_ref,
        *,
        tb_code: Optional[str] = None,
        split_name: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """Extract ``(treebank, split, original_sent_id)`` from an internal sent ref."""
        if isinstance(sent_ref, tuple) and len(sent_ref) == 3:
            return (str(sent_ref[0]), str(sent_ref[1]), str(sent_ref[2]))
        if tb_code is None or split_name is None:
            raise ValueError("tb_code and split_name are required for bare sent_id values")
        return (str(tb_code), str(split_name), str(sent_ref))

    def k_fold_validate(
        self,
        multi_genre_treebanks: List[Dict],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
        single_genre_treebanks: Optional[List[Dict]] = None,
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
            single_genre_treebanks: Optional single-genre split metadata used as
                additional anchors in `parity` mode

        Returns:
            Dictionary with cross-validation results including sentence-level accuracy
        """
        logger.info(f"Starting {self.n_folds}-fold clustering evaluation")
        logger.info(f"  Evaluating {len(multi_genre_treebanks)} multi-genre treebanks")
        logger.info(f"  Anchor mode: {self.anchor_mode}")
        logger.info(f"  Anchor pool policy: {self.anchor_pool_policy}")
        logger.info(f"  Reference weighting: {self.reference_weighting}")

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

            uses_single_genre_anchors = self.anchor_pool_policy in {"single_genre", "combined"}
            parity_single_anchor_keys = None
            if uses_single_genre_anchors and single_genre_treebanks:
                if self.protocol == "paper_parity":
                    parity_single_anchor_keys = self._select_paper_single_anchor_treebanks(
                        single_genre_treebanks=single_genre_treebanks,
                        test_treebanks=test_treebanks,
                    )
                    logger.info(
                        "  Parity single-genre anchors: %d treebank(s) across %d split(s)",
                        len(parity_single_anchor_keys),
                        sum(
                            len(self._resolve_descriptor_split_keys(tb_info))
                            for tb_info in parity_single_anchor_keys
                        ),
                    )
                else:
                    parity_single_anchor_keys = self._select_parity_single_anchor_keys(
                        single_genre_treebanks=single_genre_treebanks,
                        test_treebanks=test_treebanks,
                    )
                    logger.info(
                        "  Parity single-genre anchors: %d split(s) from %d treebank(s)",
                        len(parity_single_anchor_keys),
                        len({tb for tb, _ in parity_single_anchor_keys}),
                    )

            # For each test treebank, cluster and evaluate
            fold_result = self._evaluate_fold(
                test_treebanks,
                train_treebanks,
                sentence_metadata,
                embeddings_by_tb,
                clusterer,
                parity_single_anchor_keys=parity_single_anchor_keys,
            )

            fold_results.append(fold_result)

        # Aggregate results across folds
        results = self._aggregate_fold_results(fold_results)
        results["evaluation_mode"] = "cross_validation"
        return results

    def fixed_partition_validate(
        self,
        test_treebanks: List[Dict],
        train_treebanks: List[Tuple[str, str]],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
        single_genre_treebanks: Optional[List[Dict]] = None,
    ) -> Dict:
        """Evaluate one predefined train/test partition without cross-validation.

        Args:
            test_treebanks: Held-out test descriptors (split-level in generalization,
                treebank-level with ``split_keys`` in paper parity).
            train_treebanks: Anchor split keys used to build reference embeddings.
            sentence_metadata: Dict mapping (treebank, split, sent_id) -> genre.
            embeddings_by_tb: Precomputed embeddings keyed by (treebank, split).
            clusterer: Clustering algorithm instance.
            single_genre_treebanks: Optional additional single-genre candidates for
                parity mode.

        Returns:
            Aggregated metrics over a single fixed holdout run.
        """
        logger.info("Starting fixed-partition clustering evaluation")
        logger.info("  Training anchor splits: %d", len(train_treebanks))
        held_out_split_count = sum(
            len(self._resolve_descriptor_split_keys(tb_info)) for tb_info in test_treebanks
        )
        if self.protocol == "paper_parity":
            logger.info(
                "  Held-out test treebanks: %d across %d split(s)",
                len(test_treebanks),
                held_out_split_count,
            )
        else:
            logger.info("  Held-out test splits: %d", len(test_treebanks))
        logger.info("  Protocol: %s", self.protocol)
        logger.info("  Anchor mode: %s", self.anchor_mode)
        logger.info("  Anchor pool policy: %s", self.anchor_pool_policy)
        logger.info("  Reference weighting: %s", self.reference_weighting)

        if (
            self.anchor_pool_policy in {"train_virtual", "combined"}
            and len(train_treebanks) == 0
        ):
            raise ValueError(
                "Fixed-partition evaluation requires at least one training anchor split"
            )
        if len(test_treebanks) == 0:
            if self.protocol == "paper_parity":
                raise ValueError(
                    "Fixed-partition evaluation requires at least one held-out test treebank"
                )
            raise ValueError(
                "Fixed-partition evaluation requires at least one held-out test split"
            )

        uses_single_genre_anchors = self.anchor_pool_policy in {"single_genre", "combined"}
        parity_single_anchor_keys = None
        if uses_single_genre_anchors and single_genre_treebanks:
            if self.protocol == "paper_parity":
                parity_single_anchor_keys = self._select_paper_single_anchor_treebanks(
                    single_genre_treebanks=single_genre_treebanks,
                    test_treebanks=test_treebanks,
                )
                parity_anchor_split_count = sum(
                    len(self._resolve_descriptor_split_keys(tb_info))
                    for tb_info in parity_single_anchor_keys
                )
                logger.info(
                    "  Parity single-genre anchors: %d treebank(s) across %d split(s)",
                    len(parity_single_anchor_keys),
                    parity_anchor_split_count,
                )
            else:
                parity_single_anchor_keys = self._select_parity_single_anchor_keys(
                    single_genre_treebanks=single_genre_treebanks,
                    test_treebanks=test_treebanks,
                )
                logger.info(
                    "  Parity single-genre anchors: %d split(s) from %d treebank(s)",
                    len(parity_single_anchor_keys),
                    len({tb for tb, _ in parity_single_anchor_keys}),
                )

        fold_result = self._evaluate_fold(
            test_treebanks=test_treebanks,
            train_treebanks=train_treebanks,
            sentence_metadata=sentence_metadata,
            embeddings_by_tb=embeddings_by_tb,
            clusterer=clusterer,
            parity_single_anchor_keys=parity_single_anchor_keys,
        )
        results = self._aggregate_fold_results([fold_result])
        results["evaluation_mode"] = "fixed_partition"
        return results

    @staticmethod
    def _resolve_descriptor_split_keys(treebank_info: Dict) -> List[Tuple[str, str]]:
        """Resolve split keys from split- or treebank-level descriptors."""
        if "split_keys" in treebank_info:
            return [tuple(split_key) for split_key in treebank_info.get("split_keys", [])]

        split_name = treebank_info.get("split")
        if split_name is None:
            return []

        return [(treebank_info["treebank"], split_name)]

    def _select_parity_single_anchor_keys(
        self,
        single_genre_treebanks: List[Dict],
        test_treebanks: List[Dict],
    ) -> List[Tuple[str, str]]:
        """Select single-genre anchor splits for the active evaluation protocol."""
        selected: List[Tuple[str, str]] = []
        seen = set()

        if self.protocol == "paper_parity":
            for anchor in single_genre_treebanks:
                for anchor_key in self._resolve_descriptor_split_keys(anchor):
                    if anchor_key in seen:
                        continue
                    seen.add(anchor_key)
                    selected.append(anchor_key)
            return selected

        test_treebank_codes = {tb["treebank"] for tb in test_treebanks}
        test_split_keys = {
            split_key
            for tb in test_treebanks
            for split_key in self._resolve_descriptor_split_keys(tb)
        }
        test_languages = {tb["language"] for tb in test_treebanks if tb.get("language")}

        for anchor in single_genre_treebanks:
            anchor_split_keys = self._resolve_descriptor_split_keys(anchor)
            if not anchor_split_keys:
                continue

            # Never use test treebank data as anchors.
            if (
                anchor["treebank"] in test_treebank_codes
                or any(anchor_key in test_split_keys for anchor_key in anchor_split_keys)
            ):
                continue

            # Respect language holdout when language grouping is active.
            if self.group_by == "language" and anchor.get("language") in test_languages:
                continue

            for anchor_key in anchor_split_keys:
                if anchor_key in seen:
                    continue
                seen.add(anchor_key)
                selected.append(anchor_key)

        return selected

    def _select_paper_single_anchor_treebanks(
        self,
        single_genre_treebanks: List[Dict],
        test_treebanks: List[Dict],
    ) -> List[Dict]:
        """Select whole-treebank single-genre anchors for paper parity."""
        del test_treebanks  # Same-partition anchors are allowed by protocol design.

        selected: List[Dict] = []
        seen_treebanks = set()
        for anchor in single_genre_treebanks:
            tb_code = anchor.get("treebank")
            if not tb_code or tb_code in seen_treebanks:
                continue
            if not self._resolve_descriptor_split_keys(anchor):
                continue
            seen_treebanks.add(tb_code)
            selected.append(anchor)

        return selected

    def _add_virtual_split_anchors(
        self,
        treebank_keys: List[Tuple[str, str]],
        source_tag: str,
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        genre_combination_clusters: Dict,
    ) -> Dict[str, int]:
        """Create anchor centroids from virtual splits and add them to the pool."""
        if not treebank_keys:
            return {}

        treebank_groups = self.clustering_ops.group_splits_by_treebank(
            treebank_keys, embeddings_by_tb
        )

        anchor_counts_by_genre = defaultdict(int)
        for tb_code, tb_keys in treebank_groups.items():
            try:
                combined_embeddings, all_sent_ids, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )
            except ValueError:
                continue

            virtual_splits = self.clustering_ops.create_virtual_splits(
                tb_code,
                combined_embeddings,
                all_sent_ids,
                sent_id_to_split,
                sentence_metadata,
            )

            for genre, split_data in virtual_splits.items():
                if isinstance(split_data, dict):
                    embeddings = split_data.get("embeddings")
                    sent_ids = split_data.get("sent_ids", [])
                else:
                    # Backward-compatible test/mocks may return raw embedding arrays.
                    embeddings = split_data
                    sent_ids = []
                if embeddings is None or len(embeddings) == 0:
                    continue

                centroid = np.mean(embeddings, axis=0)
                anchor_key = (tb_code, source_tag, genre)
                genre_combination_clusters[(genre,)][anchor_key] = [
                    {
                        "cluster_id": 0,
                        "embedding": centroid,
                        "sent_ids": sent_ids,
                        "confidence": 1.0,
                    }
                ]
                anchor_counts_by_genre[genre] += 1

        return dict(anchor_counts_by_genre)

    def _add_treebank_single_genre_anchors(
        self,
        treebanks: List[Dict],
        source_tag: str,
        embeddings_by_tb: Dict,
        genre_combination_clusters: Dict,
    ) -> Dict[str, int]:
        """Create anchor centroids from whole-treebank single-genre descriptors."""
        if not treebanks:
            return {}

        anchor_counts_by_genre = defaultdict(int)
        for treebank_info in treebanks:
            split_keys = self._resolve_descriptor_split_keys(treebank_info)
            if not split_keys:
                continue

            genres = list(treebank_info.get("genres", []))
            if len(genres) != 1:
                continue
            genre = genres[0]
            tb_code = treebank_info["treebank"]

            try:
                combined_embeddings, all_sent_ids, _sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(split_keys, embeddings_by_tb)
                )
            except ValueError:
                continue

            if len(combined_embeddings) == 0:
                continue

            centroid = np.mean(combined_embeddings, axis=0)
            anchor_key = (tb_code, source_tag, genre)
            genre_combination_clusters[(genre,)][anchor_key] = [
                {
                    "cluster_id": 0,
                    "embedding": centroid,
                    "sent_ids": all_sent_ids,
                    "confidence": 1.0,
                }
            ]
            anchor_counts_by_genre[genre] += 1

        return dict(anchor_counts_by_genre)

    def _evaluate_fold(
        self,
        test_treebanks: List[Dict],
        train_treebanks: List[Tuple[str, str]],
        sentence_metadata: Dict,
        embeddings_by_tb: Dict,
        clusterer,
        parity_single_anchor_keys: Optional[List] = None,
    ) -> Dict:
        """Evaluate clustering on test treebanks for one fold.

        Args:
            test_treebanks: List of test treebank metadata dicts
            train_treebanks: List of (treebank, split) tuples for training
            sentence_metadata: Sentence-level genre metadata
            embeddings_by_tb: Precomputed embeddings
            clusterer: Clustering algorithm
            parity_single_anchor_keys: Optional additional single-genre anchor
                descriptors (paper parity) or split keys (generalization)

        Returns:
            Fold results with sentence-level predictions
        """
        # Evaluation can see duplicate bare sent_id values across treebanks/splits.
        # Qualify them once here so clustering, anchor construction, and final
        # label storage all operate on collision-free sentence references.
        sentence_metadata, embeddings_by_tb = self._qualify_evaluation_inputs(
            sentence_metadata,
            embeddings_by_tb,
        )

        # Build bootstrap cluster pool from configured anchor sources.
        genre_combination_clusters = defaultdict(dict)
        anchor_counts_by_genre = defaultdict(int)
        uses_train_virtual_anchors = self.anchor_pool_policy in {"train_virtual", "combined"}
        uses_single_genre_anchors = self.anchor_pool_policy in {"single_genre", "combined"}

        train_anchor_count = 0
        if uses_train_virtual_anchors:
            train_anchor_counts = self._add_virtual_split_anchors(
                treebank_keys=train_treebanks,
                source_tag="__virtual__",
                sentence_metadata=sentence_metadata,
                embeddings_by_tb=embeddings_by_tb,
                genre_combination_clusters=genre_combination_clusters,
            )
            for genre, count in train_anchor_counts.items():
                anchor_counts_by_genre[genre] += count
            train_anchor_count = sum(train_anchor_counts.values())

        single_anchor_count = 0
        if uses_single_genre_anchors and parity_single_anchor_keys:
            if self.protocol == "paper_parity":
                single_anchor_counts = self._add_treebank_single_genre_anchors(
                    treebanks=parity_single_anchor_keys,
                    source_tag="__single_anchor__",
                    embeddings_by_tb=embeddings_by_tb,
                    genre_combination_clusters=genre_combination_clusters,
                )
            else:
                single_anchor_counts = self._add_virtual_split_anchors(
                    treebank_keys=parity_single_anchor_keys,
                    source_tag="__single_anchor__",
                    sentence_metadata=sentence_metadata,
                    embeddings_by_tb=embeddings_by_tb,
                    genre_combination_clusters=genre_combination_clusters,
                )
            for genre, count in single_anchor_counts.items():
                anchor_counts_by_genre[genre] += count
            single_anchor_count = sum(single_anchor_counts.values())

        if uses_single_genre_anchors and not parity_single_anchor_keys:
            logger.info("  No leakage-safe single-genre anchor keys available for this fold")

        # Aggregate expected genres per treebank across all test splits.
        # This ensures cluster count reflects the full combined test treebank.
        treebank_genres_map = defaultdict(set)
        for test_tb in test_treebanks:
            tb_code = test_tb['treebank']
            treebank_genres_map[tb_code].update(test_tb.get('genres', []))

        expected_test_genres = sorted({
            genre
            for genres in treebank_genres_map.values()
            for genre in genres
        })
        missing_anchor_genres = sorted(
            set(expected_test_genres) - set(anchor_counts_by_genre.keys())
        )
        logger.info(
            "  Reference genres from anchors: %s (train anchors=%d, single-genre anchors=%d, policy=%s)",
            sorted(anchor_counts_by_genre.keys()),
            train_anchor_count,
            single_anchor_count,
            self.anchor_pool_policy,
        )
        logger.info("  Anchor counts by genre: %s", dict(sorted(anchor_counts_by_genre.items())))
        if missing_anchor_genres:
            logger.info("  Missing anchor genres in this fold: %s", missing_anchor_genres)

        # Use shared operation: group all referenced test split keys by treebank code.
        test_treebank_keys = []
        for test_tb in test_treebanks:
            test_treebank_keys.extend(self._resolve_descriptor_split_keys(test_tb))
        test_treebank_keys = list(dict.fromkeys(test_treebank_keys))
        test_treebank_groups = self.clustering_ops.group_splits_by_treebank(
            test_treebank_keys, embeddings_by_tb
        )

        # Cluster each held-out treebank and add descriptors to shared pool.
        test_sentence_batches = []
        all_true = []
        all_pred = []
        all_sent_refs = []
        all_treebank_keys = []
        all_treebank_splits = []

        for tb_code, tb_keys in test_treebank_groups.items():
            try:
                combined_embeddings, all_sent_ids_list, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )
            except ValueError:
                logger.warning(f"  No embeddings for any splits of {tb_code}, skipping")
                continue

            # Determine expected genres from the union across all combined splits.
            expected_genres = sorted(treebank_genres_map.get(tb_code, []))
            n_genres = len(expected_genres)
            if n_genres < 1:
                logger.warning(
                    "  No expected genres found for %s from test metadata; defaulting to 1 cluster",
                    tb_code,
                )
                n_genres = 1

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
                compute_metrics=False,
            )

            cluster_ids = cluster_result['cluster_ids']
            clusters = cluster_result['clusters']

            # Use shared operation: Compute cluster centroids
            cluster_centroids = self.clustering_ops.compute_cluster_centroids(
                cluster_ids, combined_embeddings, n_genres
            )

            cluster_descriptors = [
                {
                    "cluster_id": cluster_id,
                    "embedding": centroid,
                    "sent_ids": clusters.get(cluster_id, {}).get("sent_ids", []),
                }
                for cluster_id, centroid in cluster_centroids.items()
            ]

            genre_combination = tuple(expected_genres)
            if len(genre_combination) == 0:
                logger.warning(
                    "  Skipping %s during labeling: no expected genre combination found",
                    tb_code,
                )
                continue

            genre_combination_clusters[genre_combination][(tb_code, "combined")] = cluster_descriptors
            test_sentence_batches.append(
                {
                    "tb_code": tb_code,
                    "sent_ids": all_sent_ids_list,
                    "sent_id_to_split": sent_id_to_split,
                }
            )

        if len(test_sentence_batches) == 0:
            logger.warning("No clustered test treebanks for this fold")
            return {
                "accuracy": 0.0,
                "num_test": len(test_treebanks),
                "num_sentences": 0,
                "anchor_policy": self.anchor_pool_policy,
                "anchors_train_virtual": train_anchor_count,
                "anchors_single_genre": single_anchor_count,
                "anchors_total": train_anchor_count + single_anchor_count,
                "anchors_by_genre": dict(sorted(anchor_counts_by_genre.items())),
                "expected_test_genres": expected_test_genres,
                "missing_anchor_genres": missing_anchor_genres,
            }

        schedule = self.scheduler.create_schedule(set(genre_combination_clusters.keys()))
        final_labels, env_summaries = self.clustering_ops.run_bootstrap_schedule(
            schedule=schedule,
            genre_combination_clusters=genre_combination_clusters,
            final_labels={},
            preserve_methods=None,
        )

        for env_idx, summary in enumerate(env_summaries):
            logger.info(
                "    Bootstrap env %d/%d: labeled %d clusters (%d high conf, %d low conf)",
                env_idx + 1,
                len(env_summaries),
                summary["labels_assigned"],
                summary["labels_high_confidence"],
                summary["labels_low_confidence"],
            )

        # Get sentence-level predictions for held-out treebanks only.
        for batch in test_sentence_batches:
            tb_code = batch["tb_code"]
            sent_refs = batch["sent_ids"]
            sent_id_to_split = batch["sent_id_to_split"]

            for sent_ref in sent_refs:
                pred_label = final_labels.get(sent_ref)
                pred_genre = pred_label[0] if pred_label is not None else None

                split_name = sent_id_to_split[sent_ref]
                ref_tb_code, ref_split_name, raw_sent_id = self._extract_sentence_ref_parts(
                    sent_ref,
                    tb_code=tb_code,
                    split_name=split_name,
                )
                meta_key = (ref_tb_code, ref_split_name, sent_ref)
                true_genre = sentence_metadata.get(meta_key, None)

                if pred_genre and true_genre:
                    all_true.append(true_genre)
                    all_pred.append(pred_genre)
                    all_sent_refs.append(f"{ref_tb_code}:{ref_split_name}:{raw_sent_id}")
                    all_treebank_keys.append(ref_tb_code)
                    all_treebank_splits.append((ref_tb_code, ref_split_name))

        if len(all_pred) == 0:
            logger.warning("No predictions for this fold")
            return {
                "accuracy": 0.0,
                "num_test": len(test_treebanks),
                "num_sentences": 0,
                "anchor_policy": self.anchor_pool_policy,
                "anchors_train_virtual": train_anchor_count,
                "anchors_single_genre": single_anchor_count,
                "anchors_total": train_anchor_count + single_anchor_count,
                "anchors_by_genre": dict(sorted(anchor_counts_by_genre.items())),
                "expected_test_genres": expected_test_genres,
                "missing_anchor_genres": missing_anchor_genres,
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
            "sent_ids": all_sent_refs,
            "treebank_keys": all_treebank_keys,
            "treebank_split_keys": all_treebank_splits,
            "anchor_policy": self.anchor_pool_policy,
            "anchors_train_virtual": train_anchor_count,
            "anchors_single_genre": single_anchor_count,
            "anchors_total": train_anchor_count + single_anchor_count,
            "anchors_by_genre": dict(sorted(anchor_counts_by_genre.items())),
            "expected_test_genres": expected_test_genres,
            "missing_anchor_genres": missing_anchor_genres,
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
        all_treebank_keys = []
        all_treebank_splits = []
        accuracies = []
        total_sentences = []
        aggregate_anchor_counts = defaultdict(int)
        missing_anchor_genres = set()
        fold_anchor_diagnostics = []

        for result in fold_results:
            true_genres = result.get("true_genres", [])
            pred_genres = result.get("pred_genres", [])
            sent_ids = result.get("sent_ids", [])

            all_true.extend(true_genres)
            all_pred.extend(pred_genres)
            all_sent_ids.extend(sent_ids)

            true_count = len(true_genres)
            split_keys = self._resolve_fold_split_keys(result, true_count)
            treebank_keys = self._resolve_fold_treebank_keys(result, true_count, split_keys)

            all_treebank_splits.extend(split_keys)
            all_treebank_keys.extend(treebank_keys)
            accuracies.append(result["accuracy"])
            total_sentences.append(result["num_sentences"])
            result_anchor_counts = result.get("anchors_by_genre", {}) or {}
            for genre, count in result_anchor_counts.items():
                aggregate_anchor_counts[genre] += int(count)
            missing_anchor_genres.update(result.get("missing_anchor_genres", []))
            fold_anchor_diagnostics.append(
                {
                    "anchor_policy": result.get("anchor_policy", self.anchor_pool_policy),
                    "anchors_train_virtual": int(result.get("anchors_train_virtual", 0)),
                    "anchors_single_genre": int(result.get("anchors_single_genre", 0)),
                    "anchors_total": int(result.get("anchors_total", 0)),
                    "anchors_by_genre": dict(sorted(result_anchor_counts.items())),
                    "expected_test_genres": sorted(result.get("expected_test_genres", [])),
                    "missing_anchor_genres": sorted(result.get("missing_anchor_genres", [])),
                }
            )

        fold_metric_summary = self._compute_fold_metric_summary(fold_results)

        if len(all_true) == 0:
            logger.warning("No sentence-level predictions across folds")
            return {
                "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
                "std_accuracy": float(np.std(accuracies)) if accuracies else 0.0,
                "overall_accuracy": 0.0,
                "confusion_matrix": [],
                "genre_labels": [],
                "classification_report": {},
                "fold_accuracies": accuracies,
                "num_folds": len(fold_results),
                "total_sentences": 0,
                "num_sentences_per_fold": total_sentences,
                "purity": 0.0,
                "agreement_treebank": 0.0,
                "agreement_by_genre_treebank": {},
                "overlap_error_treebank": 0.0,
                "overlap_error_weighted_treebank": 0.0,
                "overlap_error_by_treebank_treebank": {},
                "agreement_split": 0.0,
                "agreement_by_genre_split": {},
                "overlap_error_split": 0.0,
                "overlap_error_weighted_split": 0.0,
                "overlap_error_by_treebank_split": {},
                "micro_f1_instance": 0.0,
                "macro_f1_instance": 0.0,
                "instance_labeled_treebanks_treebank": 0,
                "instance_labeled_treebanks_split": 0,
                "anchor_policy": self.anchor_pool_policy,
                "anchor_counts_by_genre": dict(sorted(aggregate_anchor_counts.items())),
                "missing_anchor_genres": sorted(missing_anchor_genres),
                "fold_anchor_diagnostics": fold_anchor_diagnostics,
                "evaluation_protocol": self.protocol,
                **fold_metric_summary,
            }

        # Compute overall metrics
        overall_accuracy = accuracy_score(all_true, all_pred)
        conf_matrix = confusion_matrix(all_true, all_pred)
        class_report = classification_report(
            all_true,
            all_pred,
            output_dict=True,
            zero_division=0,
        )

        # Get unique genre labels in sorted order for confusion matrix axes.
        genre_labels = sorted(set(all_true))
        if len(all_treebank_keys) != len(all_true):
            logger.warning(
                "Treebank-level key count mismatch (%d keys, %d labels); "
                "falling back to a single synthetic treebank key for paper metrics",
                len(all_treebank_keys),
                len(all_true),
            )
            all_treebank_keys = ["unknown"] * len(all_true)
        if len(all_treebank_splits) != len(all_true):
            logger.warning(
                "Split-level key count mismatch (%d keys, %d labels); "
                "falling back to a single synthetic split key for diagnostics",
                len(all_treebank_splits),
                len(all_true),
            )
            all_treebank_splits = [("unknown", "unknown")] * len(all_true)

        treebank_metrics = ClusteringEvaluationMetrics.compute_all(
            true_genres=all_true,
            pred_genres=all_pred,
            treebank_keys=all_treebank_keys,
        )
        split_metrics = ClusteringEvaluationMetrics.compute_all(
            true_genres=all_true,
            pred_genres=all_pred,
            treebank_keys=all_treebank_splits,
        )

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
            "micro_f1_instance": treebank_metrics["micro_f1_instance"],
            "macro_f1_instance": treebank_metrics["macro_f1_instance"],
            "purity": treebank_metrics["purity"],
            "agreement_treebank": treebank_metrics["agreement"],
            "agreement_by_genre_treebank": treebank_metrics["agreement_by_genre"],
            "overlap_error_treebank": treebank_metrics["overlap_error"],
            "overlap_error_weighted_treebank": treebank_metrics["overlap_error_weighted"],
            "overlap_error_by_treebank_treebank": treebank_metrics["overlap_error_by_treebank"],
            "instance_labeled_treebanks_treebank": treebank_metrics["instance_labeled_treebanks"],
            "agreement_split": split_metrics["agreement"],
            "agreement_by_genre_split": split_metrics["agreement_by_genre"],
            "overlap_error_split": split_metrics["overlap_error"],
            "overlap_error_weighted_split": split_metrics["overlap_error_weighted"],
            "overlap_error_by_treebank_split": split_metrics["overlap_error_by_treebank"],
            "instance_labeled_treebanks_split": split_metrics["instance_labeled_treebanks"],
            "anchor_policy": self.anchor_pool_policy,
            "anchor_counts_by_genre": dict(sorted(aggregate_anchor_counts.items())),
            "missing_anchor_genres": sorted(missing_anchor_genres),
            "fold_anchor_diagnostics": fold_anchor_diagnostics,
            "evaluation_protocol": self.protocol,
            **fold_metric_summary,
        }

    @staticmethod
    def _resolve_fold_split_keys(result: Dict, true_count: int) -> List[Tuple[str, str]]:
        """Resolve split-level grouping keys for one fold result."""
        split_keys = result.get("treebank_split_keys", [])
        if len(split_keys) == true_count:
            return split_keys

        sent_refs = result.get("sent_ids", [])
        backfilled = []
        for sent_ref in sent_refs:
            parts = sent_ref.split(":", 2)
            if len(parts) >= 2:
                backfilled.append((parts[0], parts[1]))
        if len(backfilled) == true_count:
            return backfilled

        return [("unknown", "unknown")] * true_count

    @staticmethod
    def _resolve_fold_treebank_keys(
        result: Dict,
        true_count: int,
        split_keys: List[Tuple[str, str]],
    ) -> List[str]:
        """Resolve treebank-level grouping keys for one fold result."""
        treebank_keys = result.get("treebank_keys", [])
        if len(treebank_keys) == true_count:
            return treebank_keys

        if len(split_keys) == true_count:
            return [tb_code for tb_code, _split in split_keys]

        sent_refs = result.get("sent_ids", [])
        backfilled = []
        for sent_ref in sent_refs:
            parts = sent_ref.split(":", 2)
            if len(parts) >= 1:
                backfilled.append(parts[0])
        if len(backfilled) == true_count:
            return backfilled

        return ["unknown"] * true_count

    def _compute_fold_metric_summary(self, fold_results: List[Dict]) -> Dict:
        """Compute per-fold mean/std for cross-fold comparable metrics."""
        metric_keys = (
            "macro_f1_instance",
            "purity",
            "agreement_treebank",
            "overlap_error_treebank",
            "agreement_split",
            "overlap_error_split",
        )
        per_metric_values = {key: [] for key in metric_keys}

        for result in fold_results:
            true_genres = result.get("true_genres", [])
            pred_genres = result.get("pred_genres", [])
            if len(true_genres) == 0 or len(pred_genres) == 0:
                continue

            true_count = len(true_genres)
            split_keys = self._resolve_fold_split_keys(result, true_count)
            treebank_keys = self._resolve_fold_treebank_keys(result, true_count, split_keys)

            treebank_metrics = ClusteringEvaluationMetrics.compute_all(
                true_genres=true_genres,
                pred_genres=pred_genres,
                treebank_keys=treebank_keys,
            )
            split_metrics = ClusteringEvaluationMetrics.compute_all(
                true_genres=true_genres,
                pred_genres=pred_genres,
                treebank_keys=split_keys,
            )
            per_metric_values["macro_f1_instance"].append(
                float(treebank_metrics["macro_f1_instance"])
            )
            per_metric_values["purity"].append(float(treebank_metrics["purity"]))
            per_metric_values["agreement_treebank"].append(
                float(treebank_metrics["agreement"])
            )
            per_metric_values["overlap_error_treebank"].append(
                float(treebank_metrics["overlap_error"])
            )
            per_metric_values["agreement_split"].append(float(split_metrics["agreement"]))
            per_metric_values["overlap_error_split"].append(
                float(split_metrics["overlap_error"])
            )

        summary = {}
        for key in metric_keys:
            values = per_metric_values[key]
            if values:
                summary[f"mean_{key}"] = float(np.mean(values))
                summary[f"std_{key}"] = float(np.std(values))
            else:
                summary[f"mean_{key}"] = 0.0
                summary[f"std_{key}"] = 0.0

        return summary
