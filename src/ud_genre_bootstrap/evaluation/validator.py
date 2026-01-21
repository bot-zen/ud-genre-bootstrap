"""Cross-validation for bootstrap evaluation."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

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
