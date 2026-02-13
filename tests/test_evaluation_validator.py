import warnings

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from ud_genre_bootstrap.evaluation.validator import ClusteringEvaluator, CrossValidator


class DummyClusterer:
    def __init__(self):
        self.calls = []

    def cluster_treebank(self, embeddings, sent_ids, n_genres, compute_metrics=True):
        self.calls.append(
            {
                "embeddings_shape": embeddings.shape,
                "sent_ids": list(sent_ids),
                "n_genres": n_genres,
                "compute_metrics": compute_metrics,
            }
        )
        return {
            "cluster_ids": np.array([0, 1]),
            "cluster_probs": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "clusters": {
                0: {"sent_ids": [sent_ids[0]], "size": 1, "confidence": 1.0},
                1: {"sent_ids": [sent_ids[1]], "size": 1, "confidence": 1.0},
            },
            "metrics": {},
        }


def test_clustering_evaluator_disables_expensive_cluster_metrics(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=2, group_by=None, random_state=42)
    clusterer = DummyClusterer()

    embeddings_by_tb = {
        ("train_tb", "train"): {
            "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "sent_id": ["train1", "train2"],
        },
        ("test_tb", "test"): {
            "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "sent_id": ["test1", "test2"],
        },
    }

    sentence_metadata = {
        ("test_tb", "test", "test1"): "news",
        ("test_tb", "test", "test2"): "forum",
    }

    test_treebanks = [
        {
            "treebank": "test_tb",
            "split": "test",
            "genres": ["news", "forum"],
            "language": "X",
        }
    ]
    train_treebanks = [("train_tb", "train")]

    def _create_virtual_splits(*_args, **_kwargs):
        return {
            "news": np.array([[1.0, 0.0]]),
            "forum": np.array([[0.0, 1.0]]),
        }

    def _build_reference_embeddings(*_args, **_kwargs):
        return {
            "news": np.array([1.0, 0.0]),
            "forum": np.array([0.0, 1.0]),
        }

    def _label_cluster_descriptors(cluster_descriptors, _reference_embeddings):
        sentence_labels = {}
        for cluster in cluster_descriptors:
            if cluster["cluster_id"] == 0:
                label = ("news", 0.99, "bootstrap-labeled")
            else:
                label = ("forum", 0.99, "bootstrap-labeled")
            for sent_id in cluster["sent_ids"]:
                sentence_labels[sent_id] = label
        return {
            0: ("news", 0.99, "bootstrap-labeled"),
            1: ("forum", 0.99, "bootstrap-labeled"),
        }, sentence_labels, {}, 2, 0

    monkeypatch.setattr(
        evaluator.clustering_ops,
        "create_virtual_splits",
        _create_virtual_splits,
    )
    monkeypatch.setattr(
        evaluator.clustering_ops,
        "build_reference_embeddings_from_virtual_splits",
        _build_reference_embeddings,
    )
    monkeypatch.setattr(
        evaluator.clustering_ops,
        "label_cluster_descriptors",
        _label_cluster_descriptors,
    )

    result = evaluator._evaluate_fold(
        test_treebanks=test_treebanks,
        train_treebanks=train_treebanks,
        sentence_metadata=sentence_metadata,
        embeddings_by_tb=embeddings_by_tb,
        clusterer=clusterer,
    )

    assert len(clusterer.calls) == 1
    assert clusterer.calls[0]["compute_metrics"] is False
    assert result["accuracy"] == 1.0
    assert result["num_sentences"] == 2


def test_clustering_evaluator_uses_shared_bootstrap_schedule_runner(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=2, group_by=None, random_state=42)
    clusterer = DummyClusterer()

    embeddings_by_tb = {
        ("train_tb", "train"): {
            "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "sent_id": ["train1", "train2"],
        },
        ("test_tb", "test"): {
            "embedding": np.array([[1.0, 0.0], [0.0, 1.0]]),
            "sent_id": ["test1", "test2"],
        },
    }

    sentence_metadata = {
        ("test_tb", "test", "test1"): "news",
        ("test_tb", "test", "test2"): "forum",
    }

    test_treebanks = [
        {
            "treebank": "test_tb",
            "split": "test",
            "genres": ["news", "forum"],
            "language": "X",
        }
    ]
    train_treebanks = [("train_tb", "train")]

    def _create_virtual_splits(*_args, **_kwargs):
        return {
            "news": {
                "embeddings": np.array([[1.0, 0.0]]),
                "sent_ids": ["train1"],
            },
            "forum": {
                "embeddings": np.array([[0.0, 1.0]]),
                "sent_ids": ["train2"],
            },
        }

    captured = {}

    def _run_bootstrap_schedule(
        schedule,
        genre_combination_clusters,
        final_labels,
        preserve_methods,
    ):
        captured["schedule"] = schedule
        captured["genre_combinations"] = set(genre_combination_clusters.keys())
        captured["preserve_methods"] = preserve_methods
        return {
            "test1": ("news", 0.99, "bootstrap-labeled"),
            "test2": ("forum", 0.99, "bootstrap-labeled"),
        }, [
            {
                "labels_assigned": 2,
                "labels_high_confidence": 2,
                "labels_low_confidence": 0,
            }
        ]

    monkeypatch.setattr(
        evaluator.clustering_ops,
        "create_virtual_splits",
        _create_virtual_splits,
    )
    monkeypatch.setattr(
        evaluator.clustering_ops,
        "run_bootstrap_schedule",
        _run_bootstrap_schedule,
    )

    result = evaluator._evaluate_fold(
        test_treebanks=test_treebanks,
        train_treebanks=train_treebanks,
        sentence_metadata=sentence_metadata,
        embeddings_by_tb=embeddings_by_tb,
        clusterer=clusterer,
    )

    assert clusterer.calls[0]["compute_metrics"] is False
    assert ("news",) in captured["genre_combinations"]
    assert ("forum", "news") in captured["genre_combinations"]
    assert captured["preserve_methods"] is None
    assert len(captured["schedule"]) >= 1
    assert result["accuracy"] == 1.0
    assert result["num_sentences"] == 2


def test_clustering_evaluator_uses_union_of_split_genres_for_cluster_count(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=2, group_by=None, random_state=42)

    class RecordingClusterer:
        def __init__(self):
            self.calls = []

        def cluster_treebank(self, embeddings, sent_ids, n_genres, compute_metrics=True):
            self.calls.append(
                {
                    "embeddings_shape": embeddings.shape,
                    "sent_ids": list(sent_ids),
                    "n_genres": n_genres,
                    "compute_metrics": compute_metrics,
                }
            )
            cluster_ids = np.arange(len(sent_ids)) % n_genres
            cluster_probs = np.zeros((len(sent_ids), n_genres), dtype=float)
            cluster_probs[np.arange(len(sent_ids)), cluster_ids] = 1.0
            clusters = {}
            for cluster_id in range(n_genres):
                cluster_sent_ids = [
                    sid for sid, cid in zip(sent_ids, cluster_ids) if cid == cluster_id
                ]
                clusters[cluster_id] = {
                    "sent_ids": cluster_sent_ids,
                    "size": len(cluster_sent_ids),
                    "confidence": 1.0 if cluster_sent_ids else 0.0,
                }

            return {
                "cluster_ids": cluster_ids,
                "cluster_probs": cluster_probs,
                "clusters": clusters,
                "metrics": {},
            }

    clusterer = RecordingClusterer()

    embeddings_by_tb = {
        ("train_tb", "train"): {
            "embedding": np.array([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]]),
            "sent_id": ["train1", "train2", "train3"],
        },
        ("test_tb", "dev"): {
            "embedding": np.array([[0.9, 0.1], [0.2, 0.8]]),
            "sent_id": ["dev1", "dev2"],
        },
        ("test_tb", "test"): {
            "embedding": np.array([[0.1, 0.9], [0.7, 0.3]]),
            "sent_id": ["test1", "test2"],
        },
    }

    sentence_metadata = {
        ("test_tb", "dev", "dev1"): "news",
        ("test_tb", "dev", "dev2"): "forum",
        ("test_tb", "test", "test1"): "wiki",
        ("test_tb", "test", "test2"): "news",
    }

    test_treebanks = [
        {
            "treebank": "test_tb",
            "split": "dev",
            "genres": ["news", "forum"],
            "language": "X",
        },
        {
            "treebank": "test_tb",
            "split": "test",
            "genres": ["wiki", "news"],
            "language": "X",
        },
    ]
    train_treebanks = [("train_tb", "train")]

    def _create_virtual_splits(*_args, **_kwargs):
        return {
            "news": {"embeddings": np.array([[1.0, 0.0]]), "sent_ids": ["train1"]},
            "forum": {"embeddings": np.array([[0.0, 1.0]]), "sent_ids": ["train2"]},
            "wiki": {"embeddings": np.array([[0.7, 0.3]]), "sent_ids": ["train3"]},
        }

    def _build_reference_embeddings(*_args, **_kwargs):
        return {
            "news": np.array([1.0, 0.0]),
            "forum": np.array([0.0, 1.0]),
            "wiki": np.array([0.7, 0.3]),
        }

    def _label_cluster_descriptors(cluster_descriptors, reference_embeddings):
        labels = sorted(reference_embeddings.keys())
        cluster_labels = {}
        sentence_labels = {}
        for cluster in cluster_descriptors:
            genre = labels[cluster["cluster_id"] % len(labels)]
            label = (genre, 0.99, "bootstrap-labeled")
            cluster_labels[cluster["cluster_id"]] = label
            for sent_id in cluster["sent_ids"]:
                sentence_labels[sent_id] = label
        return cluster_labels, sentence_labels, {}, len(cluster_labels), 0

    monkeypatch.setattr(
        evaluator.clustering_ops,
        "create_virtual_splits",
        _create_virtual_splits,
    )
    monkeypatch.setattr(
        evaluator.clustering_ops,
        "build_reference_embeddings_from_virtual_splits",
        _build_reference_embeddings,
    )
    monkeypatch.setattr(
        evaluator.clustering_ops,
        "label_cluster_descriptors",
        _label_cluster_descriptors,
    )

    result = evaluator._evaluate_fold(
        test_treebanks=test_treebanks,
        train_treebanks=train_treebanks,
        sentence_metadata=sentence_metadata,
        embeddings_by_tb=embeddings_by_tb,
        clusterer=clusterer,
    )

    assert len(clusterer.calls) == 1
    assert clusterer.calls[0]["n_genres"] == 3
    assert result["num_sentences"] == 4


def test_cross_validator_aggregate_fold_results_suppresses_undefined_metric_warnings():
    validator = CrossValidator(n_folds=2)

    fold_results = [
        {
            "accuracy": 0.5,
            "true_genres": ["news", "wiki"],
            "pred_genres": ["news", "news"],
        }
    ]

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = validator._aggregate_fold_results(fold_results)

    assert not any(
        isinstance(w.message, UndefinedMetricWarning)
        for w in caught_warnings
    )
    assert result["classification_report"]["wiki"]["precision"] == 0.0


def test_clustering_evaluator_aggregate_fold_results_suppresses_undefined_metric_warnings():
    evaluator = ClusteringEvaluator(n_folds=2)

    fold_results = [
        {
            "accuracy": 0.5,
            "num_test": 1,
            "num_sentences": 2,
            "true_genres": ["news", "wiki"],
            "pred_genres": ["news", "news"],
            "sent_ids": ["tb:test:s1", "tb:test:s2"],
        }
    ]

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        result = evaluator._aggregate_fold_results(fold_results)

    assert not any(
        isinstance(w.message, UndefinedMetricWarning)
        for w in caught_warnings
    )
    assert result["classification_report"]["wiki"]["precision"] == 0.0


def test_clustering_evaluator_reports_paper_aligned_metrics():
    evaluator = ClusteringEvaluator(n_folds=2)

    fold_results = [
        {
            "accuracy": 0.75,
            "num_test": 2,
            "num_sentences": 8,
            "true_genres": [
                "news", "news", "wiki", "wiki",  # tb1
                "news", "news", "wiki", "wiki",  # tb2
            ],
            "pred_genres": [
                "news", "news", "news", "news",  # tb1
                "news", "news", "wiki", "wiki",  # tb2
            ],
            "sent_ids": [
                "tb1:test:s1", "tb1:test:s2", "tb1:test:s3", "tb1:test:s4",
                "tb2:test:s1", "tb2:test:s2", "tb2:test:s3", "tb2:test:s4",
            ],
            "treebank_split_keys": [
                ("tb1", "test"), ("tb1", "test"), ("tb1", "test"), ("tb1", "test"),
                ("tb2", "test"), ("tb2", "test"), ("tb2", "test"), ("tb2", "test"),
            ],
        }
    ]

    result = evaluator._aggregate_fold_results(fold_results)

    assert result["micro_f1_instance"] == pytest.approx(0.75)
    assert result["purity"] == pytest.approx(0.75)
    assert result["agreement"] == pytest.approx(0.75)
    assert result["overlap_error"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["overlap_error_weighted"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["instance_labeled_treebanks"] == 2
    assert result["agreement_by_genre"]["news"] == pytest.approx(1.0)
    assert result["agreement_by_genre"]["wiki"] == pytest.approx(0.5)
    assert result["overlap_error_by_treebank"]["tb1:test"] == pytest.approx(0.2928932188, abs=1e-8)
    assert result["overlap_error_by_treebank"]["tb2:test"] == pytest.approx(0.0)


def test_clustering_evaluator_metrics_backfill_treebank_keys_from_sent_ids():
    evaluator = ClusteringEvaluator(n_folds=2)
    fold_results = [
        {
            "accuracy": 1.0,
            "num_test": 1,
            "num_sentences": 2,
            "true_genres": ["news", "wiki"],
            "pred_genres": ["news", "wiki"],
            "sent_ids": ["tbx:test:s1", "tbx:test:s2"],
        }
    ]

    result = evaluator._aggregate_fold_results(fold_results)

    assert result["instance_labeled_treebanks"] == 1
    assert result["overlap_error_by_treebank"]["tbx:test"] == pytest.approx(0.0)
