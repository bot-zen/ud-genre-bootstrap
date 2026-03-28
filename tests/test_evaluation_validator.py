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
            ("test_tb", "test", "test1"): ("news", 0.99, "bootstrap-labeled"),
            ("test_tb", "test", "test2"): ("forum", 0.99, "bootstrap-labeled"),
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


def test_clustering_evaluator_qualifies_duplicate_sentence_ids_across_treebanks(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=2, group_by=None, random_state=42)

    class CollisionClusterer:
        def cluster_treebank(self, embeddings, sent_ids, n_genres, compute_metrics=True):
            del embeddings, n_genres, compute_metrics
            return {
                "cluster_ids": np.zeros(len(sent_ids), dtype=int),
                "cluster_probs": np.ones((len(sent_ids), 1), dtype=float),
                "clusters": {
                    0: {
                        "sent_ids": list(sent_ids),
                        "size": len(sent_ids),
                        "confidence": 1.0,
                    }
                },
                "metrics": {},
            }

    embeddings_by_tb = {
        ("tb_a", "test"): {
            "embedding": np.array([[1.0, 0.0], [1.0, 0.0]]),
            "sent_id": ["dup", "a2"],
        },
        ("tb_b", "test"): {
            "embedding": np.array([[0.0, 1.0], [0.0, 1.0]]),
            "sent_id": ["dup", "b2"],
        },
    }
    sentence_metadata = {
        ("tb_a", "test", "dup"): "news",
        ("tb_a", "test", "a2"): "news",
        ("tb_b", "test", "dup"): "wiki",
        ("tb_b", "test", "b2"): "wiki",
    }
    test_treebanks = [
        {
            "treebank": "tb_a",
            "split": "test",
            "genres": ["news", "wiki"],
            "language": "a",
        },
        {
            "treebank": "tb_b",
            "split": "test",
            "genres": ["news", "wiki"],
            "language": "b",
        },
    ]

    def _run_bootstrap_schedule(
        schedule,
        genre_combination_clusters,
        final_labels,
        preserve_methods,
    ):
        del schedule, final_labels, preserve_methods
        qualified_labels = {}
        for _genre_combination, treebank_clusters in genre_combination_clusters.items():
            for tb_key, clusters in treebank_clusters.items():
                tb_code = tb_key[0]
                if tb_code not in {"tb_a", "tb_b"}:
                    continue
                label = ("news", 0.99, "bootstrap-labeled") if tb_code == "tb_a" else ("wiki", 0.99, "bootstrap-labeled")
                for cluster in clusters:
                    for sent_ref in cluster.get("sent_ids", []):
                        qualified_labels[sent_ref] = label
        return qualified_labels, [
            {
                "labels_assigned": 2,
                "labels_high_confidence": 2,
                "labels_low_confidence": 0,
            }
        ]

    monkeypatch.setattr(evaluator.clustering_ops, "run_bootstrap_schedule", _run_bootstrap_schedule)

    result = evaluator._evaluate_fold(
        test_treebanks=test_treebanks,
        train_treebanks=[],
        sentence_metadata=sentence_metadata,
        embeddings_by_tb=embeddings_by_tb,
        clusterer=CollisionClusterer(),
    )

    assert result["accuracy"] == pytest.approx(1.0)
    assert result["num_sentences"] == 4
    assert result["pred_genres"] == ["news", "news", "wiki", "wiki"]
    assert result["sent_ids"] == [
        "tb_a:test:dup",
        "tb_a:test:a2",
        "tb_b:test:dup",
        "tb_b:test:b2",
    ]


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
    assert result["macro_f1_instance"] == pytest.approx(0.7333333333, abs=1e-8)
    assert result["purity"] == pytest.approx(0.75)
    assert result["agreement_treebank"] == pytest.approx(0.75)
    assert result["overlap_error_treebank"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["overlap_error_weighted_treebank"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["instance_labeled_treebanks_treebank"] == 2
    assert result["agreement_split"] == pytest.approx(0.75)
    assert result["overlap_error_split"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["instance_labeled_treebanks_split"] == 2
    assert result["mean_macro_f1_instance"] == pytest.approx(0.7333333333, abs=1e-8)
    assert result["std_macro_f1_instance"] == pytest.approx(0.0)
    assert result["mean_purity"] == pytest.approx(0.75)
    assert result["std_purity"] == pytest.approx(0.0)
    assert result["mean_agreement_treebank"] == pytest.approx(0.75)
    assert result["std_agreement_treebank"] == pytest.approx(0.0)
    assert result["mean_overlap_error_treebank"] == pytest.approx(0.1464466094, abs=1e-8)
    assert result["std_overlap_error_treebank"] == pytest.approx(0.0)
    assert result["agreement_by_genre_treebank"]["news"] == pytest.approx(1.0)
    assert result["agreement_by_genre_treebank"]["wiki"] == pytest.approx(0.5)
    assert result["overlap_error_by_treebank_treebank"]["tb1"] == pytest.approx(0.2928932188, abs=1e-8)
    assert result["overlap_error_by_treebank_treebank"]["tb2"] == pytest.approx(0.0)
    assert result["overlap_error_by_treebank_split"]["tb1:test"] == pytest.approx(0.2928932188, abs=1e-8)
    assert result["overlap_error_by_treebank_split"]["tb2:test"] == pytest.approx(0.0)


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

    assert result["instance_labeled_treebanks_treebank"] == 1
    assert result["instance_labeled_treebanks_split"] == 1
    assert result["overlap_error_by_treebank_treebank"]["tbx"] == pytest.approx(0.0)
    assert result["overlap_error_by_treebank_split"]["tbx:test"] == pytest.approx(0.0)


def test_select_parity_single_anchor_keys_respects_leakage_constraints():
    evaluator = ClusteringEvaluator(n_folds=2, group_by="language", anchor_mode="parity")

    single_genre_treebanks = [
        {"treebank": "mono_de", "split": "train", "language": "de"},
        {"treebank": "mono_en", "split": "train", "language": "en"},
        {"treebank": "test_tb", "split": "train", "language": "en"},
        {"treebank": "mono_fr", "split": "train", "language": "fr"},
        {"treebank": "mono_fr", "split": "train", "language": "fr"},  # duplicate
    ]
    test_treebanks = [
        {"treebank": "test_tb", "split": "test", "language": "en", "genres": ["news", "wiki"]},
    ]

    selected = evaluator._select_parity_single_anchor_keys(
        single_genre_treebanks=single_genre_treebanks,
        test_treebanks=test_treebanks,
    )

    assert ("mono_de", "train") in selected
    assert ("mono_fr", "train") in selected
    assert ("mono_en", "train") not in selected
    assert ("test_tb", "train") not in selected
    assert selected.count(("mono_fr", "train")) == 1


def test_select_parity_single_anchor_keys_paper_parity_keeps_same_partition_anchors():
    evaluator = ClusteringEvaluator(
        n_folds=1,
        group_by="language",
        anchor_mode="parity",
        protocol="paper_parity",
    )

    single_genre_treebanks = [
        {"treebank": "mono_en", "split": "test", "language": "en"},
        {"treebank": "mono_de", "split": "test", "language": "de"},
        {"treebank": "mono_en", "split": "test", "language": "en"},
    ]
    test_treebanks = [
        {"treebank": "mix_en", "split": "test", "language": "en", "genres": ["news", "wiki"]},
    ]

    selected = evaluator._select_parity_single_anchor_keys(
        single_genre_treebanks=single_genre_treebanks,
        test_treebanks=test_treebanks,
    )

    assert selected == [("mono_en", "test"), ("mono_de", "test")]


def test_k_fold_validate_filters_parity_anchors_by_test_language(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=2, group_by="language", anchor_mode="parity")

    multi_genre_treebanks = [
        {
            "treebank": "tb_en",
            "split": "train",
            "genres": ["news", "wiki"],
            "language": "en",
        },
        {
            "treebank": "tb_de",
            "split": "train",
            "genres": ["news", "wiki"],
            "language": "de",
        },
    ]
    single_genre_treebanks = [
        {"treebank": "mono_en", "split": "train", "language": "en"},
        {"treebank": "mono_de", "split": "train", "language": "de"},
        {"treebank": "mono_fr", "split": "train", "language": "fr"},
    ]
    anchor_language_by_key = {
        (item["treebank"], item["split"]): item["language"]
        for item in single_genre_treebanks
    }

    captured_fold_inputs = []

    def _fake_evaluate_fold(
        test_treebanks,
        train_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        parity_single_anchor_keys=None,
    ):
        captured_fold_inputs.append(
            {
                "test_languages": {tb["language"] for tb in test_treebanks},
                "parity_keys": list(parity_single_anchor_keys or []),
            }
        )
        return {
            "accuracy": 1.0,
            "num_test": len(test_treebanks),
            "num_sentences": 1,
            "true_genres": ["news"],
            "pred_genres": ["news"],
            "sent_ids": ["tb:test:s1"],
            "treebank_split_keys": [("tb", "test")],
        }

    monkeypatch.setattr(evaluator, "_evaluate_fold", _fake_evaluate_fold)

    result = evaluator.k_fold_validate(
        multi_genre_treebanks=multi_genre_treebanks,
        sentence_metadata={},
        embeddings_by_tb={},
        clusterer=object(),
        single_genre_treebanks=single_genre_treebanks,
    )

    assert result["num_folds"] == 2
    assert len(captured_fold_inputs) == 2
    for fold_input in captured_fold_inputs:
        test_languages = fold_input["test_languages"]
        for anchor_key in fold_input["parity_keys"]:
            assert anchor_language_by_key[anchor_key] not in test_languages


def test_fixed_partition_validate_runs_single_holdout(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=5, group_by="language", anchor_mode="strict")

    test_treebanks = [
        {
            "treebank": "tb_test",
            "split": "test",
            "genres": ["news", "wiki"],
            "language": "en",
        }
    ]
    train_treebanks = [("tb_train", "train")]

    def _fake_evaluate_fold(
        test_treebanks,
        train_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        parity_single_anchor_keys=None,
    ):
        return {
            "accuracy": 0.5,
            "num_test": len(test_treebanks),
            "num_sentences": 2,
            "true_genres": ["news", "wiki"],
            "pred_genres": ["news", "news"],
            "sent_ids": ["tb_test:test:s1", "tb_test:test:s2"],
            "treebank_keys": ["tb_test", "tb_test"],
            "treebank_split_keys": [("tb_test", "test"), ("tb_test", "test")],
        }

    monkeypatch.setattr(evaluator, "_evaluate_fold", _fake_evaluate_fold)

    result = evaluator.fixed_partition_validate(
        test_treebanks=test_treebanks,
        train_treebanks=train_treebanks,
        sentence_metadata={},
        embeddings_by_tb={},
        clusterer=object(),
    )

    assert result["num_folds"] == 1
    assert result["fold_accuracies"] == [0.5]
    assert result["overall_accuracy"] == pytest.approx(0.5)
    assert result["mean_accuracy"] == pytest.approx(0.5)
    assert result["std_accuracy"] == pytest.approx(0.0)


def test_fixed_partition_validate_parity_filters_anchors(monkeypatch):
    evaluator = ClusteringEvaluator(n_folds=5, group_by="language", anchor_mode="parity")

    test_treebanks = [
        {
            "treebank": "tb_test",
            "split": "test",
            "genres": ["news", "wiki"],
            "language": "en",
        }
    ]
    train_treebanks = [("tb_train", "train")]
    single_genre_treebanks = [
        {"treebank": "mono_en", "split": "train", "language": "en"},
        {"treebank": "mono_de", "split": "train", "language": "de"},
    ]

    captured = {}

    def _fake_evaluate_fold(
        test_treebanks,
        train_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        parity_single_anchor_keys=None,
    ):
        captured["parity_keys"] = list(parity_single_anchor_keys or [])
        return {
            "accuracy": 1.0,
            "num_test": len(test_treebanks),
            "num_sentences": 1,
            "true_genres": ["news"],
            "pred_genres": ["news"],
            "sent_ids": ["tb_test:test:s1"],
            "treebank_split_keys": [("tb_test", "test")],
        }

    monkeypatch.setattr(evaluator, "_evaluate_fold", _fake_evaluate_fold)

    result = evaluator.fixed_partition_validate(
        test_treebanks=test_treebanks,
        train_treebanks=train_treebanks,
        sentence_metadata={},
        embeddings_by_tb={},
        clusterer=object(),
        single_genre_treebanks=single_genre_treebanks,
    )

    assert result["num_folds"] == 1
    assert ("mono_de", "train") in captured["parity_keys"]
    assert ("mono_en", "train") not in captured["parity_keys"]


def test_fixed_partition_validate_paper_parity_passes_treebank_anchor_descriptors(monkeypatch):
    evaluator = ClusteringEvaluator(
        n_folds=1,
        group_by=None,
        anchor_mode="parity",
        anchor_pool_policy="single_genre",
        protocol="paper_parity",
    )

    test_treebanks = [
        {
            "treebank": "mix_tb",
            "split_keys": [("mix_tb", "test"), ("mix_tb", "dev")],
            "genres": ["news", "wiki"],
            "language": "en",
        }
    ]
    single_genre_treebanks = [
        {
            "treebank": "mono_tb",
            "split_keys": [("mono_tb", "test"), ("mono_tb", "dev")],
            "genres": ["news"],
            "language": "de",
        }
    ]

    captured = {}

    def _fake_evaluate_fold(
        test_treebanks,
        train_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        parity_single_anchor_keys=None,
    ):
        captured["anchors"] = list(parity_single_anchor_keys or [])
        return {
            "accuracy": 1.0,
            "num_test": len(test_treebanks),
            "num_sentences": 1,
            "true_genres": ["news"],
            "pred_genres": ["news"],
            "sent_ids": ["mix_tb:test:s1"],
            "treebank_split_keys": [("mix_tb", "test")],
        }

    monkeypatch.setattr(evaluator, "_evaluate_fold", _fake_evaluate_fold)

    result = evaluator.fixed_partition_validate(
        test_treebanks=test_treebanks,
        train_treebanks=[],
        sentence_metadata={},
        embeddings_by_tb={},
        clusterer=object(),
        single_genre_treebanks=single_genre_treebanks,
    )

    assert result["num_folds"] == 1
    assert captured["anchors"] == single_genre_treebanks


def test_anchor_pool_policy_auto_resolves_from_anchor_mode():
    strict_eval = ClusteringEvaluator(anchor_mode="strict")
    parity_eval = ClusteringEvaluator(anchor_mode="parity")

    assert strict_eval.anchor_pool_policy == "train_virtual"
    assert parity_eval.anchor_pool_policy == "combined"


def test_k_fold_validate_single_genre_policy_uses_single_anchors_without_parity_mode(monkeypatch):
    evaluator = ClusteringEvaluator(
        n_folds=2,
        group_by="language",
        anchor_mode="strict",
        anchor_pool_policy="single_genre",
    )

    multi_genre_treebanks = [
        {
            "treebank": "tb_en",
            "split": "train",
            "genres": ["news", "wiki"],
            "language": "en",
        },
        {
            "treebank": "tb_de",
            "split": "train",
            "genres": ["news", "wiki"],
            "language": "de",
        },
    ]
    single_genre_treebanks = [
        {"treebank": "mono_en", "split": "train", "language": "en"},
        {"treebank": "mono_de", "split": "train", "language": "de"},
        {"treebank": "mono_fr", "split": "train", "language": "fr"},
    ]
    anchor_language_by_key = {
        (item["treebank"], item["split"]): item["language"]
        for item in single_genre_treebanks
    }

    captured_fold_inputs = []

    def _fake_evaluate_fold(
        test_treebanks,
        train_treebanks,
        sentence_metadata,
        embeddings_by_tb,
        clusterer,
        parity_single_anchor_keys=None,
    ):
        captured_fold_inputs.append(
            {
                "test_languages": {tb["language"] for tb in test_treebanks},
                "parity_keys": list(parity_single_anchor_keys or []),
            }
        )
        return {
            "accuracy": 1.0,
            "num_test": len(test_treebanks),
            "num_sentences": 1,
            "true_genres": ["news"],
            "pred_genres": ["news"],
            "sent_ids": ["tb:test:s1"],
            "treebank_split_keys": [("tb", "test")],
        }

    monkeypatch.setattr(evaluator, "_evaluate_fold", _fake_evaluate_fold)

    result = evaluator.k_fold_validate(
        multi_genre_treebanks=multi_genre_treebanks,
        sentence_metadata={},
        embeddings_by_tb={},
        clusterer=object(),
        single_genre_treebanks=single_genre_treebanks,
    )

    assert result["num_folds"] == 2
    assert len(captured_fold_inputs) == 2
    for fold_input in captured_fold_inputs:
        test_languages = fold_input["test_languages"]
        assert len(fold_input["parity_keys"]) >= 1
        for anchor_key in fold_input["parity_keys"]:
            assert anchor_language_by_key[anchor_key] not in test_languages


def test_aggregate_fold_results_reports_anchor_diagnostics():
    evaluator = ClusteringEvaluator(n_folds=2)
    fold_results = [
        {
            "accuracy": 1.0,
            "num_test": 1,
            "num_sentences": 2,
            "true_genres": ["news", "wiki"],
            "pred_genres": ["news", "wiki"],
            "sent_ids": ["tb:test:s1", "tb:test:s2"],
            "anchor_policy": "combined",
            "anchors_train_virtual": 3,
            "anchors_single_genre": 2,
            "anchors_total": 5,
            "anchors_by_genre": {"news": 3, "wiki": 2},
            "expected_test_genres": ["news", "wiki", "spoken"],
            "missing_anchor_genres": ["spoken"],
        }
    ]

    result = evaluator._aggregate_fold_results(fold_results)

    assert result["anchor_policy"] == "train_virtual"
    assert result["anchor_counts_by_genre"] == {"news": 3, "wiki": 2}
    assert result["missing_anchor_genres"] == ["spoken"]
    assert len(result["fold_anchor_diagnostics"]) == 1
    assert result["fold_anchor_diagnostics"][0]["anchors_total"] == 5
