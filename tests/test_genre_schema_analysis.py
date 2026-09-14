import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ud_genre_bootstrap.utils.genre_schema_analysis import (
    analyze_genre_schema,
    build_data_driven_candidates,
    classification_metrics_from_confusion,
    load_candidate_schema_config,
    project_confusion_matrix,
    validate_candidate_mapping,
)


def test_validate_candidate_mapping_requires_all_source_genres():
    with pytest.raises(ValueError, match="missing source genre"):
        validate_candidate_mapping(
            {"news": "informational"},
            ["news", "wiki"],
            target_genres=["informational"],
        )


def test_validate_candidate_mapping_rejects_unknown_source_genres():
    with pytest.raises(ValueError, match="unknown source genre"):
        validate_candidate_mapping(
            {"news": "informational", "blog": "interactional"},
            ["news"],
            target_genres=["informational", "interactional"],
        )


def test_validate_candidate_mapping_rejects_unknown_target_genres():
    with pytest.raises(ValueError, match="unknown target genre"):
        validate_candidate_mapping(
            {"news": "informational", "wiki": "encyclopedia"},
            ["news", "wiki"],
            target_genres=["informational"],
        )


def test_default_candidate_config_includes_udmultigenre_informed_seed():
    source_genres, candidates = load_candidate_schema_config(
        Path("configs/genre_schema_reduction.yaml")
    )

    assert len(source_genres) == 18
    assert "udmultigenre_informed_9" in candidates
    candidate = candidates["udmultigenre_informed_9"]
    assert len(candidate.target_genres) == 9
    assert candidate.mapping["news"] == "news"
    assert candidate.mapping["spoken"] == "spoken"
    assert candidate.mapping["blog"] == "interactional"
    assert candidate.mapping["reviews"] == "interactional"


def test_project_confusion_matrix_aggregates_reduced_labels():
    matrix = [
        [5, 2, 1],
        [1, 6, 0],
        [3, 0, 4],
    ]
    labels = ["news", "wiki", "blog"]
    mapping = {
        "news": "informational",
        "wiki": "informational",
        "blog": "interactional",
    }

    projected, projected_labels = project_confusion_matrix(
        matrix,
        labels,
        mapping,
        ["informational", "interactional"],
    )

    assert projected_labels == ["informational", "interactional"]
    assert projected == [[14, 1], [3, 4]]
    metrics = classification_metrics_from_confusion(projected)
    assert metrics["micro_f1"] == pytest.approx(18 / 22)
    assert metrics["macro_f1"] == pytest.approx((28 / 32 + 8 / 12) / 2)


def test_data_driven_candidates_use_cluster_merge_score():
    pair_rows = [
        {
            "genre_a": "news",
            "genre_b": "wiki",
            "combined_score": 0.95,
            "cluster_merge_score": 0.2,
        }
    ]

    candidates = build_data_driven_candidates(
        pair_rows=pair_rows,
        source_genres=["news", "wiki"],
        thresholds=(0.5,),
    )

    assert candidates == {}


def test_analyze_genre_schema_writes_report_artifacts(tmp_path):
    release_dir = tmp_path / "genres"
    clusters_dir = release_dir / "clusters"
    clusters_dir.mkdir(parents=True)

    rows = [
        ("tb1", "train", "s1", "news", 1.0, "virtual-split"),
        ("tb1", "train", "s2", "news", 1.0, "virtual-split"),
        ("tb1", "train", "s3", "wiki", 1.0, "virtual-split"),
        ("tb1", "train", "s4", "wiki", 1.0, "virtual-split"),
        ("tb1", "train", "s5", "blog", 0.9, "cluster-derived"),
        ("tb1", "train", "s6", "blog", 0.9, "cluster-derived"),
    ]
    pd.DataFrame(
        rows,
        columns=["treebank", "split", "sent_id", "genre", "confidence", "method"],
    ).to_parquet(release_dir / "all_genres.parquet", index=False)
    pd.DataFrame(
        [
            ("tb1", "train", "s1", 0),
            ("tb1", "train", "s2", 0),
            ("tb1", "train", "s3", 0),
            ("tb1", "train", "s4", 0),
            ("tb1", "train", "s5", 1),
            ("tb1", "train", "s6", 1),
        ],
        columns=["treebank", "split", "sent_id", "cluster_id"],
    ).to_parquet(clusters_dir / "cluster_assignments.parquet", index=False)

    candidate_config = tmp_path / "schema.yaml"
    candidate_config.write_text(
        yaml.safe_dump(
            {
                "source_genres": ["blog", "news", "wiki"],
                "candidate_schemas": {
                    "coarse": {
                        "description": "Fixture coarse schema.",
                        "target_genres": ["informational", "interactional"],
                        "mapping": {
                            "blog": "interactional",
                            "news": "informational",
                            "wiki": "informational",
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    evaluation_results = tmp_path / "eval.json"
    evaluation_results.write_text(
        json.dumps(
            {
                "results": {
                    "genre_labels": ["blog", "news", "wiki"],
                    "confusion_matrix": [[3, 1, 0], [0, 4, 2], [0, 1, 5]],
                }
            }
        ),
        encoding="utf-8",
    )
    cluster_state = tmp_path / "cluster_state.pkl"
    with cluster_state.open("wb") as handle:
        pickle.dump(
            {
                "embeddings_by_tb": {
                    ("tb1", "train"): {
                        "sent_id": [
                            ("tb1", "train", "s1"),
                            ("tb1", "train", "s2"),
                            ("tb1", "train", "s3"),
                            ("tb1", "train", "s4"),
                        ],
                        "embedding": np.array(
                            [
                                [1.0, 0.0],
                                [0.9, 0.1],
                                [0.8, 0.2],
                                [0.7, 0.3],
                            ]
                        ),
                    }
                }
            },
            handle,
        )

    output_dir = tmp_path / "analysis"
    result = analyze_genre_schema(
        release_dir=release_dir,
        candidate_config=candidate_config,
        output_dir=output_dir,
        evaluation_results=evaluation_results,
        cluster_state=cluster_state,
        ud_version="fixture",
    )

    assert result["candidate_schema_count"] == 1
    assert result["evaluation_available"] is True
    assert result["centroid_available"] is True
    assert (output_dir / "report.md").exists()
    assert (output_dir / "candidate_mappings.json").exists()
    assert (output_dir / "genre_similarity.tsv").exists()
    assert "embedding_centroid_similarity" in (
        output_dir / "genre_similarity.tsv"
    ).read_text()
    assert "cluster_merge_score" in (
        output_dir / "genre_similarity.tsv"
    ).read_text()
    scores = json.loads((output_dir / "projection_scores.json").read_text())
    assert scores["coarse"]["target_genre_count"] == 2
    assert scores["coarse"]["evaluation_micro_f1"] > 0.8
