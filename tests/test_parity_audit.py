from ud_genre_bootstrap.evaluation.parity_audit import (
    annotate_current_cluster_map,
    build_markdown,
    compare_systems,
    summarize_per_treebank,
)


def test_summarize_per_treebank_computes_metrics():
    records = [
        {"treebank": "tb1", "true": "news", "pred": "news"},
        {"treebank": "tb1", "true": "wiki", "pred": "news"},
        {"treebank": "tb2", "true": "spoken", "pred": "spoken"},
    ]

    summary = summarize_per_treebank(records)

    assert set(summary) == {"tb1", "tb2"}
    assert summary["tb1"]["n"] == 2
    assert summary["tb1"]["accuracy"] == 0.5
    assert summary["tb2"]["accuracy"] == 1.0


def test_compare_systems_reports_agreement_and_disagreements():
    current = [
        {
            "treebank": "tb1",
            "split": "test",
            "sent_id": "s1",
            "true": "news",
            "pred": "news",
        },
        {
            "treebank": "tb1",
            "split": "test",
            "sent_id": "s2",
            "true": "wiki",
            "pred": "news",
        },
    ]
    original = [
        {
            "treebank": "tb1",
            "split": "test",
            "sent_id": "s1",
            "true": "news",
            "pred": "news",
        },
        {
            "treebank": "tb1",
            "split": "test",
            "sent_id": "s2",
            "true": "wiki",
            "pred": "wiki",
        },
    ]

    comparison = compare_systems(current, original)

    assert comparison["tb1"]["shared_sentences"] == 2
    assert comparison["tb1"]["prediction_agreement"] == 0.5
    assert comparison["tb1"]["sample_disagreements"][0]["sent_id"] == "s2"


def test_build_markdown_mentions_scope_and_treebanks():
    report = {
        "current": {
            "clustering_treebanks": ["tb1", "tb2"],
            "scoring_treebanks": ["tb1"],
            "missing_anchor_genres": ["blog"],
            "cluster_map": {"tb1": {"0": {"label": "news"}}},
            "per_treebank": {
                "tb1": {
                    "accuracy": 0.5,
                    "macro_f1": 0.4,
                    "pred_counts": {"news": 2},
                }
            },
        },
        "original": {
            "test_treebanks_seen": ["tb1", "tb2", "tb3"],
            "anchor_counts": {"news": 3},
            "schedule_new_genres": ["blog"],
            "cluster_map": {"tb1": {"0": {"label": "wiki"}}},
            "per_treebank": {
                "tb1": {
                    "accuracy": 0.7,
                    "macro_f1": 0.6,
                    "pred_counts": {"wiki": 2},
                }
            },
        },
        "comparison": {
            "tb1": {
                "prediction_agreement": 0.25,
                "shared_sentences": 4,
                "sample_disagreements": [],
            }
        },
    }

    markdown = build_markdown(report)

    assert "Current clustering treebanks: tb1, tb2" in markdown
    assert "Current scored paper treebanks: tb1" in markdown
    assert "### tb1" in markdown
    assert (
        "Current vs original prediction agreement: 0.2500 over 4 shared sentences"
        in markdown
    )


def test_annotate_current_cluster_map_adds_cluster_level_labels():
    cluster_map = {
        "tb1": {
            0: {"initial_sent_count": 2, "expected_genres": ["news", "wiki"]},
            1: {"initial_sent_count": 1, "expected_genres": ["news", "wiki"]},
        }
    }
    cluster_sent_refs = {
        "tb1": {
            0: ["tb1:test:s1", "tb1:test:s2"],
            1: ["tb1:test:s3"],
        }
    }
    final_labels = {
        "tb1:test:s1": ("news", 0.91, "bootstrap-labeled"),
        "tb1:test:s2": ("news", 0.88, "bootstrap-labeled"),
        "tb1:test:s3": ("wiki", 0.42, "bootstrap-inferred"),
    }

    annotate_current_cluster_map(cluster_map, cluster_sent_refs, final_labels)

    assert cluster_map["tb1"][0]["label"] == "news"
    assert cluster_map["tb1"][0]["confidence"] == 0.91
    assert cluster_map["tb1"][0]["method"] == "bootstrap-labeled"
    assert cluster_map["tb1"][0]["labeled_sent_count"] == 2
    assert cluster_map["tb1"][1]["label"] == "wiki"
    assert cluster_map["tb1"][1]["method"] == "bootstrap-inferred"
