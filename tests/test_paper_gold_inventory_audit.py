from ud_genre_bootstrap.evaluation.paper_gold_inventory_audit import (
    build_markdown,
    summarize_treebank_inventory,
)


def test_summarize_treebank_inventory_detects_missing_and_extra_genres():
    summary = summarize_treebank_inventory(
        tb_code="tb1",
        language="lang",
        paper_genres=["news", "wiki", "social"],
        metadata_genres=["news", "wiki"],
        observed_counts_by_split={
            "train": {"news": 3, "wiki": 1},
            "test": {"blog": 2},
        },
        total_sentences_by_split={"train": 6, "test": 4},
    )

    assert summary["paper_missing_from_sentence_gold"] == ["social"]
    assert summary["paper_missing_from_treebank_metadata"] == ["social"]
    assert summary["observed_extra_vs_paper"] == ["blog"]
    assert summary["total_sentences"] == 10
    assert summary["labeled_sentences"] == 6
    assert summary["unlabeled_sentences"] == 4


def test_build_markdown_mentions_summary_and_split_breakdown():
    report = {
        "paths": {
            "config": "cfg.yaml",
            "split_map": "split.parquet",
        },
        "summary": {
            "treebanks_audited": ["tb1"],
            "unsupported_paper_genres": {"social": ["tb1"]},
            "paper_missing_from_treebank_metadata": {},
            "observed_extras_vs_paper": {"blog": ["tb1"]},
        },
        "treebanks": {
            "tb1": {
                "language": "lang",
                "paper_genres": ["news", "social", "wiki"],
                "metadata_genres": ["news", "wiki"],
                "observed_sentence_genres": ["blog", "news"],
                "observed_sentence_counts": {"blog": 2, "news": 3},
                "paper_missing_from_sentence_gold": ["social", "wiki"],
                "paper_missing_from_treebank_metadata": ["social"],
                "metadata_missing_from_sentence_gold": ["wiki"],
                "observed_extra_vs_paper": ["blog"],
                "total_sentences": 7,
                "labeled_sentences": 5,
                "unlabeled_sentences": 2,
                "split_breakdown": {
                    "test": {
                        "total_sentences": 7,
                        "labeled_sentences": 5,
                        "unlabeled_sentences": 2,
                        "observed_counts": {"blog": 2, "news": 3},
                        "observed_genres": ["blog", "news"],
                    }
                },
            }
        },
    }

    markdown = build_markdown(report)

    assert "Treebanks audited: tb1" in markdown
    assert 'Paper genres unsupported by sentence-level gold: {"social": ["tb1"]}' in markdown
    assert "### tb1" in markdown
    assert "test: total=7, labeled=5, unlabeled=2, genres={'blog': 2, 'news': 3}" in markdown
