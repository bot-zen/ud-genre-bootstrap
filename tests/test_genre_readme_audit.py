"""Tests for README-based genre extraction audits."""

from pathlib import Path
from textwrap import dedent

import pytest

from ud_genre_bootstrap.utils.conllu import iter_conllu_sentence_metadata
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
from ud_genre_bootstrap.utils.genre_readme_audit import (
    _stable_sentence_sample_key,
    audit_genre_readmes,
    render_audit_markdown,
)


def _write_conllu(path: Path, blocks: list[str]) -> None:
    path.write_text(
        "\n\n".join(dedent(block).strip() for block in blocks) + "\n\n",
        encoding="utf-8",
    )


def test_audit_prioritizes_mapping_fixes_and_readme_hints(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    alias_dir = ud_root / "UD_Demo-Alias"
    alias_dir.mkdir()
    (alias_dir / "README.md").write_text("Genre: grammar-examples\n", encoding="utf-8")
    _write_conllu(
        alias_dir / "xx_alias-ud-train.conllu",
        [
            """
            # sent_id = a1
            # genre = examples
            # text = Example.
            """,
        ],
    )

    mapping_dir = ud_root / "UD_Demo-Mapping"
    mapping_dir.mkdir()
    (mapping_dir / "README.md").write_text("Genre: news social\n", encoding="utf-8")
    _write_conllu(
        mapping_dir / "xx_mapping-ud-train.conllu",
        [
            """
            # sent_id = c1
            # genre = chat
            # text = Chat message.
            """,
        ],
    )

    hint_dir = ud_root / "UD_Demo-Hint"
    hint_dir.mkdir()
    (hint_dir / "README.md").write_text(
        "Genre: news wiki\n"
        "The sent_id prefix marks the source document: n means news, w means wiki.\n",
        encoding="utf-8",
    )
    _write_conllu(
        hint_dir / "xx_hint-ud-train.conllu",
        [
            """
            # sent_id = n001
            # text = News sentence.
            """,
            """
            # sent_id = w001
            # text = Wiki sentence.
            """,
        ],
    )

    mapper = GenreMapper(genre_mapping_path=Path("configs/genre_mappings.json"))
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
        threshold=0.95,
    )
    by_treebank = {item["treebank"]: item for item in report["treebanks"]}

    assert report["summary"]["treebanks"] == 3
    assert report["readme_genre_counts"] == {
        "news": 2,
        "wiki": 1,
        "social": 1,
        "grammar-examples": 1,
    }

    assert by_treebank["xx_mapping"]["priority"] == "high"
    assert by_treebank["xx_mapping"]["action"] == "add_mapping"
    assert by_treebank["xx_mapping"]["sentence_scan"]["unmapped_raw_genres"] == {
        "chat": 1
    }

    assert by_treebank["xx_hint"]["priority"] == "medium"
    assert by_treebank["xx_hint"]["action"] == "inspect_readme_hint_for_pattern"
    assert "sent_id" in by_treebank["xx_hint"]["readme_hints"]

    assert by_treebank["xx_alias"]["priority"] == "integrated"
    assert by_treebank["xx_alias"]["action"] == "covered_by_alias_mapping"
    assert by_treebank["xx_alias"]["sentence_scan"]["alias_counts"] == {
        "examples -> grammar-examples": 1
    }

    markdown = render_audit_markdown(report, max_candidates=2)
    assert "# Genre README Audit" in markdown
    assert "`add_mapping`" in markdown
    assert "`inspect_readme_hint_for_pattern`" in markdown


def test_audit_honors_treebank_specific_patternless_mappings(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    mapped_dir = ud_root / "UD_Demo-TreebankMapping"
    mapped_dir.mkdir()
    (mapped_dir / "README.md").write_text("Genre: news social\n", encoding="utf-8")
    _write_conllu(
        mapped_dir / "xx_tbmapping-ud-train.conllu",
        [
            """
            # sent_id = c1
            # genre = chat
            # text = Chat message.
            """,
        ],
    )

    patterns_path = tmp_path / "patterns.json"
    patterns_path.write_text(
        '{"xx_tbmapping": [{"genre_mapping": {"chat": "social"}}]}',
        encoding="utf-8",
    )

    mapper = GenreMapper(metadata_patterns_path=patterns_path)
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
        threshold=0.95,
    )

    item = report["treebanks"][0]
    assert report["summary"]["configured_pattern_treebanks"] == 0
    assert report["summary"]["patternless_mapping_treebanks"] == 1
    assert item["configured_patterns"] is False
    assert item["configured_extraction"]["regex_pattern_count"] == 0
    assert item["configured_extraction"]["patternless_mapping_count"] == 1
    assert item["priority"] == "integrated"
    assert item["action"] == "covered_by_alias_mapping"
    assert item["sentence_scan"]["alias_counts"] == {"chat -> social": 1}
    assert item["sentence_scan"]["unmapped_raw_genres"] == {}


def test_audit_can_sort_candidates_by_sentence_count(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    small_dir = ud_root / "UD_Demo-Asmall"
    small_dir.mkdir()
    (small_dir / "README.md").write_text(
        "Genre: news wiki\nThe sent_id prefix identifies the source.\n",
        encoding="utf-8",
    )
    _write_conllu(
        small_dir / "xx_asmall-ud-train.conllu",
        [
            """
            # sent_id = s1
            # text = Small.
            """,
        ],
    )

    large_dir = ud_root / "UD_Demo-Zlarge"
    large_dir.mkdir()
    (large_dir / "README.md").write_text(
        "Genre: news wiki\nThe sent_id prefix identifies the source.\n",
        encoding="utf-8",
    )
    _write_conllu(
        large_dir / "xx_zlarge-ud-train.conllu",
        [
            """
            # sent_id = l1
            # text = Large 1.
            """,
            """
            # sent_id = l2
            # text = Large 2.
            """,
            """
            # sent_id = l3
            # text = Large 3.
            """,
        ],
    )

    mapper = GenreMapper(genre_mapping_path=Path("configs/genre_mappings.json"))
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
        threshold=0.95,
        sort_by="sentences",
    )

    assert report["sort_by"] == "sentences"
    assert [item["treebank"] for item in report["candidates"]] == [
        "xx_zlarge",
        "xx_asmall",
    ]

    markdown = render_audit_markdown(report, max_candidates=2)
    assert (
        "| `medium` | `inspect_readme_hint_for_pattern` | "
        "`xx_zlarge` | 3 | 3 | 3 |"
    ) in markdown


def test_audit_rejects_unknown_sort_mode(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()
    mapper = GenreMapper(genre_mapping_path=Path("configs/genre_mappings.json"))

    with pytest.raises(ValueError, match="Unknown audit sort mode"):
        audit_genre_readmes(
            ud_root=ud_root,
            genre_mapper=mapper,
            ud_version="2.18",
            sort_by="unknown",
        )


def test_audit_reports_readme_metadata_genre_disagreement(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    treebank_dir = ud_root / "UD_Demo-Disagreement"
    treebank_dir.mkdir()
    (treebank_dir / "README.md").write_text("Genre: news\n", encoding="utf-8")
    _write_conllu(
        treebank_dir / "xx_disagree-ud-train.conllu",
        [
            """
            # sent_id = d1
            # text = Disagreement.
            """,
        ],
    )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"xx_disagree": {"dirname": "UD_Demo-Disagreement", "genre": ["wiki"]}}',
        encoding="utf-8",
    )

    mapper = GenreMapper(genre_mapping_path=Path("configs/genre_mappings.json"))
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
        metadata_path=metadata_path,
    )

    item = report["treebanks"][0]
    assert item["declared_genres"] == ["news", "wiki"]
    assert item["readme_only_genres"] == ["news"]
    assert item["metadata_only_genres"] == ["wiki"]
    assert item["genre_sources_agree"] is False
    assert item["genre_source_status"] == "disagreement"
    assert report["summary"]["genre_source_disagreements"] == 1


def test_audit_flags_multiple_extracted_genres_per_sentence(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    treebank_dir = ud_root / "UD_Demo-Collision"
    treebank_dir.mkdir()
    (treebank_dir / "README.md").write_text("Genre: news wiki\n", encoding="utf-8")
    _write_conllu(
        treebank_dir / "xx_collision-ud-train.conllu",
        [
            """
            # sent_id = both-1
            # text = Ambiguous.
            """,
        ],
    )
    patterns_path = tmp_path / "patterns.json"
    patterns_path.write_text(
        (
            '{"xx_collision": ['
            '{"pattern": "# sent_id = both", "genre": "news"},'
            '{"pattern": "# sent_id = both", "genre": "wiki"}'
            ']}'
        ),
        encoding="utf-8",
    )

    mapper = GenreMapper(metadata_patterns_path=patterns_path)
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
    )

    item = report["treebanks"][0]
    assert item["priority"] == "high"
    assert item["action"] == "review_conflicting_patterns"
    assert item["sentence_scan"]["multi_genre_sentence_count"] == 1
    assert item["sentence_scan"]["extracted_genre_counts"] == {
        "news": 1,
        "wiki": 1,
    }
    assert item["sentence_scan"]["extracted_genre_set_counts"] == {
        "news + wiki": 1,
    }
    assert item["sentence_scan"]["collision_examples"][0]["genres"] == [
        "news",
        "wiki",
    ]
    assert report["summary"]["multi_genre_sentence_treebanks"] == 1


def test_audit_max_sentence_cap_uses_stable_hash_sampling(tmp_path):
    ud_root = tmp_path / "ud-treebanks-v2.18"
    ud_root.mkdir()

    treebank_dir = ud_root / "UD_Demo-Sampling"
    treebank_dir.mkdir()
    (treebank_dir / "README.md").write_text("Genre: news wiki\n", encoding="utf-8")
    conllu_path = treebank_dir / "xx_sampling-ud-train.conllu"
    _write_conllu(
        conllu_path,
        [
            f"""
            # sent_id = sample-{idx}
            # genre = raw{idx}
            # text = Sample {idx}.
            """
            for idx in range(10)
        ],
    )

    expected_sentences = sorted(
        iter_conllu_sentence_metadata(conllu_path),
        key=_stable_sentence_sample_key,
    )[:3]
    expected_raw_genres = {
        f"raw{sentence.sent_id.rsplit('-', 1)[1]}": 1
        for sentence in expected_sentences
    }
    first_three_genres = {f"raw{idx}": 1 for idx in range(3)}
    assert expected_raw_genres != first_three_genres

    mapper = GenreMapper(genre_mapping_path=Path("configs/genre_mappings.json"))
    report = audit_genre_readmes(
        ud_root=ud_root,
        genre_mapper=mapper,
        ud_version="2.18",
        max_sentences_per_treebank=3,
    )

    item = report["treebanks"][0]
    assert item["sentence_scan"]["available_sentences"] == 10
    assert item["sentence_scan"]["total_sentences"] == 3
    assert item["sentence_scan"]["raw_direct_genre_counts"] == expected_raw_genres
