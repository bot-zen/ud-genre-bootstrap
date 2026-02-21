"""Tests for data loading fast-paths."""

import json
import pytest

from ud_genre_bootstrap.utils.data_loader import UDDataLoader
from ud_genre_bootstrap.utils.genre_coverage import GenreCoverageAnalyzer
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper


def test_iter_treebank_sentences_metadata_only_local(tmp_path):
    """Metadata-only iteration should avoid token parsing for local CoNLL-U."""
    ud_root = tmp_path / "ud"
    conllu_dir = ud_root / "UD_TestTB" / "r2.17"
    conllu_dir.mkdir(parents=True)
    conllu_path = conllu_dir / "xx_testtb-ud-train.conllu"
    conllu_path.write_text(
        "# sent_id = s1\n"
        "# text = Hello\n"
        "# newdoc genre = news\n"
        "1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = s2\n"
        "# text = World\n"
        "# genre = wiki\n"
        "1\tWorld\t_\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )

    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "xx_testtb": {
                    "lcode": "xx",
                    "genre": [],
                    "splits": {
                        "train": {
                            "files": ["UD_TestTB/r2.17/xx_testtb-ud-train.conllu"],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    loader = UDDataLoader(
        ud_source=f"local://{ud_root}",
        metadata_path=metadata_path,
    )

    metadata_rows = list(
        loader.iter_treebank_sentences("xx_testtb", "train", metadata_only=True)
    )
    assert len(metadata_rows) == 2
    assert metadata_rows[0]["sent_id"] == "s1"
    assert metadata_rows[1]["sent_id"] == "s2"
    assert metadata_rows[0]["text"] == "Hello"
    assert metadata_rows[1]["text"] == "World"
    assert metadata_rows[0]["comments"] == ["# sent_id = s1", "# text = Hello", "# newdoc genre = news"]
    assert all("tokens" not in row for row in metadata_rows)

    full_rows = list(loader.iter_treebank_sentences("xx_testtb", "train"))
    assert len(full_rows) == 2
    assert full_rows[0]["tokens"] == ["Hello"]
    assert full_rows[1]["tokens"] == ["World"]


def test_genre_coverage_analyzer_uses_metadata_only_iteration():
    """Coverage analysis should call the metadata-only sentence iterator."""

    class StubLoader:
        def __init__(self):
            self.calls = []

        def iter_treebank_sentences(self, treebank_code, split, metadata_only=False):
            self.calls.append((treebank_code, split, metadata_only))
            yield {"sent_id": "s1", "comments": ["# genre = news"]}

        def get_treebank_genres(self, treebank_code):
            return []

    loader = StubLoader()
    mapper = GenreMapper(data_loader=loader)
    analyzer = GenreCoverageAnalyzer(data_loader=loader, genre_mapper=mapper)

    coverage = analyzer.analyze_split("xx_testtb", "train")
    assert coverage is not None
    assert coverage.total_sentences == 1
    assert coverage.sentences_with_genre == 1
    assert coverage.genres == {"news"}
    assert loader.calls == [("xx_testtb", "train", True)]


def test_iter_treebank_sentences_metadata_only_hf_uses_materialized_comments(monkeypatch):
    """HF metadata iteration should use batch materialization when available."""
    import ud_genre_bootstrap.utils.data_loader as data_loader_module

    class StubDataset:
        def __init__(self):
            self.column_names = ["sent_id", "text", "comments", "genre", "tokens"]
            self.selected_columns = None

        def select_columns(self, columns):
            self.selected_columns = list(columns)
            return self

        def iter(self, batch_size=1000):
            yield {
                "sent_id": ["s1", "s2"],
                "text": ["Hello", "World"],
                "comments": [["__SENT_ID__", "__TEXT__", "newdoc id = d1"], ["genre = news"]],
                "genre": [None, "news"],
            }

        def __iter__(self):
            raise AssertionError("Row iteration fallback should not run in this test")

    def fake_materialize_comment_markers_batch(batch):
        assert batch["comments"][0][:2] == ["__SENT_ID__", "__TEXT__"]
        return {
            "sent_id": batch["sent_id"],
            "text": batch["text"],
            "comments": [
                ["sent_id = s1", "text = Hello", "newdoc id = d1"],
                ["genre = news"],
            ],
            "genre": batch["genre"],
        }

    monkeypatch.setattr(
        data_loader_module,
        "_materialize_comment_markers_batch",
        fake_materialize_comment_markers_batch,
    )

    class StubLoader(UDDataLoader):
        def __init__(self):
            self.ud_source = "hf://dummy/repo"
            self.ud_version = "2.17"
            self.metadata_path = None
            self.metadata = {}
            self._dataset = StubDataset()

        def load_treebank(self, treebank_code: str, split: str = "train"):
            return self._dataset

    loader = StubLoader()
    rows = list(loader.iter_treebank_sentences("xx_testtb", "train", metadata_only=True))

    assert loader._dataset.selected_columns == ["sent_id", "text", "comments", "genre"]
    assert len(rows) == 2
    assert rows[0]["comments"] == ["sent_id = s1", "text = Hello", "newdoc id = d1"]
    assert rows[1]["comments"] == ["genre = news"]
    assert rows[1]["genre"] == "news"


def test_ud_source_requires_explicit_scheme(tmp_path):
    """Only local:// and hf:// URIs are accepted for ud_source."""
    with pytest.raises(ValueError, match="Invalid ud_source"):
        UDDataLoader(ud_source=str(tmp_path))


def test_metadata_auto_loads_from_local_source_root(tmp_path):
    """Local metadata should load from <local-root>/metadata.json by default."""
    ud_root = tmp_path / "ud"
    ud_root.mkdir(parents=True)
    metadata_path = ud_root / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "xx_testtb": {
                    "lcode": "xx",
                    "genre": ["news"],
                    "splits": {},
                }
            }
        ),
        encoding="utf-8",
    )

    loader = UDDataLoader(ud_source=f"local://{ud_root}")
    assert loader.get_treebank_codes() == ["xx_testtb"]


def test_local_split_resolution_has_no_alt_path_fallback(tmp_path):
    """Local file resolution should use only metadata-declared file paths."""
    ud_root = tmp_path / "ud"
    treebank_dir = ud_root / "UD_TestTB"
    treebank_dir.mkdir(parents=True)
    # Intentionally place file at a path that does NOT match metadata.
    (treebank_dir / "xx_testtb-ud-train.conllu").write_text(
        "# sent_id = s1\n"
        "# text = Hello\n"
        "1\tHello\t_\tINTJ\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )

    (ud_root / "metadata.json").write_text(
        json.dumps(
            {
                "xx_testtb": {
                    "lcode": "xx",
                    "genre": [],
                    "splits": {
                        "train": {
                            "files": ["UD_TestTB/r2.17/xx_testtb-ud-train.conllu"],
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    loader = UDDataLoader(ud_source=f"local://{ud_root}")
    assert loader._resolve_local_split_files("xx_testtb", "train") == []
