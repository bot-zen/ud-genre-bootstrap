"""Tests for data loading fast-paths."""

import json

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
