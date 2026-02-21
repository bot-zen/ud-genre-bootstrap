"""Tests for sentence split-map utilities."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from ud_genre_bootstrap.utils.paper_split_converter import (
    build_sentence_split_map_from_index_split,
)
from ud_genre_bootstrap.utils.sentence_split_map import (
    filter_embeddings_by_sentence_split_map,
    load_sentence_split_map,
)


def test_load_sentence_split_map_with_partition_filter(tmp_path):
    split_map_path = tmp_path / "split_map.parquet"
    pd.DataFrame(
        [
            {
                "partition": "train",
                "global_index": 0,
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "n001",
            },
            {
                "partition": "test",
                "global_index": 1,
                "treebank": "xx_demo",
                "split": "train",
                "sent_id": "w001",
            },
        ]
    ).to_parquet(split_map_path, index=False)

    split_map = load_sentence_split_map(split_map_path, partitions=["train"])

    assert split_map.selected_rows == 1
    assert split_map.includes_split("xx_demo", "train")
    assert split_map.includes_sentence("xx_demo", "train", "n001")
    assert not split_map.includes_sentence("xx_demo", "train", "w001")


def test_filter_embeddings_by_sentence_split_map(tmp_path):
    split_map_path = tmp_path / "split_map.csv"
    split_map_path.write_text(
        "partition,global_index,treebank,split,sent_id\n"
        "train,0,xx_demo,train,n001\n"
        "train,1,yy_demo,train,y001\n",
        encoding="utf-8",
    )
    split_map = load_sentence_split_map(split_map_path, partitions=["train"])

    embeddings_by_tb = {
        ("xx_demo", "train"): {
            "sent_id": ["n001", "w001"],
            "embedding": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        },
        ("yy_demo", "train"): {
            "sent_id": ["y001", "y002"],
            "embedding": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        },
    }

    filtered_embeddings, stats = filter_embeddings_by_sentence_split_map(
        embeddings_by_tb, split_map
    )

    assert stats.kept_splits == 2
    assert stats.dropped_splits == 0
    assert stats.kept_sentences == 2
    assert stats.dropped_sentences == 2
    assert filtered_embeddings[("xx_demo", "train")]["sent_id"] == ["n001"]
    assert filtered_embeddings[("yy_demo", "train")]["sent_id"] == ["y001"]


def test_build_sentence_split_map_from_index_split(tmp_path):
    treebanks_root = tmp_path / "ud"
    tb1 = treebanks_root / "UD_A"
    tb2 = treebanks_root / "UD_B"
    tb1.mkdir(parents=True, exist_ok=True)
    tb2.mkdir(parents=True, exist_ok=True)

    (tb1 / "aa_demo-ud-train.conllu").write_text(
        "# sent_id = a1\n"
        "# text = First.\n"
        "1\tFirst\tfirst\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n"
        "# sent_id = a2\n"
        "# text = Second.\n"
        "1\tSecond\tsecond\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (tb2 / "bb_demo-ud-test.conllu").write_text(
        "# sent_id = b1\n"
        "# text = Third.\n"
        "1\tThird\tthird\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (treebanks_root / "metadata.json").write_text(
        '{"aa_demo":{"lcode":"aa","genre":["news"],"splits":{"train":{"files":["UD_A/aa_demo-ud-train.conllu"]}}},"bb_demo":{"lcode":"bb","genre":["wiki"],"splits":{"test":{"files":["UD_B/bb_demo-ud-test.conllu"]}}}}',
        encoding="utf-8",
    )

    split_pickle_path = tmp_path / "split.pkl"
    with split_pickle_path.open("wb") as handle:
        pickle.dump({"train": [0, 2], "dev": [1], "test": []}, handle)

    output_path = tmp_path / "split_map.csv"
    stats = build_sentence_split_map_from_index_split(
        treebanks_root=treebanks_root,
        split_pickle_path=split_pickle_path,
        output_path=output_path,
        partitions=["train", "dev"],
    )

    assert stats.target_indices == 3
    assert stats.rows_written == 3
    assert stats.unmatched_target_indices == 0

    frame = pd.read_csv(output_path)
    assert list(frame["partition"]) == ["train", "dev", "train"]
    assert list(frame["treebank"]) == ["aa_demo", "aa_demo", "bb_demo"]
    assert list(frame["split"]) == ["train", "train", "test"]
    assert list(frame["sent_id"]) == ["a1", "a2", "b1"]


def test_build_sentence_split_map_skips_treebank_load_errors(tmp_path, monkeypatch: pytest.MonkeyPatch):
    split_pickle_path = tmp_path / "split.pkl"
    with split_pickle_path.open("wb") as handle:
        pickle.dump({"train": [0], "dev": [], "test": []}, handle)

    output_path = tmp_path / "split_map.csv"

    class _StubLoader:
        def __init__(self, ud_source: str, ud_version: str = "2.17", metadata_path=None):
            self.ud_source = ud_source
            self.ud_version = ud_version
            self.metadata_path = metadata_path

        def get_treebank_codes(self):
            return ["aa_demo", "ar_nyuad"]

        def get_available_splits(self, treebank_code: str):
            return ["train"]

        def iter_treebank_sentences(self, treebank_code: str, split: str, metadata_only: bool = True):
            if treebank_code == "ar_nyuad":
                raise ValueError("BuilderConfig 'ar_nyuad' not found.")
            return iter([{"sent_id": "a1"}])

    monkeypatch.setattr(
        "ud_genre_bootstrap.utils.paper_split_converter.UDDataLoader",
        _StubLoader,
    )

    stats = build_sentence_split_map_from_index_split(
        split_pickle_path=split_pickle_path,
        output_path=output_path,
        partitions=["train"],
        ud_source="hf://commul/universal_dependencies",
        ud_version="2.8",
    )

    assert stats.rows_written == 1
    assert stats.load_errors == 1
    assert stats.load_error_treebanks == ("ar_nyuad",)
    assert stats.unmatched_target_indices == 0

    frame = pd.read_csv(output_path)
    assert list(frame["treebank"]) == ["aa_demo"]
    assert list(frame["sent_id"]) == ["a1"]


def test_build_sentence_split_map_uses_metadata_file_order_for_splits(tmp_path):
    treebanks_root = tmp_path / "ud"
    tb1 = treebanks_root / "UD_X"
    tb1.mkdir(parents=True, exist_ok=True)

    (tb1 / "xx_demo-ud-dev.conllu").write_text(
        "# sent_id = d1\n"
        "# text = Dev.\n"
        "1\tDev\tdev\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (tb1 / "xx_demo-ud-test.conllu").write_text(
        "# sent_id = t1\n"
        "# text = Test.\n"
        "1\tTest\ttest\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (tb1 / "xx_demo-ud-train.conllu").write_text(
        "# sent_id = r1\n"
        "# text = Train.\n"
        "1\tTrain\ttrain\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (treebanks_root / "metadata.json").write_text(
        '{"xx_demo":{"lcode":"xx","genre":["news"],"splits":{"train":{"files":["UD_X/xx_demo-ud-train.conllu"]},"dev":{"files":["UD_X/xx_demo-ud-dev.conllu"]},"test":{"files":["UD_X/xx_demo-ud-test.conllu"]}}}}',
        encoding="utf-8",
    )

    split_pickle_path = tmp_path / "split.pkl"
    with split_pickle_path.open("wb") as handle:
        pickle.dump({"train": [0], "dev": [1], "test": [2]}, handle)

    output_path = tmp_path / "split_map.csv"
    stats = build_sentence_split_map_from_index_split(
        treebanks_root=treebanks_root,
        split_pickle_path=split_pickle_path,
        output_path=output_path,
    )

    assert stats.rows_written == 3
    frame = pd.read_csv(output_path)
    assert list(frame["split"]) == ["dev", "test", "train"]
    assert list(frame["sent_id"]) == ["d1", "t1", "r1"]


def test_build_sentence_split_map_uses_metadata_file_order_for_treebanks(tmp_path):
    treebanks_root = tmp_path / "ud"
    tb_a = treebanks_root / "UD_A"
    tb_b = treebanks_root / "UD_B"
    tb_a.mkdir(parents=True, exist_ok=True)
    tb_b.mkdir(parents=True, exist_ok=True)

    (tb_a / "zz_demo-ud-train.conllu").write_text(
        "# sent_id = z1\n"
        "# text = Z.\n"
        "1\tZ\tz\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (tb_b / "aa_demo-ud-train.conllu").write_text(
        "# sent_id = a1\n"
        "# text = A.\n"
        "1\tA\ta\tNOUN\t_\t_\t0\troot\t_\t_\n"
        "\n",
        encoding="utf-8",
    )
    (treebanks_root / "metadata.json").write_text(
        '{"aa_demo":{"lcode":"aa","genre":["news"],"splits":{"train":{"files":["UD_B/aa_demo-ud-train.conllu"]}}},"zz_demo":{"lcode":"zz","genre":["news"],"splits":{"train":{"files":["UD_A/zz_demo-ud-train.conllu"]}}}}',
        encoding="utf-8",
    )

    split_pickle_path = tmp_path / "split.pkl"
    with split_pickle_path.open("wb") as handle:
        pickle.dump({"train": [0, 1]}, handle)

    output_path = tmp_path / "split_map.csv"
    stats = build_sentence_split_map_from_index_split(
        treebanks_root=treebanks_root,
        split_pickle_path=split_pickle_path,
        output_path=output_path,
        partitions=["train"],
    )

    assert stats.rows_written == 2
    frame = pd.read_csv(output_path)
    assert list(frame["treebank"]) == ["zz_demo", "aa_demo"]
    assert list(frame["sent_id"]) == ["z1", "a1"]
