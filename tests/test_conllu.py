"""Tests for shared CoNLL-U metadata iteration."""

from textwrap import dedent

from ud_genre_bootstrap.utils.conllu import (
    iter_conllu_sentence_metadata,
    iter_sentence_dicts_with_inherited_comment_metadata,
    parse_comment_key_value,
)


def test_conllu_iterator_carries_split_comments_and_inherited_metadata(tmp_path):
    conllu_path = tmp_path / "xx_demo-ud-train.conllu"
    conllu_path.write_text(
        dedent(
            """
            # newdoc id = d1
            # newdoc genre = news
            # newpar id = p1
            # sent_id = s1
            # text = One.
            1\tOne\t_\tNUM\t_\t_\t0\troot\t_\t_

            # sent_id = s2
            # text = Two.
            1\tTwo\t_\tNUM\t_\t_\t0\troot\t_\t_
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    rows = list(iter_conllu_sentence_metadata(conllu_path, include_tokens=True))

    assert len(rows) == 2
    assert rows[0].split == "train"
    assert rows[0].comment_values["sent_id"] == ["s1"]
    assert rows[0].inherited_newdoc_metadata["newdoc genre"] == ["news"]
    assert rows[0].inherited_newpar_metadata["newpar id"] == ["p1"]
    assert rows[0].token_rows[0][1] == "One"

    assert rows[1].sent_id == "s2"
    assert rows[1].inherited_newdoc_metadata["newdoc genre"] == ["news"]
    assert rows[1].inherited_newpar_metadata["newpar id"] == ["p1"]
    assert "# newdoc genre = news" in rows[1].inherited_comments()
    assert "# newpar id = p1" in rows[1].to_sentence_dict()["comments"]


def test_parse_comment_key_value_accepts_hashless_hf_comments():
    assert parse_comment_key_value("newdoc id = d1") == ("newdoc id", "d1")
    assert parse_comment_key_value("# newdoc id = d1") == ("newdoc id", "d1")


def test_sentence_dict_iterator_carries_hashless_hf_newdoc_metadata():
    rows = list(
        iter_sentence_dicts_with_inherited_comment_metadata(
            [
                {
                    "sent_id": "s1",
                    "text": "One.",
                    "comments": ["newdoc id = d1", "__SENT_ID__", "__TEXT__"],
                },
                {
                    "sent_id": "s2",
                    "text": "Two.",
                    "comments": ["__SENT_ID__", "__TEXT__"],
                },
            ],
            split="train",
        )
    )

    assert rows[0]["inherited_newdoc_metadata"] == {"newdoc id": ["d1"]}
    assert rows[1]["inherited_newdoc_metadata"] == {"newdoc id": ["d1"]}
    assert "# newdoc id = d1" in rows[1]["comments"]
    assert rows[1]["split"] == "train"
