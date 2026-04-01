"""Helpers for split-qualified sentence references."""

from typing import Any, Dict, Tuple


SentenceRef = Tuple[str, str, str]


def qualify_sentence_ref(tb_code: str, split_name: str, sent_id: Any) -> SentenceRef:
    """Build a sentence ref unique across treebanks and splits."""
    if (
        isinstance(sent_id, tuple)
        and len(sent_id) == 3
        and str(sent_id[0]) == str(tb_code)
        and str(sent_id[1]) == str(split_name)
    ):
        return (str(sent_id[0]), str(sent_id[1]), str(sent_id[2]))

    return (str(tb_code), str(split_name), str(sent_id))


def extract_sentence_ref_parts(
    sent_ref: Any,
    *,
    tb_code: str | None = None,
    split_name: str | None = None,
) -> SentenceRef:
    """Extract ``(treebank, split, original_sent_id)`` from a sentence ref."""
    if isinstance(sent_ref, tuple) and len(sent_ref) == 3:
        return (str(sent_ref[0]), str(sent_ref[1]), str(sent_ref[2]))

    if tb_code is None or split_name is None:
        raise ValueError("tb_code and split_name are required for bare sent_id values")

    return (str(tb_code), str(split_name), str(sent_ref))


def qualify_embeddings_for_split(tb_code: str, split_name: str, emb_data: Dict) -> Dict:
    """Rewrite one embedding batch to use split-qualified sentence refs."""
    qualified = dict(emb_data)
    qualified["sent_id"] = [
        qualify_sentence_ref(tb_code, split_name, sent_id)
        for sent_id in emb_data.get("sent_id", [])
    ]
    return qualified


def qualify_sentence_metadata(sentence_metadata: Dict[Tuple[str, str, Any], str]) -> Dict:
    """Rewrite sentence metadata to use split-qualified sentence refs."""
    qualified = {}
    for (tb_code, split_name, sent_id), genre in sentence_metadata.items():
        sent_ref = qualify_sentence_ref(tb_code, split_name, sent_id)
        qualified[(str(tb_code), str(split_name), sent_ref)] = genre
    return qualified
