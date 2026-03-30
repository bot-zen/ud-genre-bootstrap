from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ud_genre_bootstrap.bootstrapping.bootstrapper import GenreBootstrapper
from ud_genre_bootstrap.cli import resolve_paper_evaluation_treebank_genres
from ud_genre_bootstrap.utils.config import load_config
from ud_genre_bootstrap.utils.sentence_split_map import load_sentence_split_map


@dataclass
class PaperGoldInventoryAuditPaths:
    config_path: Path
    split_map_path: Path
    json_out: Path
    markdown_out: Path


def summarize_treebank_inventory(
    tb_code: str,
    language: str,
    paper_genres: Iterable[str],
    metadata_genres: Iterable[str],
    observed_counts_by_split: Dict[str, Dict[str, int]],
    total_sentences_by_split: Dict[str, int],
) -> Dict[str, Any]:
    """Summarize paper-vs-metadata-vs-sentence-gold inventory for one treebank."""
    paper = sorted(set(paper_genres))
    metadata = sorted(set(metadata_genres))

    observed_counts: Counter[str] = Counter()
    for split_counts in observed_counts_by_split.values():
        observed_counts.update(split_counts)
    observed = sorted(observed_counts)

    labeled_sentence_count = int(sum(observed_counts.values()))
    total_sentence_count = int(sum(total_sentences_by_split.values()))
    unlabeled_sentence_count = total_sentence_count - labeled_sentence_count

    split_breakdown = {}
    for split_name in sorted(total_sentences_by_split):
        split_counts = dict(sorted(observed_counts_by_split.get(split_name, {}).items()))
        labeled_count = sum(split_counts.values())
        total_count = int(total_sentences_by_split[split_name])
        split_breakdown[split_name] = {
            "total_sentences": total_count,
            "labeled_sentences": labeled_count,
            "unlabeled_sentences": total_count - labeled_count,
            "observed_counts": split_counts,
            "observed_genres": sorted(split_counts),
        }

    return {
        "treebank": tb_code,
        "language": language,
        "paper_genres": paper,
        "metadata_genres": metadata,
        "observed_sentence_genres": observed,
        "observed_sentence_counts": dict(sorted(observed_counts.items())),
        "paper_missing_from_sentence_gold": sorted(set(paper) - set(observed)),
        "paper_missing_from_treebank_metadata": sorted(set(paper) - set(metadata)),
        "metadata_missing_from_sentence_gold": sorted(set(metadata) - set(observed)),
        "observed_extra_vs_paper": sorted(set(observed) - set(paper)),
        "metadata_extra_vs_paper": sorted(set(metadata) - set(paper)),
        "observed_not_in_treebank_metadata": sorted(set(observed) - set(metadata)),
        "total_sentences": total_sentence_count,
        "labeled_sentences": labeled_sentence_count,
        "unlabeled_sentences": unlabeled_sentence_count,
        "split_breakdown": split_breakdown,
    }


def build_paper_gold_inventory_audit(
    paths: PaperGoldInventoryAuditPaths,
    treebanks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    cfg = load_config(paths.config_path)
    bootstrapper = GenreBootstrapper(cfg)
    paper_treebank_genre_map = resolve_paper_evaluation_treebank_genres(
        bootstrapper.data_loader
    )
    target_treebanks = set(treebanks or paper_treebank_genre_map.keys())

    all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()
    if cfg.exclude_treebanks:
        all_treebank_data = [
            tb for tb in all_treebank_data if tb["id"] not in cfg.exclude_treebanks
        ]

    test_split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])

    observed_counts_by_treebank: Dict[str, Dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    total_sentences_by_treebank: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    languages_by_treebank: Dict[str, str] = {}

    for tb in all_treebank_data:
        tb_code = tb["id"]
        if tb_code not in target_treebanks:
            continue
        language = tb.get("language", tb_code.split("_", 1)[0])
        languages_by_treebank[tb_code] = language
        for split_name in bootstrapper.data_loader.get_available_splits(tb_code):
            if not test_split_map.includes_split(tb_code, split_name):
                continue

            sentence_iter = bootstrapper.data_loader.iter_treebank_sentences(
                tb_code,
                split_name,
                metadata_only=True,
            )
            for idx, sentence in enumerate(sentence_iter):
                sent_id = sentence.get("sent_id", f"{tb_code}_{split_name}_{idx}")
                if not test_split_map.includes_sentence(tb_code, split_name, sent_id):
                    continue
                total_sentences_by_treebank[tb_code][split_name] += 1
                genres = bootstrapper.genre_mapper.extract_genres_from_metadata(
                    sentence,
                    tb_code,
                )
                if not genres:
                    continue
                observed_counts_by_treebank[tb_code][split_name][genres[0]] += 1

    treebank_reports = {}
    for tb_code in sorted(target_treebanks):
        paper_genres = paper_treebank_genre_map.get(tb_code)
        if not paper_genres:
            continue
        treebank_reports[tb_code] = summarize_treebank_inventory(
            tb_code=tb_code,
            language=languages_by_treebank.get(tb_code, tb_code.split("_", 1)[0]),
            paper_genres=paper_genres,
            metadata_genres=bootstrapper.data_loader.get_treebank_genres(tb_code) or [],
            observed_counts_by_split={
                split: dict(sorted(counts.items()))
                for split, counts in observed_counts_by_treebank.get(tb_code, {}).items()
            },
            total_sentences_by_split=dict(
                sorted(total_sentences_by_treebank.get(tb_code, {}).items())
            ),
        )

    unsupported_paper_genres: Dict[str, List[str]] = defaultdict(list)
    paper_missing_from_metadata: Dict[str, List[str]] = defaultdict(list)
    observed_extras_vs_paper: Dict[str, List[str]] = defaultdict(list)
    for tb_code, tb_report in treebank_reports.items():
        for genre in tb_report["paper_missing_from_sentence_gold"]:
            unsupported_paper_genres[genre].append(tb_code)
        for genre in tb_report["paper_missing_from_treebank_metadata"]:
            paper_missing_from_metadata[genre].append(tb_code)
        for genre in tb_report["observed_extra_vs_paper"]:
            observed_extras_vs_paper[genre].append(tb_code)

    return {
        "paths": {
            "config": str(paths.config_path),
            "split_map": str(paths.split_map_path),
        },
        "summary": {
            "treebanks_audited": sorted(treebank_reports),
            "unsupported_paper_genres": {
                genre: sorted(treebanks)
                for genre, treebanks in sorted(unsupported_paper_genres.items())
            },
            "paper_missing_from_treebank_metadata": {
                genre: sorted(treebanks)
                for genre, treebanks in sorted(paper_missing_from_metadata.items())
            },
            "observed_extras_vs_paper": {
                genre: sorted(treebanks)
                for genre, treebanks in sorted(observed_extras_vs_paper.items())
            },
        },
        "treebanks": treebank_reports,
    }


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Paper Gold Inventory Audit",
        "",
        "## Scope",
        f"- Config: `{report['paths']['config']}`",
        f"- Sentence split map: `{report['paths']['split_map']}`",
        f"- Treebanks audited: {', '.join(report['summary']['treebanks_audited'])}",
        (
            "- Paper genres unsupported by sentence-level gold: "
            f"{json.dumps(report['summary']['unsupported_paper_genres'], ensure_ascii=False, sort_keys=True)}"
        ),
        (
            "- Paper genres missing from current treebank metadata: "
            f"{json.dumps(report['summary']['paper_missing_from_treebank_metadata'], ensure_ascii=False, sort_keys=True)}"
        ),
        (
            "- Sentence-level genres not present in paper mapping: "
            f"{json.dumps(report['summary']['observed_extras_vs_paper'], ensure_ascii=False, sort_keys=True)}"
        ),
        "",
        "## Per-Treebank",
    ]

    for tb_code, tb_report in report["treebanks"].items():
        lines.extend(
            [
                f"### {tb_code}",
                f"- Language: {tb_report['language']}",
                f"- Paper genres: {tb_report['paper_genres']}",
                f"- Current treebank metadata genres: {tb_report['metadata_genres']}",
                f"- Observed sentence-level genres: {tb_report['observed_sentence_genres']}",
                f"- Observed sentence-level counts: {tb_report['observed_sentence_counts']}",
                f"- Paper genres missing from sentence-level gold: {tb_report['paper_missing_from_sentence_gold']}",
                f"- Paper genres missing from current treebank metadata: {tb_report['paper_missing_from_treebank_metadata']}",
                f"- Metadata genres missing from sentence-level gold: {tb_report['metadata_missing_from_sentence_gold']}",
                f"- Sentence-level genres not in paper mapping: {tb_report['observed_extra_vs_paper']}",
                f"- Sentence counts: total={tb_report['total_sentences']}, labeled={tb_report['labeled_sentences']}, unlabeled={tb_report['unlabeled_sentences']}",
                "- Split breakdown:",
            ]
        )
        for split_name, split_report in tb_report["split_breakdown"].items():
            lines.append(
                "  - "
                f"{split_name}: total={split_report['total_sentences']}, "
                f"labeled={split_report['labeled_sentences']}, "
                f"unlabeled={split_report['unlabeled_sentences']}, "
                f"genres={split_report['observed_counts']}"
            )
        lines.append("")

    return "\n".join(lines)


def write_paper_gold_inventory_audit(
    report: Dict[str, Any],
    paths: PaperGoldInventoryAuditPaths,
) -> None:
    paths.json_out.parent.mkdir(parents=True, exist_ok=True)
    paths.json_out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    paths.markdown_out.write_text(build_markdown(report) + "\n")
