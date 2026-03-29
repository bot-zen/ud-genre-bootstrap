#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ud_genre_bootstrap.evaluation.parity_audit import (
    ParityAuditPaths,
    run_parity_audit,
    write_parity_audit,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the current strict paper-parity flow against the original "
            "ud-genre GMM+L-style pipeline on the same reconstructed UD 2.8 split."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/sweeps/2.8-apples-fixed-parity-uniform.yaml",
        help="Current paper-parity config to audit.",
    )
    parser.add_argument(
        "--sentence-split-map",
        default="configs/apples/paper-split-map-v2.8.parquet",
        help="Sentence split map used for the parity reconstruction.",
    )
    parser.add_argument(
        "--original-repo",
        default="../ud-genre",
        help="Path to the original ud-genre repository.",
    )
    parser.add_argument(
        "--ud-root",
        default="../huggingface/universal_dependencies/tools/ud-treebanks-v2.8",
        help="Path to the UD 2.8 checkout used by the original pipeline side.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/ud-genre-bootstrap_cache/2.8/bert-base-multilingual-cased_mean_64_-1-embeddings",
        help="Embedding cache directory shared by both sides of the audit.",
    )
    parser.add_argument(
        "--output-prefix",
        default="output/parity_audit/paper_parity_vs_original",
        help="Prefix for the generated .json and .md report files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prefix = Path(args.output_prefix)
    paths = ParityAuditPaths(
        config_path=Path(args.config),
        split_map_path=Path(args.sentence_split_map),
        original_repo=Path(args.original_repo).resolve(),
        ud_root=Path(args.ud_root).resolve(),
        cache_dir=Path(args.cache_dir),
        json_out=prefix.with_suffix(".json"),
        markdown_out=prefix.with_suffix(".md"),
    )
    report = run_parity_audit(paths)
    write_parity_audit(report, paths)
    print(
        json.dumps(
            {
                "json": str(paths.json_out),
                "markdown": str(paths.markdown_out),
                "current_scoring_treebanks": report["current"]["scoring_treebanks"],
                "original_test_treebanks_seen": len(report["original"]["test_treebanks_seen"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
