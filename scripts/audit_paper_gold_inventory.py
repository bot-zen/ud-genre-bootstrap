#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ud_genre_bootstrap.evaluation.paper_gold_inventory_audit import (
    PaperGoldInventoryAuditPaths,
    build_paper_gold_inventory_audit,
    write_paper_gold_inventory_audit,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit paper treebank genre inventories against current treebank metadata "
            "and sentence-level gold on the reconstructed paper test partition."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/sweeps/2.8-apples-fixed-parity-uniform.yaml",
        help="Paper-parity config to use for metadata extraction.",
    )
    parser.add_argument(
        "--sentence-split-map",
        default="configs/apples/paper-split-map-v2.8.parquet",
        help="Sentence split map used for the reconstructed paper test partition.",
    )
    parser.add_argument(
        "--treebank",
        action="append",
        default=None,
        help="Treebank code to audit. Repeat to restrict the audit to a subset.",
    )
    parser.add_argument(
        "--output-prefix",
        default="output/parity_audit/paper_gold_inventory_audit",
        help="Prefix for the generated .json and .md audit files.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    prefix = Path(args.output_prefix)
    paths = PaperGoldInventoryAuditPaths(
        config_path=Path(args.config),
        split_map_path=Path(args.sentence_split_map),
        json_out=prefix.with_suffix(".json"),
        markdown_out=prefix.with_suffix(".md"),
    )
    report = build_paper_gold_inventory_audit(paths, treebanks=args.treebank)
    write_paper_gold_inventory_audit(report, paths)
    print(
        json.dumps(
            {
                "json": str(paths.json_out),
                "markdown": str(paths.markdown_out),
                "treebanks": report["summary"]["treebanks_audited"],
                "unsupported_paper_genres": report["summary"][
                    "unsupported_paper_genres"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
