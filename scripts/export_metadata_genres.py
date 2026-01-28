#!/usr/bin/env python
"""Export genre labels directly from metadata without bootstrap.

Useful for treebanks with complete sentence-level metadata (like PUD)
where bootstrap labeling is not needed.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
from ud_genre_bootstrap.utils.config import load_config
from ud_genre_bootstrap.utils.data_loader import UDDataLoader
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_metadata_genres(config_path: Path, output_path: Path, treebank_filter: list = None):
    """Export genre labels from metadata for all sentences.

    Uses the same logic as the bootstrapper:
    1. Extract genres from sentence-level metadata (e.g., PUD patterns)
    2. Fall back to treebank-level metadata for single-genre treebanks (e.g., PoSTWITA)

    Args:
        config_path: Path to config file
        output_path: Output path for all_genres.parquet
        treebank_filter: Optional list of treebank codes to process
    """
    # Load config
    cfg = load_config(config_path)

    # Apply treebank filter - CLI overrides config
    if treebank_filter is None and cfg.include_treebanks:
        treebank_filter = cfg.include_treebanks
        logger.info(f"Using treebanks from config: {', '.join(treebank_filter)}")

    # Initialize components
    data_loader = UDDataLoader(
        ud_source=cfg.ud_source,
        ud_version=cfg.ud_version,
    )

    # Initialize genre mapper
    mapping_path = Path(cfg.genre_extraction.mapping_path) if cfg.genre_extraction.mapping_path else None
    patterns_path = cfg.genre_extraction.patterns_path
    if patterns_path:
        if isinstance(patterns_path, list):
            patterns_path = [Path(p) for p in patterns_path]
        else:
            patterns_path = Path(patterns_path)

    genre_mapper = GenreMapper(
        genre_mapping_path=mapping_path,
        metadata_patterns_path=patterns_path,
        canonical_genres=cfg.genre_extraction.canonical_genres,
    )

    # Extract genres from all sentences
    all_genre_data = []

    for tb_code, split, dataset in data_loader.iter_all_treebanks(treebank_filter=treebank_filter):
        logger.info(f"Processing {tb_code}:{split} ({len(dataset)} sentences)")

        # First pass: check if treebank has sentence-level metadata
        # (Same logic as bootstrapper._cluster_treebanks())
        genres_from_sentences = set()
        for sent in dataset:
            sent_genres = genre_mapper.extract_genres_from_metadata(sent, tb_code)
            genres_from_sentences.update(sent_genres)

        # Determine treebank-level genres
        if not genres_from_sentences:
            # No sentence-level metadata, use treebank-level
            raw_genres = data_loader.get_treebank_genres(tb_code)
            # Normalize genres using genre mapper (same as bootstrapper)
            treebank_genres = [
                genre_mapper.normalize_genre(g, tb_code) for g in raw_genres
            ]
            # Remove duplicates after normalization
            treebank_genres = list(set(treebank_genres))
        else:
            treebank_genres = sorted(genres_from_sentences)

        # Check if this is a single-genre treebank
        is_single_genre = len(treebank_genres) == 1

        if is_single_genre:
            # Single-genre treebank: label all sentences with that genre
            single_genre = treebank_genres[0]
            logger.info(f"  Single-genre treebank: all sentences labeled as '{single_genre}'")

            for idx, sentence in enumerate(dataset):
                sent_id = sentence.get('sent_id', f'{tb_code}_{split}_{idx}')
                all_genre_data.append({
                    'sent_id': sent_id,
                    'genre': single_genre,
                    'confidence': 1.0,
                    'method': 'single-genre-treebank',
                })
        else:
            # Multi-genre treebank: extract from sentence metadata
            for idx, sentence in enumerate(dataset):
                sent_id = sentence.get('sent_id', f'{tb_code}_{split}_{idx}')

                # Extract genres from metadata
                genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)

                if genres:
                    # Use first genre if multiple (most should have exactly one)
                    genre = genres[0]

                    all_genre_data.append({
                        'sent_id': sent_id,
                        'genre': genre,
                        'confidence': 1.0,  # Metadata-based = 100% confidence
                        'method': 'metadata',
                    })
                else:
                    # No metadata found
                    all_genre_data.append({
                        'sent_id': sent_id,
                        'genre': None,
                        'confidence': 0.0,
                        'method': 'missing',
                    })

    # Create DataFrame
    df = pd.DataFrame(all_genre_data)

    # Save as parquet
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Print statistics
    total = len(df)
    labeled = df[df['genre'].notna()].shape[0]
    missing = df[df['genre'].isna()].shape[0]

    logger.info(f"Exported {total} sentences")
    logger.info(f"  {labeled} with genres ({labeled/total:.1%})")
    logger.info(f"  {missing} without genres ({missing/total:.1%})")
    logger.info(f"Saved to: {output_path}")

    # Show genre distribution
    if labeled > 0:
        genre_counts = df[df['genre'].notna()]['genre'].value_counts()
        logger.info("\nGenre distribution:")
        for genre, count in genre_counts.items():
            logger.info(f"  {genre}: {count} ({count/labeled:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Export genre labels from metadata without bootstrap"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path for all_genres.parquet",
    )
    parser.add_argument(
        "--treebank",
        "-t",
        help="Specific treebank(s) to process (comma-separated)",
    )

    args = parser.parse_args()

    treebank_filter = None
    if args.treebank:
        treebank_filter = [tb.strip() for tb in args.treebank.split(",")]

    export_metadata_genres(args.config, args.output, treebank_filter)


if __name__ == "__main__":
    main()
