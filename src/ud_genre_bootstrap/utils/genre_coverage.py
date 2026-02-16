"""Utilities for analyzing genre coverage in treebanks.

This module provides shared functionality for:
- Extracting sentence-level genre metadata
- Computing coverage statistics
- Identifying fully/partially covered treebanks
- Grouping analysis by treebank (across all splits)

Used by: test-genre, evaluate, and coverage analysis commands.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class SplitCoverage:
    """Coverage statistics for a single treebank split."""

    treebank: str
    split: str
    total_sentences: int
    sentences_with_genre: int
    coverage: float  # 0.0 to 1.0
    genres: Set[str]
    genre_counts: Dict[str, int]

    def is_fully_covered(self, threshold: float = 0.95) -> bool:
        """Check if this split has sufficient coverage."""
        return self.coverage >= threshold

    @property
    def is_multi_genre(self) -> bool:
        """Check if this split has multiple genres."""
        return len(self.genres) >= 2


@dataclass
class TreebankCoverage:
    """Coverage statistics for an entire treebank (all splits combined)."""

    treebank: str
    splits: Dict[str, SplitCoverage]  # Maps split name -> coverage

    @property
    def all_splits(self) -> List[str]:
        """Get all split names."""
        return sorted(self.splits.keys())

    @property
    def total_sentences(self) -> int:
        """Total sentences across all splits."""
        return sum(split.total_sentences for split in self.splits.values())

    @property
    def total_with_genre(self) -> int:
        """Total sentences with genre metadata across all splits."""
        return sum(split.sentences_with_genre for split in self.splits.values())

    @property
    def overall_coverage(self) -> float:
        """Overall coverage across all splits."""
        if self.total_sentences == 0:
            return 0.0
        return self.total_with_genre / self.total_sentences

    @property
    def all_genres(self) -> Set[str]:
        """All genres found across all splits."""
        genres = set()
        for split in self.splits.values():
            genres.update(split.genres)
        return genres

    def is_fully_covered(self, threshold: float = 0.95) -> bool:
        """Check if all splits have sufficient coverage."""
        if not self.splits:
            return False
        return all(split.is_fully_covered(threshold) for split in self.splits.values())

    def is_partially_covered(self, threshold: float = 0.95) -> bool:
        """Check if some but not all splits have sufficient coverage."""
        if not self.splits:
            return False
        covered_splits = [split.is_fully_covered(threshold) for split in self.splits.values()]
        return any(covered_splits) and not all(covered_splits)

    def get_covered_splits(self, threshold: float = 0.95) -> List[str]:
        """Get list of splits with sufficient coverage."""
        return [
            split_name for split_name, split in self.splits.items()
            if split.is_fully_covered(threshold)
        ]

    def get_uncovered_splits(self, threshold: float = 0.95) -> List[str]:
        """Get list of splits with insufficient coverage."""
        return [
            split_name for split_name, split in self.splits.items()
            if not split.is_fully_covered(threshold)
        ]


class GenreCoverageAnalyzer:
    """Analyzes genre coverage across treebanks.

    This class provides shared functionality for extracting sentence-level
    genre metadata and computing coverage statistics. It's used by multiple
    commands (test-genre, evaluate, coverage reporting) to ensure consistent
    analysis across the codebase.
    """

    def __init__(self, data_loader, genre_mapper):
        """Initialize coverage analyzer.

        Args:
            data_loader: UDDataLoader instance for loading treebanks
            genre_mapper: GenreMapper instance for extracting genres
        """
        self.data_loader = data_loader
        self.genre_mapper = genre_mapper

    def analyze_split(
        self,
        treebank_code: str,
        split_name: str,
    ) -> Optional[SplitCoverage]:
        """Analyze genre coverage for a single treebank split.

        Args:
            treebank_code: Treebank identifier (e.g., 'en_pud')
            split_name: Split name ('train', 'dev', or 'test')

        Returns:
            SplitCoverage object or None if treebank cannot be loaded
        """
        try:
            sentence_iter = self.data_loader.iter_treebank_sentences(
                treebank_code,
                split_name,
                metadata_only=True,
            )
        except Exception as e:
            logger.warning(f"Could not load {treebank_code}:{split_name}: {e}")
            return None

        # Extract genres for each sentence
        total_sentences = 0
        sentences_with_genre = 0
        genre_counts = defaultdict(int)

        try:
            for sentence in sentence_iter:
                total_sentences += 1

                # Extract genre from metadata
                genres = self.genre_mapper.extract_genres_from_metadata(
                    sentence, treebank_code
                )

                if genres:
                    # Use first genre if multiple
                    primary_genre = genres[0]
                    genre_counts[primary_genre] += 1
                    sentences_with_genre += 1
        except Exception as e:
            logger.warning(f"Could not load {treebank_code}:{split_name}: {e}")
            return None

        coverage = sentences_with_genre / total_sentences if total_sentences > 0 else 0.0

        return SplitCoverage(
            treebank=treebank_code,
            split=split_name,
            total_sentences=total_sentences,
            sentences_with_genre=sentences_with_genre,
            coverage=coverage,
            genres=set(genre_counts.keys()),
            genre_counts=dict(genre_counts),
        )

    def analyze_treebank(
        self,
        treebank_code: str,
        splits: Optional[List[str]] = None,
    ) -> TreebankCoverage:
        """Analyze genre coverage for an entire treebank (all splits).

        Args:
            treebank_code: Treebank identifier (e.g., 'en_pud')
            splits: Specific splits to analyze, or None for all available

        Returns:
            TreebankCoverage object with analysis for all splits
        """
        if splits is None:
            splits = self.data_loader.get_available_splits(treebank_code)

        split_coverages = {}
        for split_name in splits:
            coverage = self.analyze_split(treebank_code, split_name)
            if coverage is not None:
                split_coverages[split_name] = coverage

        return TreebankCoverage(
            treebank=treebank_code,
            splits=split_coverages,
        )

    def analyze_treebanks(
        self,
        treebank_codes: List[str],
    ) -> Dict[str, TreebankCoverage]:
        """Analyze genre coverage for multiple treebanks.

        Args:
            treebank_codes: List of treebank identifiers

        Returns:
            Dict mapping treebank code -> TreebankCoverage
        """
        results = {}
        for tb_code in treebank_codes:
            results[tb_code] = self.analyze_treebank(tb_code)
        return results

    def get_fully_covered_treebanks(
        self,
        treebank_codes: List[str],
        threshold: float = 0.95,
    ) -> List[str]:
        """Get list of treebanks with full coverage across all splits.

        Args:
            treebank_codes: List of treebank identifiers to check
            threshold: Minimum coverage threshold (default: 0.95)

        Returns:
            List of treebank codes with full coverage
        """
        results = self.analyze_treebanks(treebank_codes)
        return [
            tb_code for tb_code, coverage in results.items()
            if coverage.is_fully_covered(threshold)
        ]

    def get_partially_covered_treebanks(
        self,
        treebank_codes: List[str],
        threshold: float = 0.95,
    ) -> Dict[str, TreebankCoverage]:
        """Get treebanks with partial coverage (some splits covered, some not).

        Args:
            treebank_codes: List of treebank identifiers to check
            threshold: Minimum coverage threshold (default: 0.95)

        Returns:
            Dict mapping treebank code -> TreebankCoverage for partially covered treebanks
        """
        results = self.analyze_treebanks(treebank_codes)
        return {
            tb_code: coverage for tb_code, coverage in results.items()
            if coverage.is_partially_covered(threshold)
        }

    def extract_sentence_metadata(
        self,
        treebank_codes: List[str],
    ) -> Dict[Tuple[str, str, str], str]:
        """Extract sentence-level genre metadata for multiple treebanks.

        This is used by evaluation and other commands that need sentence-level
        genre labels for cross-validation.

        Args:
            treebank_codes: List of treebank identifiers

        Returns:
            Dict mapping (treebank, split, sent_id) -> genre
        """
        sentence_metadata = {}

        for tb_code in treebank_codes:
            splits = self.data_loader.get_available_splits(tb_code)

            for split_name in splits:
                try:
                    sentence_iter = self.data_loader.iter_treebank_sentences(
                        tb_code,
                        split_name,
                        metadata_only=True,
                    )
                except Exception as e:
                    logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                    continue

                try:
                    for idx, sentence in enumerate(sentence_iter):
                        sent_id = sentence.get('sent_id', f'{tb_code}_{split_name}_{idx}')
                        genres = self.genre_mapper.extract_genres_from_metadata(
                            sentence, tb_code
                        )

                        if genres:
                            primary_genre = genres[0]
                            sentence_metadata[(tb_code, split_name, sent_id)] = primary_genre
                except Exception as e:
                    logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                    continue

        return sentence_metadata

    def print_coverage_report(
        self,
        treebank_codes: List[str],
        threshold: float = 0.95,
        show_splits: bool = True,
    ):
        """Print a formatted coverage report for treebanks.

        Args:
            treebank_codes: List of treebank identifiers
            threshold: Minimum coverage threshold for "fully covered"
            show_splits: Whether to show per-split details
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        results = self.analyze_treebanks(treebank_codes)

        # Categorize treebanks
        fully_covered = []
        partially_covered = []
        not_covered = []

        for tb_code, coverage in results.items():
            if coverage.is_fully_covered(threshold):
                fully_covered.append((tb_code, coverage))
            elif coverage.is_partially_covered(threshold):
                partially_covered.append((tb_code, coverage))
            else:
                not_covered.append((tb_code, coverage))

        # Summary
        console.print(f"\n[bold]Genre Coverage Analysis[/bold]")
        console.print(f"Threshold: {threshold * 100:.0f}%\n")

        console.print(f"[green]✓ Fully covered:[/green] {len(fully_covered)} treebanks")
        console.print(f"[yellow]⚠ Partially covered:[/yellow] {len(partially_covered)} treebanks")
        console.print(f"[red]✗ Not covered:[/red] {len(not_covered)} treebanks")

        # Detailed tables
        if fully_covered:
            table = Table(title="Fully Covered Treebanks", show_header=True)
            table.add_column("Treebank", style="cyan")
            table.add_column("Splits", style="blue")
            table.add_column("Sentences", style="magenta", justify="right")
            table.add_column("Coverage", style="green", justify="right")
            table.add_column("Genres", style="yellow")

            for tb_code, coverage in sorted(fully_covered):
                splits_str = ", ".join(coverage.all_splits)
                genres_str = ", ".join(sorted(coverage.all_genres))
                table.add_row(
                    tb_code,
                    splits_str,
                    str(coverage.total_sentences),
                    f"{coverage.overall_coverage * 100:.1f}%",
                    genres_str,
                )

            console.print()
            console.print(table)

        if partially_covered:
            table = Table(title="Partially Covered Treebanks", show_header=True)
            table.add_column("Treebank", style="cyan")
            table.add_column("Covered Splits", style="green")
            table.add_column("Uncovered Splits", style="red")
            table.add_column("Overall Coverage", style="yellow", justify="right")

            for tb_code, coverage in sorted(partially_covered):
                covered = ", ".join(coverage.get_covered_splits(threshold))
                uncovered = ", ".join(coverage.get_uncovered_splits(threshold))
                table.add_row(
                    tb_code,
                    covered or "—",
                    uncovered or "—",
                    f"{coverage.overall_coverage * 100:.1f}%",
                )

            console.print()
            console.print(table)
