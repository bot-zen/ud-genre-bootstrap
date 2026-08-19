"""Command-line interface for ud-genre-bootstrap."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ud_genre_bootstrap.bootstrapping import GenreBootstrapper
from ud_genre_bootstrap.evaluation import CrossValidator
from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.release_artifacts import (
    list_release_upload_files,
    prepare_release_directory,
    publish_release_directory_to_hf_git,
    upload_release_directory_to_hub,
)
from ud_genre_bootstrap.utils.release_identity import (
    resolve_release_hf_branches,
    resolve_release_hf_repo,
    resolve_release_hf_revisions,
    resolve_release_identity,
)
from ud_genre_bootstrap.utils.sentence_refs import qualify_sentence_ref

# Create Typer app
app = typer.Typer(
    name="ud-genre-bootstrap",
    help="Bootstrap genre classification for Universal Dependencies treebanks",
    add_completion=False,
)

# Rich console
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


PAPER_TREEBANK_MAPPING_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "ud-genre-tb-genres.json"
)


def normalize_paper_treebank_key(key: str) -> str:
    """Normalize paper treebank keys for fuzzy matching across metadata variants."""
    alias_map = {
        "portugese": "portuguese",
    }

    parts = [part.strip() for part in (key or "").split("/", 1)]
    if len(parts) != 2:
        parts = [key or "", ""]

    normalized_parts = []
    for part in parts:
        lowered = part.strip().lower()
        lowered = alias_map.get(lowered, lowered)
        normalized_parts.append(re.sub(r"[^a-z0-9]+", "", lowered))

    return "/".join(normalized_parts)


def build_paper_treebank_key(treebank_code: str, metadata: Optional[Dict]) -> Optional[str]:
    """Build a paper-style `Language/Treebank` key from current treebank metadata."""
    metadata = metadata or {}

    language = metadata.get("language")
    treebank_name = metadata.get("treebank_name")

    dirname = str(metadata.get("dirname") or "").strip()
    if dirname.startswith("UD_"):
        dirname_body = dirname[3:]
        if "-" in dirname_body:
            dirname_language, dirname_treebank = dirname_body.split("-", 1)
            if not language:
                language = dirname_language.replace("_", " ")
            if not treebank_name:
                treebank_name = dirname_treebank

    if not treebank_name and "_" in treebank_code:
        treebank_name = treebank_code.split("_", 1)[1]

    if not language or not treebank_name:
        return None

    return f"{language}/{treebank_name}"


def load_paper_treebank_mapping(
    mapping_path: Optional[Path] = None,
) -> Dict[str, Dict[str, str]]:
    """Load the vendored paper treebank mapping file."""
    mapping_path = mapping_path or PAPER_TREEBANK_MAPPING_PATH
    if not mapping_path.exists():
        raise ValueError(f"Paper treebank mapping not found: {mapping_path}")

    with open(mapping_path, encoding="utf-8") as handle:
        return json.load(handle)


def resolve_paper_evaluation_treebank_genres(
    data_loader,
    mapping_path: Optional[Path] = None,
) -> Dict[str, List[str]]:
    """Resolve current treebank IDs to paper-mapped global genre inventories."""
    paper_mapping = load_paper_treebank_mapping(mapping_path)
    normalized_to_entry = {
        normalize_paper_treebank_key(key): sorted(set(value.values()))
        for key, value in paper_mapping.items()
    }
    metadata = getattr(data_loader, "metadata", {}) or {}

    matched_genres: Dict[str, List[str]] = {}
    for treebank_code in data_loader.get_treebank_codes():
        paper_key = build_paper_treebank_key(treebank_code, metadata.get(treebank_code))
        if not paper_key:
            continue
        normalized_key = normalize_paper_treebank_key(paper_key)
        if normalized_key in normalized_to_entry:
            matched_genres[treebank_code] = normalized_to_entry[normalized_key]

    return matched_genres


def resolve_paper_evaluation_treebank_ids(
    data_loader,
    mapping_path: Optional[Path] = None,
) -> set[str]:
    """Resolve current treebank IDs that belong to the original paper eval scope."""
    return set(
        resolve_paper_evaluation_treebank_genres(
            data_loader,
            mapping_path=mapping_path,
        )
    )


def get_treebank_descriptor_split_keys(treebank_info: Dict) -> List[Tuple[str, str]]:
    """Resolve split keys from split-level or treebank-level evaluation descriptors."""
    if "split_keys" in treebank_info:
        return [tuple(split_key) for split_key in treebank_info.get("split_keys", [])]

    split_name = treebank_info.get("split")
    if split_name is None:
        return []

    return [(treebank_info["treebank"], split_name)]


def collect_treebank_descriptor_split_keys(treebanks: List[Dict]) -> set[Tuple[str, str]]:
    """Collect split keys referenced by split- or treebank-level descriptors."""
    split_keys: set[Tuple[str, str]] = set()
    for treebank_info in treebanks:
        split_keys.update(get_treebank_descriptor_split_keys(treebank_info))
    return split_keys


def filter_treebank_descriptors_by_available_splits(
    treebanks: List[Dict],
    available_splits: set[Tuple[str, str]],
) -> List[Dict]:
    """Drop unavailable split keys while preserving descriptor shape."""
    filtered_treebanks: List[Dict] = []
    for treebank_info in treebanks:
        kept_split_keys = sorted(
            split_key
            for split_key in get_treebank_descriptor_split_keys(treebank_info)
            if split_key in available_splits
        )
        if not kept_split_keys:
            continue

        updated_info = dict(treebank_info)
        if "split_keys" in updated_info:
            updated_info["split_keys"] = kept_split_keys
        else:
            updated_info["split"] = kept_split_keys[0][1]
        filtered_treebanks.append(updated_info)

    return filtered_treebanks


def apply_treebank_exclusions(cfg: Config, data_loader, treebank_filter: Optional[List[str]] = None) -> Optional[List[str]]:
    """Apply config exclusions to treebank filter.

    Args:
        cfg: Configuration object
        data_loader: UDDataLoader instance
        treebank_filter: Optional list of treebanks to include (from --treebank flag)

    Returns:
        Filtered list of treebanks (never None if exclusions specified)
    """
    if not cfg.exclude_treebanks:
        return treebank_filter

    # If no treebank filter, start with all treebanks
    if not treebank_filter:
        treebank_filter = data_loader.get_treebank_codes()

    # Remove exclusions
    excluded = [tb for tb in treebank_filter if tb in cfg.exclude_treebanks]
    filtered = [tb for tb in treebank_filter if tb not in cfg.exclude_treebanks]

    if excluded:
        console.print(f"[yellow]Excluded {len(excluded)} treebanks from config: {', '.join(excluded)}[/yellow]")

    return filtered


def parse_treebank_csv(treebank_csv: str) -> List[str]:
    """Parse comma-separated treebank codes into a normalized list."""
    return [tb.strip() for tb in treebank_csv.split(",") if tb.strip()]


def parse_inline_treebank_sets(set_definitions: Optional[List[str]]) -> Dict[str, List[str]]:
    """Parse inline named treebank sets from CLI options.

    Supports definitions in the form `name=tb1,tb2` (or `name:tb1,tb2`).
    """
    if not set_definitions:
        return {}

    parsed_sets: Dict[str, List[str]] = {}

    for raw_definition in set_definitions:
        if "=" in raw_definition:
            set_name, treebanks_csv = raw_definition.split("=", 1)
        elif ":" in raw_definition:
            set_name, treebanks_csv = raw_definition.split(":", 1)
        else:
            raise ValueError(
                f"Invalid --treebank-set '{raw_definition}'. Use 'name=tb1,tb2,...'"
            )

        set_name = set_name.strip()
        if not set_name:
            raise ValueError(
                f"Invalid --treebank-set '{raw_definition}'. Set name is empty."
            )
        if set_name in parsed_sets:
            raise ValueError(
                f"Duplicate evaluation set name '{set_name}' in --treebank-set."
            )

        treebanks = parse_treebank_csv(treebanks_csv)
        if not treebanks:
            raise ValueError(
                f"Invalid --treebank-set '{raw_definition}'. No treebanks provided."
            )

        parsed_sets[set_name] = treebanks

    return parsed_sets


def resolve_config_treebank_sets(
    cfg: Config,
    set_names: Optional[List[str]],
) -> Dict[str, List[str]]:
    """Resolve named evaluation sets from configuration."""
    if not set_names:
        return {}

    available_sets = cfg.evaluation.treebank_sets or {}
    missing_sets = [set_name for set_name in set_names if set_name not in available_sets]
    if missing_sets:
        raise ValueError(
            f"Unknown --set value(s): {', '.join(missing_sets)}. "
            f"Available sets: {', '.join(sorted(available_sets.keys())) or '[none]'}"
        )

    return {set_name: available_sets[set_name] for set_name in set_names}


def build_treebank_eval_stats(multi_genre_treebanks: List[Dict]) -> Dict[str, Dict[str, int]]:
    """Aggregate per-treebank stats used for evaluation set planning."""
    stats: Dict[str, Dict] = {}
    for treebank in multi_genre_treebanks:
        tb_code = treebank["treebank"]
        if tb_code not in stats:
            stats[tb_code] = {
                "splits": 0,
                "sentences": 0,
                "genres": set(),
            }

        stats[tb_code]["splits"] += 1
        stats[tb_code]["sentences"] += int(treebank.get("sentence_count", 0))
        stats[tb_code]["genres"].update(treebank.get("genres", []))

    normalized_stats: Dict[str, Dict[str, int]] = {}
    for tb_code, tb_stats in stats.items():
        normalized_stats[tb_code] = {
            "splits": int(tb_stats["splits"]),
            "sentences": int(tb_stats["sentences"]),
            "virtual_splits": len(tb_stats["genres"]),
        }

    return normalized_stats


def build_progressive_treebank_sets(
    multi_genre_treebanks: List[Dict],
    min_size: int,
    step: int = 1,
) -> Dict[str, List[str]]:
    """Build cumulative evaluation sets ordered by virtual-split potential.

    Treebanks are ordered by:
    1. Descending number of potential virtual splits (unique genres)
    2. Descending sentence count
    3. Treebank code (stable tie-breaker)
    """
    if step < 1:
        raise ValueError("progressive step must be >= 1")

    treebank_stats = build_treebank_eval_stats(multi_genre_treebanks)
    ordered_treebanks = sorted(
        treebank_stats.items(),
        key=lambda item: (
            -item[1]["virtual_splits"],
            -item[1]["sentences"],
            item[0],
        ),
    )
    ordered_codes = [tb_code for tb_code, _ in ordered_treebanks]

    if not ordered_codes:
        return {}

    start_size = max(1, min_size)
    if start_size > len(ordered_codes):
        return {}

    progressive_sets: Dict[str, List[str]] = {}
    for size in range(start_size, len(ordered_codes) + 1, step):
        progressive_sets[f"progressive_top_{size}"] = ordered_codes[:size]

    if len(ordered_codes) not in {len(treebanks) for treebanks in progressive_sets.values()}:
        progressive_sets[f"progressive_top_{len(ordered_codes)}"] = ordered_codes

    return progressive_sets


def check_evaluation_fold_feasibility(
    multi_genre_treebanks: List[Dict],
    n_folds: int,
    group_by: Optional[str],
) -> Tuple[bool, str]:
    """Check whether a treebank subset can support requested n-fold CV."""
    n_items = len(multi_genre_treebanks)
    if n_items < n_folds:
        return (
            False,
            f"needs at least {n_folds} multi-genre treebank splits, found {n_items}",
        )

    if group_by == "treebank":
        n_groups = len({tb["treebank"] for tb in multi_genre_treebanks})
        if n_groups < n_folds:
            return (
                False,
                f"group_by=treebank requires at least {n_folds} treebanks, found {n_groups}",
            )
    elif group_by == "language":
        n_groups = len({tb["language"] for tb in multi_genre_treebanks})
        if n_groups < n_folds:
            return (
                False,
                f"group_by=language requires at least {n_folds} languages, found {n_groups}",
            )

    return True, ""


def normalize_evaluation_protocol(
    protocol: Optional[str],
    default_protocol: str = "generalization",
) -> str:
    """Normalize evaluation protocol from CLI/config inputs."""
    raw_protocol = default_protocol if protocol is None else protocol
    normalized = (raw_protocol or "generalization").strip().lower()
    alias_map = {
        "paper": "paper_parity",
        "parity": "paper_parity",
        "generalisation": "generalization",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized not in {"generalization", "paper_parity"}:
        raise ValueError(
            f"Invalid evaluation protocol '{raw_protocol}'. "
            "Use 'generalization' or 'paper_parity'."
        )
    return normalized


def normalize_anchor_mode(anchor_mode: Optional[str], default_mode: str = "strict") -> str:
    """Normalize evaluation anchor mode from CLI/config inputs."""
    mode = default_mode if anchor_mode is None else anchor_mode
    normalized = (mode or "strict").strip().lower()
    if normalized not in {"strict", "parity"}:
        raise ValueError(
            f"Invalid anchor mode '{mode}'. Use 'strict' or 'parity'."
        )
    return normalized


def normalize_anchor_pool_policy(
    anchor_pool_policy: Optional[str],
    default_policy: str = "auto",
    *,
    anchor_mode: str = "strict",
) -> str:
    """Normalize evaluation anchor-pool policy from CLI/config inputs."""
    raw_policy = default_policy if anchor_pool_policy is None else anchor_pool_policy
    normalized = (raw_policy or "auto").strip().lower()
    alias_map = {
        "train_virtual_only": "train_virtual",
        "single_genre_only": "single_genre",
        "virtual_only": "train_virtual",
    }
    normalized = alias_map.get(normalized, normalized)
    if normalized == "auto":
        return "combined" if anchor_mode == "parity" else "train_virtual"
    if normalized not in {"train_virtual", "single_genre", "combined"}:
        raise ValueError(
            f"Invalid anchor pool policy '{raw_policy}'. "
            "Use 'auto', 'train_virtual', 'single_genre', or 'combined'."
        )
    return normalized


def _format_protocol_treebanks(split_keys: List[Tuple[str, str]], limit: int = 5) -> str:
    """Format split keys as a compact treebank list."""
    treebanks = sorted({tb for tb, _split in split_keys})
    if not treebanks:
        return "none"
    shown = treebanks[:limit]
    if len(treebanks) > limit:
        shown.append(f"... +{len(treebanks) - limit} more")
    return ", ".join(shown)


def _format_protocol_split_keys(split_keys: List[Tuple[str, str]], limit: int = 5) -> str:
    """Format split keys compactly for human-readable diagnostics."""
    unique_keys = sorted(set(split_keys))
    if not unique_keys:
        return "none"
    shown = [f"{tb}:{split}" for tb, split in unique_keys[:limit]]
    if len(unique_keys) > limit:
        shown.append(f"... +{len(unique_keys) - limit} more")
    return ", ".join(shown)


def build_fixed_partition_protocol_report(
    *,
    protocol: str,
    anchor_partitions: List[str],
    test_partition: str,
    requested_anchor_split_keys: List[Tuple[str, str]],
    requested_test_split_keys: List[Tuple[str, str]],
    source_missing_anchor_split_keys: Optional[List[Tuple[str, str]]] = None,
    source_missing_test_split_keys: Optional[List[Tuple[str, str]]] = None,
    load_error_split_keys: Optional[List[Tuple[str, str]]] = None,
    no_metadata_test_split_keys: Optional[List[Tuple[str, str]]] = None,
    dropped_anchor_split_keys: Optional[List[Tuple[str, str]]] = None,
    dropped_test_split_keys: Optional[List[Tuple[str, str]]] = None,
    dropped_single_anchor_split_keys: Optional[List[Tuple[str, str]]] = None,
    missing_anchor_genres: Optional[List[str]] = None,
    overlap_sentences: int = 0,
) -> Dict[str, object]:
    """Build a compact fixed-partition protocol report for CLI output/results."""
    requested_anchor_split_keys = sorted(set(requested_anchor_split_keys))
    requested_test_split_keys = sorted(set(requested_test_split_keys))
    source_missing_anchor_split_keys = sorted(set(source_missing_anchor_split_keys or []))
    source_missing_test_split_keys = sorted(set(source_missing_test_split_keys or []))
    load_error_split_keys = sorted(set(load_error_split_keys or []))
    no_metadata_test_split_keys = sorted(set(no_metadata_test_split_keys or []))
    dropped_anchor_split_keys = sorted(set(dropped_anchor_split_keys or []))
    dropped_test_split_keys = sorted(set(dropped_test_split_keys or []))
    dropped_single_anchor_split_keys = sorted(set(dropped_single_anchor_split_keys or []))
    missing_anchor_genres = sorted(set(missing_anchor_genres or []))

    notes: List[str] = []
    deviations: List[str] = []

    if protocol == "paper_parity":
        notes.append(
            f"Paper-parity anchor policy: same-partition single-genre anchors from '{test_partition}'."
        )
        if overlap_sentences > 0:
            notes.append(
                f"Anchor/test overlap: {overlap_sentences} sentence(s) shared by design."
            )
    else:
        notes.append(
            f"Generalization fixed holdout: anchors={', '.join(anchor_partitions)} -> test={test_partition}."
        )
        if overlap_sentences > 0:
            deviations.append(
                f"Anchor/test overlap outside parity protocol: {overlap_sentences} sentence(s)."
            )

    if source_missing_anchor_split_keys:
        deviations.append(
            "Split-map anchor keys missing from UD source: "
            f"{len(source_missing_anchor_split_keys)} split(s) "
            f"(treebanks: {_format_protocol_treebanks(source_missing_anchor_split_keys)})"
        )
    if source_missing_test_split_keys:
        deviations.append(
            "Split-map test keys missing from UD source: "
            f"{len(source_missing_test_split_keys)} split(s) "
            f"(treebanks: {_format_protocol_treebanks(source_missing_test_split_keys)})"
        )
    if load_error_split_keys:
        deviations.append(
            "Load errors while reading split-map-selected data: "
            f"{len(load_error_split_keys)} split(s) "
            f"({_format_protocol_split_keys(load_error_split_keys)})"
        )
    if no_metadata_test_split_keys:
        deviations.append(
            "Split-map-selected test splits without usable genre metadata: "
            f"{len(no_metadata_test_split_keys)} split(s) "
            f"({_format_protocol_split_keys(no_metadata_test_split_keys)})"
        )
    if dropped_anchor_split_keys:
        deviations.append(
            "Anchor splits removed after embedding filtering: "
            f"{len(dropped_anchor_split_keys)} split(s) "
            f"({_format_protocol_split_keys(dropped_anchor_split_keys)})"
        )
    if dropped_test_split_keys:
        deviations.append(
            "Test splits removed after embedding filtering: "
            f"{len(dropped_test_split_keys)} split(s) "
            f"({_format_protocol_split_keys(dropped_test_split_keys)})"
        )
    if dropped_single_anchor_split_keys:
        deviations.append(
            "Single-genre anchor candidates removed after embedding filtering: "
            f"{len(dropped_single_anchor_split_keys)} split(s) "
            f"({_format_protocol_split_keys(dropped_single_anchor_split_keys)})"
        )
    if missing_anchor_genres:
        deviations.append(
            "Expected test genres without anchor support: "
            f"{', '.join(missing_anchor_genres)}"
        )

    return {
        "protocol_scope": {
            "anchor_partitions": list(anchor_partitions),
            "test_partition": test_partition,
            "requested_anchor_split_keys": len(requested_anchor_split_keys),
            "requested_test_split_keys": len(requested_test_split_keys),
            "overlap_sentences": overlap_sentences,
        },
        "protocol_notes": notes,
        "protocol_deviations": deviations,
    }


def safe_label_for_filename(label: str) -> str:
    """Normalize labels for safe file names."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def build_exported_genre_lookup(df) -> Tuple[Dict, bool]:
    """Build a sentence-level genre lookup from ``all_genres.parquet``."""
    key_columns = {"treebank", "split", "sent_id"}
    if key_columns.issubset(df.columns):
        lookup = {
            qualify_sentence_ref(row.treebank, row.split, row.sent_id): row.genre
            for row in df[["treebank", "split", "sent_id", "genre"]].itertuples(index=False)
        }
        return lookup, True

    lookup = {
        str(row.sent_id): row.genre
        for row in df[["sent_id", "genre"]].itertuples(index=False)
    }
    return lookup, False


def load_config_from_path(config_path: Optional[Path]) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to config YAML file

    Returns:
        Config object
    """
    from ud_genre_bootstrap.utils.config import load_config

    if config_path:
        console.print(f"[blue]Loading config from:[/blue] {config_path}")
        cfg = load_config(config_path)
        setattr(cfg, "_config_path", str(config_path))
        return cfg
    else:
        console.print("[blue]Using default configuration[/blue]")
        return Config()


@app.command()
def run(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to process (e.g., en_ewt or en_ewt,de_gsd). If not specified, uses all treebanks or config include_treebanks.",
    ),
):
    """Run the full bootstrapping pipeline.

    This executes all stages: embedding, clustering, labeling, and export.
    """
    console.print("\n[bold cyan]UD Genre Bootstrap Pipeline[/bold cyan]")
    console.print("=" * 60)

    try:
        # Load configuration
        cfg = load_config_from_path(config)

        if output:
            cfg.output.genres_path = str(output)
            console.print(f"[blue]Output directory:[/blue] {output}")

        # Initialize bootstrapper
        console.print("\n[yellow]Initializing bootstrapper...[/yellow]")
        bootstrapper = GenreBootstrapper(cfg)

        # Parse treebank filter (comma-separated)
        # CLI flag takes precedence over config
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
            console.print(f"[blue]Processing treebanks:[/blue] {', '.join(treebank_filter)}")
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            console.print(f"[blue]Processing treebanks from config:[/blue] {', '.join(treebank_filter)}")
        else:
            console.print("[blue]Processing all treebanks[/blue]")

        # Apply config exclusions
        treebank_filter = apply_treebank_exclusions(cfg, bootstrapper.data_loader, treebank_filter)

        # Run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running bootstrap pipeline...", total=None)
            results = bootstrapper.fit(treebank_filter=treebank_filter)

        # Display results
        console.print("\n[bold green]✓ Pipeline complete![/bold green]")
        _display_results(results)

        # Save cluster results using shared helper function.
        embeddings_by_tb = getattr(bootstrapper, "last_embeddings_by_tb", {}) or {}
        if not embeddings_by_tb:
            # Fallback for stubs/tests that don't expose cached embeddings.
            for tb_key in bootstrapper.treebank_clusters.keys():
                if len(tb_key) == 3:
                    tb_code, split, _genre_tag = tb_key
                else:
                    tb_code, split = tb_key
                cache_key = f"{tb_code}_{split}"
                if (
                    hasattr(bootstrapper.embedding_generator, "embedding_cache")
                    and cache_key in bootstrapper.embedding_generator.embedding_cache
                ):
                    embeddings_by_tb[(tb_code, split)] = bootstrapper.embedding_generator.embedding_cache[cache_key]

        _save_cluster_results(bootstrapper, embeddings_by_tb, cfg.output.genres_path)

        if cfg.output.push_to_hub:
            console.print("\n[yellow]Uploading release artifacts to Hugging Face Hub...[/yellow]")
            bootstrapper.push_to_hub(
                resolve_release_hf_repo(cfg),
                resolve_release_hf_revisions(cfg),
            )

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Pipeline failed")
        raise typer.Exit(1)


@app.command()
def embed(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to embed (e.g., en_ewt or en_ewt,de_gsd). If not specified, embeds all.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Embedding model to use (overrides config)",
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="Push embeddings to HuggingFace Hub after generation",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing cached embeddings",
    ),
):
    """Generate sentence embeddings for UD treebanks.

    Creates embeddings for all sentences in the specified treebank(s).
    """
    console.print("\n[bold cyan]UD Sentence Embedding Generation[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        if model:
            cfg.embeddings.model = model
            console.print(f"[blue]Using model:[/blue] {model}")

        # Initialize bootstrapper (which includes embedding generator)
        bootstrapper = GenreBootstrapper(cfg)

        # Parse treebank filter (comma-separated)
        # CLI flag takes precedence over config
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            console.print(f"[blue]Using treebanks from config:[/blue] {', '.join(treebank_filter)}")

        # Apply config exclusions
        treebank_filter = apply_treebank_exclusions(cfg, bootstrapper.data_loader, treebank_filter)

        if overwrite:
            console.print("[yellow]⚠ Overwrite mode enabled - will regenerate all embeddings[/yellow]")

        if treebank_filter:
            if len(treebank_filter) == 1:
                console.print(f"\n[yellow]Generating embeddings for {treebank_filter[0]}...[/yellow]")
            else:
                console.print(f"\n[yellow]Generating embeddings for {len(treebank_filter)} treebanks: {', '.join(treebank_filter[:5])}{'...' if len(treebank_filter) > 5 else ''}[/yellow]")
        else:
            console.print("\n[yellow]Generating embeddings for all treebanks...[/yellow]")

        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_filter, overwrite=overwrite)

        console.print(f"\n[bold green]✓ Generated embeddings for {len(embeddings_by_tb)} treebank splits[/bold green]")

        if push:
            console.print("\n[yellow]Pushing to HuggingFace Hub...[/yellow]")
            # TODO: Implement HF Hub push
            console.print("[red]HF Hub push not yet implemented[/red]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Embedding failed")
        raise typer.Exit(1)


@app.command()
def cluster(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to cluster (e.g., en_ewt or en_ewt,de_gsd). If not specified, clusters all.",
    ),
    clusters: Optional[Path] = typer.Option(
        None,
        "--clusters",
        help="Path to existing cluster directory. If provided, loads existing clusters and only re-clusters specified --treebank(s).",
        exists=True,
        dir_okay=True,
    ),
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu",
        help="Use GPU acceleration for clustering (requires cuML for K-Means)",
    ),
):
    """Cluster treebank sentences into genre groups.

    Uses GMM or K-Means clustering to group sentences based on embeddings.

    When --clusters is provided with --treebank, only re-clusters the specified
    treebank(s) and keeps all other clusters unchanged. Useful for updating
    clusters after changing genre mappings for specific treebanks.
    """
    console.print("\n[bold cyan]UD Treebank Clustering[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Override device setting if --use-gpu flag is provided
        if use_gpu:
            cfg.clustering.device = "cuda"

        bootstrapper = GenreBootstrapper(cfg)

        # Parse treebank filter (comma-separated)
        # CLI flag takes precedence over config
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            console.print(f"[blue]Using treebanks from config:[/blue] {', '.join(treebank_filter)}")

        # Apply config exclusions
        treebank_filter = apply_treebank_exclusions(cfg, bootstrapper.data_loader, treebank_filter)

        # Load existing cluster state if --clusters is provided
        existing_embeddings_by_tb = {}
        existing_treebank_clusters = {}
        if clusters:
            cluster_state_path = clusters / "cluster_state.pkl"
            if cluster_state_path.exists():
                console.print(f"\n[blue]Loading existing cluster state from {cluster_state_path}...[/blue]")
                existing_embeddings_by_tb = bootstrapper.load_cluster_state(cluster_state_path)
                existing_treebank_clusters = dict(bootstrapper.treebank_clusters)
                console.print(f"[green]✓ Loaded {len(existing_treebank_clusters)} existing cluster(s)[/green]")

                if not treebank_filter:
                    console.print(f"[yellow]⚠ Warning: --clusters provided but no --treebank specified.[/yellow]")
                    console.print(f"[yellow]  Will re-cluster all treebanks. Use --treebank to selectively update.[/yellow]")
            else:
                console.print(f"[yellow]⚠ Warning: cluster_state.pkl not found in {clusters}[/yellow]")

        if treebank_filter:
            if len(treebank_filter) == 1:
                action = "Re-clustering" if existing_treebank_clusters else "Clustering"
                console.print(f"\n[yellow]{action} {treebank_filter[0]}...[/yellow]")
            else:
                action = "Re-clustering" if existing_treebank_clusters else "Clustering"
                console.print(f"\n[yellow]{action} {len(treebank_filter)} treebanks: {', '.join(treebank_filter[:5])}{'...' if len(treebank_filter) > 5 else ''}[/yellow]")
        else:
            console.print("\n[yellow]Clustering all treebanks...[/yellow]")

        # Generate/load embeddings (only for treebanks being clustered)
        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_filter)

        # Cluster treebanks (only the filtered ones)
        console.print("\n[yellow]Clustering treebanks...[/yellow]")
        bootstrapper._cluster_treebanks(embeddings_by_tb)

        # Merge with existing clusters if we loaded state
        if existing_treebank_clusters:
            reclustered_treebanks = (
                set(treebank_filter)
                if treebank_filter
                else {tb_code for tb_code, _split in embeddings_by_tb.keys()}
            )

            # Preserve embeddings for treebanks that were not re-clustered so label
            # command can still compute cluster centroids from saved state.
            for key, emb_data in existing_embeddings_by_tb.items():
                if key[0] not in reclustered_treebanks and key not in embeddings_by_tb:
                    embeddings_by_tb[key] = emb_data

            existing_treebank_ids = {tb_code for tb_code, _split in existing_embeddings_by_tb.keys()}
            n_updated = len(reclustered_treebanks)
            n_kept = len(existing_treebank_ids - reclustered_treebanks)
            console.print(f"[blue]Updated {n_updated} treebank(s), kept {n_kept} unchanged[/blue]")

        console.print(f"\n[bold green]✓ Clustered {len(bootstrapper.treebank_clusters)} cluster entries[/bold green]")

        # Display cluster statistics
        _display_cluster_stats(bootstrapper.treebank_clusters)

        # Save cluster results using shared helper function
        _save_cluster_results(bootstrapper, embeddings_by_tb, cfg.output.genres_path)

        # Save embeddings for visualization (optional, can be large)
        from pathlib import Path as PathLib
        output_dir = PathLib(cfg.output.genres_path) / "clusters"
        console.print(f"[blue]Embeddings already cached at:[/blue] {cfg.embeddings.cache_dir if cfg.embeddings.cache_dir else 'not configured'}")

        console.print(f"\n[bold green]✓ Cluster results saved to {output_dir}[/bold green]")
        if config:
            console.print(f"[blue]To visualize clusters, run:[/blue] uv run ud-genre-bootstrap visualize-clusters --config {config}")
        else:
            console.print(f"[blue]To visualize clusters, run:[/blue] uv run ud-genre-bootstrap visualize-clusters --clusters {output_dir}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Clustering failed")
        raise typer.Exit(1)


@app.command()
def label(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    clusters: Optional[Path] = typer.Option(
        None,
        "--clusters",
        help="Path to directory containing cluster_state.pkl (from cluster command)",
        exists=True,
        dir_okay=True,
    ),
):
    """Label clusters using bootstrap algorithm.

    Applies the bootstrapping schedule to assign genre labels to clusters.

    If --clusters is provided, loads pre-computed cluster state instead of
    regenerating embeddings and re-clustering (much faster).
    """
    console.print("\n[bold cyan]Bootstrap Genre Labeling[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        bootstrapper = GenreBootstrapper(cfg)

        # Check if we should load pre-computed cluster state
        cluster_state_path = None
        if clusters:
            cluster_state_path = clusters / "cluster_state.pkl"
            if not cluster_state_path.exists():
                console.print(f"[yellow]Warning: cluster_state.pkl not found in {clusters}, will regenerate clusters[/yellow]")
                cluster_state_path = None

        if cluster_state_path:
            # Load pre-computed cluster state
            console.print(f"\n[blue]Loading pre-computed cluster state from {cluster_state_path}...[/blue]")
            embeddings_by_tb = bootstrapper.load_cluster_state(cluster_state_path)

            console.print("\n[yellow]Computing cluster embeddings...[/yellow]")
            bootstrapper._compute_cluster_embeddings(embeddings_by_tb)
        else:
            # Run through clustering from scratch
            console.print("\n[yellow]Generating embeddings for all treebanks...[/yellow]")
            embeddings_by_tb = bootstrapper._generate_embeddings()

            console.print("\n[yellow]Clustering treebanks...[/yellow]")
            bootstrapper._cluster_treebanks(embeddings_by_tb)

            console.print("\n[yellow]Computing cluster embeddings...[/yellow]")
            bootstrapper._compute_cluster_embeddings(embeddings_by_tb)

        # Create schedule and label
        console.print("\n[yellow]Creating bootstrap schedule...[/yellow]")
        schedule = bootstrapper._create_schedule()

        console.print(f"[blue]Schedule length:[/blue] {len(schedule)} environments")

        # Display schedule summary
        _display_schedule_summary(schedule)

        console.print("\n[yellow]Running bootstrap labeling...[/yellow]")
        bootstrapper.execute_bootstrap_labeling(schedule=schedule)

        # Export results
        console.print("\n[yellow]Exporting genre assignments...[/yellow]")
        results = bootstrapper._export_results()

        console.print(f"\n[bold green]✓ Labeling complete![/bold green]")
        console.print(f"[green]✓ Labeled {results['labeled_sentences']} sentences[/green]")
        console.print(f"[blue]Genre assignments saved to:[/blue] {cfg.output.genres_path}/all_genres.parquet")

        if cfg.output.push_to_hub:
            console.print("\n[yellow]Uploading release artifacts to Hugging Face Hub...[/yellow]")
            bootstrapper.push_to_hub(
                resolve_release_hf_repo(cfg),
                resolve_release_hf_revisions(cfg),
            )

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Labeling failed")
        raise typer.Exit(1)


@app.command("upload")
def upload_release(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help=(
            "Existing release directory containing all_genres.parquet. "
            "Defaults to output.genres_path from config."
        ),
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        help="Hugging Face dataset repo override. Defaults to output.genres_hf_repo.",
    ),
    revision: Optional[List[str]] = typer.Option(
        None,
        "--revision",
        help="Hugging Face dataset revision/branch override. Can be repeated. Defaults to release.hf_revisions.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the upload plan without touching Hugging Face.",
    ),
    include_main: bool = typer.Option(
        False,
        "--include-main",
        help=(
            "Also upload to the Hugging Face `main` branch, which is the default "
            "revision shown in the web UI and used when no revision is specified."
        ),
    ),
):
    """Upload an existing release directory to Hugging Face Hub."""
    console.print("\n[bold cyan]UD Genre Release Upload[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        if output:
            cfg.output.genres_path = str(output)
            console.print(f"[blue]Release directory:[/blue] {output}")

        output_path = Path(cfg.output.genres_path)
        labels_path = output_path / "all_genres.parquet"
        if not labels_path.exists():
            raise ValueError(
                f"Release file not found: {labels_path}. "
                "Run `label` first or pass `--output` to an existing release directory."
            )

        repo_id = repo or resolve_release_hf_repo(cfg)
        if not repo_id:
            raise ValueError(
                "No Hugging Face dataset repo configured. "
                "Set `release.hf_repo`/`output.genres_hf_repo` in config or pass `--repo`."
            )

        target_revisions = resolve_release_hf_revisions(cfg, revision)
        if include_main and "main" not in target_revisions:
            target_revisions.append("main")
        if not target_revisions:
            raise ValueError(
                "No Hugging Face revisions configured. "
                "Set `release.hf_revisions` in config or pass `--revision`."
            )

        cfg.release.hf_repo = repo_id
        cfg.release.hf_revisions = target_revisions
        cfg.output.genres_hf_repo = repo_id
        cfg.output.genres_revision = target_revisions[0]
        release_identity = resolve_release_identity(cfg)

        console.print(f"[blue]Repo:[/blue] {repo_id}")
        console.print(f"[blue]Revisions:[/blue] {', '.join(target_revisions)}")
        console.print(f"[blue]Artifact ID:[/blue] {release_identity['artifact_id']}")
        console.print(f"[blue]Git branch:[/blue] {release_identity.get('git_branch') or 'n/a'}")
        console.print(f"[blue]Git tag:[/blue] {release_identity.get('git_tag') or 'n/a'}")

        if dry_run:
            prepare_release_directory(cfg, output_path)
            upload_files = list_release_upload_files(output_path)
            console.print("\n[bold yellow]Dry run: no Hugging Face calls will be made.[/bold yellow]")
            console.print("[blue]Files to upload:[/blue]")
            for upload_file in upload_files:
                console.print(f"  - {upload_file.relative_to(output_path)}")
            console.print("\n[bold green]✓ Dry run complete[/bold green]")
            return

        if not cfg.output.hf_token:
            raise ValueError(
                "No Hugging Face token configured. "
                "Set `output.hf_token` in config or via environment-backed config expansion."
            )

        console.print(
            "[yellow]Reusing existing all_genres.parquet and regenerating release artifacts before upload...[/yellow]"
        )

        upload_release_directory_to_hub(cfg, output_path, repo_id, target_revisions)

        console.print("\n[bold green]✓ Upload complete![/bold green]")
        console.print(f"[green]✓ Uploaded release artifacts from {output_path}[/green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Upload failed")
        raise typer.Exit(1)


@app.command("publish")
def publish_release(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Existing release directory containing all_genres.parquet. Defaults to output.genres_path from config.",
    ),
    hf_repo_dir: Path = typer.Option(
        ...,
        "--hf-repo-dir",
        help="Local Git checkout of the Hugging Face dataset repository.",
        exists=True,
        file_okay=False,
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        help="Hugging Face dataset repo override. Defaults to release.hf_repo.",
    ),
    branch: Optional[List[str]] = typer.Option(
        None,
        "--branch",
        help=(
            "Moving Hugging Face dataset branch. Can be repeated. "
            "Defaults to release.hf_branches."
        ),
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the Git publish plan without touching the HF checkout.",
    ),
    include_main: bool = typer.Option(
        False,
        "--include-main",
        help=(
            "Also move the configured HF default branch, normally `main`, to the "
            "published artifact commit."
        ),
    ),
    push: bool = typer.Option(
        False,
        "--push",
        help="Push the updated HF branches and artifact tag to the checkout's origin.",
    ),
):
    """Publish an existing release directory through a local HF dataset Git checkout."""
    console.print("\n[bold cyan]UD Genre Git Publish[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        if output:
            cfg.output.genres_path = str(output)
            console.print(f"[blue]Release directory:[/blue] {output}")

        output_path = Path(cfg.output.genres_path)
        labels_path = output_path / "all_genres.parquet"
        if not labels_path.exists():
            raise ValueError(
                f"Release file not found: {labels_path}. "
                "Run `label` first or pass `--output` to an existing release directory."
            )

        repo_id = repo or resolve_release_hf_repo(cfg)
        if repo_id:
            cfg.release.hf_repo = repo_id
            cfg.output.genres_hf_repo = repo_id

        target_branches = resolve_release_hf_branches(cfg, branch)
        if not target_branches:
            raise ValueError(
                "No Hugging Face branches configured. "
                "Set `release.hf_branches` in config or pass `--branch`."
            )
        cfg.release.hf_branches = target_branches
        cfg.output.genres_revision = target_branches[0]

        release_identity = resolve_release_identity(cfg)

        console.print(f"[blue]Repo:[/blue] {release_identity.get('hf_repo') or 'n/a'}")
        console.print(f"[blue]HF checkout:[/blue] {hf_repo_dir}")
        console.print(f"[blue]HF branches:[/blue] {', '.join(target_branches)}")
        console.print(f"[blue]HF tag:[/blue] {release_identity.get('hf_tag')}")
        console.print(f"[blue]Artifact ID:[/blue] {release_identity['artifact_id']}")
        console.print(f"[blue]Source tag:[/blue] {release_identity.get('source_tag')}")

        result = publish_release_directory_to_hf_git(
            cfg,
            output_path,
            hf_repo_dir,
            include_main=include_main,
            push=push,
            dry_run=dry_run,
        )

        if dry_run:
            console.print("\n[bold yellow]Dry run: no HF Git checkout changes will be made.[/bold yellow]")
            console.print("[blue]Files to publish:[/blue]")
            for publish_file in result["files"]:
                console.print(f"  - {publish_file}")
            console.print(
                f"[blue]Target branches:[/blue] {', '.join(result['target_branches'])}"
            )
            console.print("\n[bold green]✓ Dry run complete[/bold green]")
            return

        console.print("\n[bold green]✓ Publish complete![/bold green]")
        console.print(f"[green]✓ HF commit: {result.get('hf_commit')}[/green]")
        console.print(f"[green]✓ HF tag: {result.get('hf_tag')}[/green]")
        if push:
            console.print("[green]✓ Pushed HF branches and artifact tag[/green]")
        else:
            console.print("[yellow]Run `git push` in the HF checkout, or rerun with `--push`.[/yellow]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Publish failed")
        raise typer.Exit(1)


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to evaluate (e.g., en_ewt or en_ewt,de_gsd). If not specified, evaluates all.",
    ),
    n_folds: Optional[int] = typer.Option(
        None,
        "--n-folds",
        "-k",
        help="Number of folds for cross-validation (overrides config)",
    ),
    group_by: Optional[str] = typer.Option(
        None,
        "--group-by",
        help="Variable to group by: 'treebank', 'language', or None (overrides config)",
    ),
    protocol: Optional[str] = typer.Option(
        None,
        "--protocol",
        help="Evaluation protocol: 'generalization' (default) or 'paper_parity' (original fixed-split GMM+L-style evaluation).",
    ),
    anchor_mode: Optional[str] = typer.Option(
        None,
        "--anchor-mode",
        help="Reference-anchor mode: 'strict' (fold-train anchors only) or 'parity' (plus broader single-genre anchors).",
    ),
    anchor_pool_policy: Optional[str] = typer.Option(
        None,
        "--anchor-pool-policy",
        help=(
            "Anchor source policy: 'auto', 'train_virtual', 'single_genre', or 'combined'. "
            "'auto' maps strict->train_virtual and parity->combined."
        ),
    ),
    eval_set: Optional[List[str]] = typer.Option(
        None,
        "--set",
        "-s",
        help="Named evaluation set from config key evaluation.treebank_sets. Repeat for multiple sets.",
    ),
    treebank_set: Optional[List[str]] = typer.Option(
        None,
        "--treebank-set",
        help="Inline named set definition 'name=tb1,tb2,...'. Repeat for multiple sets.",
    ),
    progressive: bool = typer.Option(
        False,
        "--progressive",
        help="Run cumulative evaluations on increasingly larger treebank pools.",
    ),
    progressive_step: int = typer.Option(
        1,
        "--progressive-step",
        min=1,
        help="Number of treebanks to add per progressive evaluation stage.",
    ),
    sentence_split_map: Optional[Path] = typer.Option(
        None,
        "--sentence-split-map",
        help=(
            "Optional sentence split map (.parquet/.csv/.tsv) with columns "
            "treebank, split, sent_id and optional partition."
        ),
        exists=True,
        dir_okay=False,
    ),
    split_partition: Optional[List[str]] = typer.Option(
        None,
        "--split-partition",
        help=(
            "Optional partition(s) to select from --sentence-split-map "
            "(e.g., train/dev/test). Repeat flag for multiple partitions."
        ),
    ),
    fixed_partition: bool = typer.Option(
        False,
        "--fixed-partition",
        help=(
            "Use one predefined anchor/test partition split from --sentence-split-map "
            "instead of k-fold cross-validation."
        ),
    ),
    anchor_partition: Optional[List[str]] = typer.Option(
        None,
        "--anchor-partition",
        help=(
            "Partition(s) used as anchors in --fixed-partition mode "
            "(default: train + dev). Repeat flag for multiple partitions."
        ),
    ),
    test_partition: str = typer.Option(
        "test",
        "--test-partition",
        help="Partition used as held-out test set in --fixed-partition mode.",
    ),
):
    """Evaluate clustering + labeling on multi-genre treebanks.

    Tests the actual problem the framework solves: clustering mixed sentences
    and assigning genres. Reports sentence-level micro-F1-equivalent accuracy against ground truth
    metadata.

    For each cross-validation fold:
    - Training: Uses reference treebanks to build genre embeddings
    - Testing: Clusters multi-genre treebanks and labels clusters
    - Evaluation: Compares predicted genres vs. true genres for each sentence
    """
    console.print("\n[bold cyan]Clustering Evaluation (Sentence-Level)[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Get evaluation config values (CLI overrides config)
        eval_cfg = cfg.evaluation.metadata_validation
        protocol_val = normalize_evaluation_protocol(
            protocol,
            getattr(eval_cfg, "protocol", "generalization"),
        )
        fixed_partition_mode = fixed_partition or protocol_val == "paper_parity"
        n_folds_val = n_folds if n_folds is not None else eval_cfg.k
        group_by_val = group_by if group_by is not None else eval_cfg.group_by
        anchor_mode_val = normalize_anchor_mode(anchor_mode, eval_cfg.anchor_mode)
        anchor_pool_policy_val = normalize_anchor_pool_policy(
            anchor_pool_policy,
            getattr(eval_cfg, "anchor_pool_policy", "auto"),
            anchor_mode=anchor_mode_val,
        )

        if protocol_val == "paper_parity":
            if sentence_split_map is None:
                raise ValueError(
                    "Paper-parity evaluation requires --sentence-split-map."
                )
            fixed_partition_mode = True
            group_by_val = None
            anchor_mode_val = "parity"
            anchor_pool_policy_val = "single_genre"
            if group_by is not None:
                console.print(
                    f"[yellow]Ignoring --group-by={group_by} in paper-parity mode "
                    "(fixed test-partition protocol)[/yellow]"
                )
            if anchor_pool_policy is not None:
                raise ValueError(
                    "Paper-parity protocol requires single-genre anchors only. "
                    "Do not override --anchor-pool-policy."
                )
            if anchor_mode is not None:
                raise ValueError(
                    "Paper-parity protocol requires parity anchor mode. "
                    "Do not override --anchor-mode."
                )

        uses_single_genre_anchors = anchor_pool_policy_val in {"single_genre", "combined"}
        uses_train_virtual_anchors = anchor_pool_policy_val in {"train_virtual", "combined"}
        if fixed_partition_mode and sentence_split_map is None:
            raise ValueError("Fixed-partition mode requires --sentence-split-map.")
        if fixed_partition_mode and split_partition:
            raise ValueError(
                "Use --anchor-partition/--test-partition with --fixed-partition, "
                "not --split-partition."
            )
        if fixed_partition_mode:
            if n_folds is not None and n_folds != 1:
                console.print(
                    f"[yellow]Ignoring --n-folds={n_folds} in fixed-partition mode "
                    "(single holdout run)[/yellow]"
                )
            n_folds_val = 1

        split_map_filter = None
        split_map_skipped_splits = 0
        split_map_skipped_sentences = 0
        if sentence_split_map is not None and not fixed_partition_mode:
            from ud_genre_bootstrap.utils.sentence_split_map import (
                filter_embeddings_by_sentence_split_map,
                load_sentence_split_map,
            )

            split_map_filter = load_sentence_split_map(
                sentence_split_map,
                partitions=split_partition,
            )
            selected_partition_label = (
                ", ".join(split_map_filter.selected_partitions)
                if split_map_filter.selected_partitions
                else "all"
            )
            console.print(
                f"[blue]Sentence split map:[/blue] {sentence_split_map} "
                f"(partition(s): {selected_partition_label}, "
                f"rows: {split_map_filter.selected_rows}, "
                f"split keys: {len(split_map_filter.split_to_sent_ids)})"
            )
            if split_map_filter.dropped_rows > 0:
                console.print(
                    f"[yellow]Split map dropped {split_map_filter.dropped_rows} "
                    f"row(s) with missing/empty key fields[/yellow]"
                )

        # Resolve evaluation set definitions (config + inline CLI)
        config_eval_sets = resolve_config_treebank_sets(cfg, eval_set)
        inline_eval_sets = parse_inline_treebank_sets(treebank_set)

        overlapping_set_names = set(config_eval_sets).intersection(inline_eval_sets)
        if overlapping_set_names:
            raise ValueError(
                "Duplicate evaluation set name(s) across --set and --treebank-set: "
                + ", ".join(sorted(overlapping_set_names))
            )

        explicit_eval_sets = {**config_eval_sets, **inline_eval_sets}
        if treebank and explicit_eval_sets:
            raise ValueError("Use either --treebank or explicit evaluation sets (--set / --treebank-set), not both.")
        if progressive and explicit_eval_sets:
            raise ValueError("Use either --progressive or explicit evaluation sets, not both together.")
        if fixed_partition_mode and (explicit_eval_sets or progressive):
            raise ValueError(
                "Fixed-partition mode does not support --set/--treebank-set/--progressive."
            )

        # Parse base treebank filter
        treebank_filter = None
        if explicit_eval_sets:
            treebank_filter = sorted({
                tb_code
                for set_treebanks in explicit_eval_sets.values()
                for tb_code in set_treebanks
            })
            console.print(
                f"[blue]Evaluating {len(explicit_eval_sets)} named set(s) over "
                f"{len(treebank_filter)} explicit treebank(s)[/blue]"
            )
        elif treebank:
            treebank_filter = parse_treebank_csv(treebank)
            if len(treebank_filter) == 1:
                console.print(f"[blue]Evaluating:[/blue] {treebank_filter[0]}")
            else:
                console.print(f"[blue]Evaluating {len(treebank_filter)} treebanks:[/blue] {', '.join(treebank_filter)}")
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            if len(treebank_filter) == 1:
                console.print(f"[blue]Evaluating (from config):[/blue] {treebank_filter[0]}")
            else:
                console.print(f"[blue]Evaluating {len(treebank_filter)} treebanks (from config):[/blue] {', '.join(treebank_filter)}")
        else:
            console.print("[blue]Evaluating all treebanks[/blue]")

        console.print(f"[blue]Protocol:[/blue] {protocol_val}")
        if fixed_partition_mode:
            console.print(
                "[blue]Evaluation mode:[/blue] fixed-partition holdout "
                "(no cross-validation refolding)"
            )
        else:
            console.print(
                f"[blue]CV settings:[/blue] {n_folds_val}-fold, group_by={group_by_val}"
            )
        console.print(f"[blue]Anchor mode:[/blue] {anchor_mode_val}")
        console.print(f"[blue]Anchor pool policy:[/blue] {anchor_pool_policy_val}")
        console.print(f"[blue]Min confidence:[/blue] {cfg.bootstrapping.min_confidence}")
        console.print(f"[blue]Min margin:[/blue] {cfg.bootstrapping.min_margin}")
        console.print(f"[blue]Reference weighting:[/blue] {cfg.bootstrapping.reference_weighting}")
        if progressive:
            console.print(f"[blue]Progressive mode:[/blue] enabled (step={progressive_step})")

        # Initialize clustering evaluator
        from ud_genre_bootstrap.evaluation.validator import ClusteringEvaluator

        evaluator = ClusteringEvaluator(
            n_folds=n_folds_val,
            group_by=group_by_val,
            min_confidence=cfg.bootstrapping.min_confidence,
            min_margin=cfg.bootstrapping.min_margin,
            max_iterations=cfg.bootstrapping.max_iterations,
            anchor_mode=anchor_mode_val,
            anchor_pool_policy=anchor_pool_policy_val,
            reference_weighting=cfg.bootstrapping.reference_weighting,
            protocol=protocol_val,
        )

        # Load treebank metadata and genre mapper
        console.print("\n[yellow]Loading treebank metadata...[/yellow]")
        bootstrapper = GenreBootstrapper(cfg)

        # Initialize genre mapper for coverage checking
        from pathlib import Path as PathLib
        from ud_genre_bootstrap.utils.genre_mapping import GenreMapper

        mapping_path = None
        patterns_path = None
        if cfg.genre_extraction.mapping_path:
            mapping_path = PathLib(cfg.genre_extraction.mapping_path)
        if cfg.genre_extraction.patterns_path:
            if isinstance(cfg.genre_extraction.patterns_path, list):
                patterns_path = [PathLib(p) for p in cfg.genre_extraction.patterns_path]
            else:
                patterns_path = PathLib(cfg.genre_extraction.patterns_path)

        genre_mapper = GenreMapper(
            genre_mapping_path=mapping_path,
            metadata_patterns_path=patterns_path,
            canonical_genres=cfg.genre_extraction.canonical_genres,
            data_loader=bootstrapper.data_loader,
        )

        # Get all treebank metadata
        all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()

        # Apply exclusions from config
        if cfg.exclude_treebanks:
            all_treebank_data = [
                tb for tb in all_treebank_data if tb['id'] not in cfg.exclude_treebanks
            ]

        # Determine which treebanks are evaluation targets (multi-genre candidates).
        if treebank_filter:
            evaluation_treebank_ids = {tb_id for tb_id in treebank_filter}
        else:
            evaluation_treebank_ids = {tb["id"] for tb in all_treebank_data}

        paper_treebank_genre_map = None
        paper_scoring_treebanks = None
        if protocol_val == "paper_parity":
            paper_treebank_genre_map = resolve_paper_evaluation_treebank_genres(
                bootstrapper.data_loader
            )
            paper_scope_treebank_ids = set(paper_treebank_genre_map)
            if not paper_scope_treebank_ids:
                raise ValueError(
                    "Paper-parity protocol could not resolve any treebanks from "
                    f"{PAPER_TREEBANK_MAPPING_PATH}."
                )
            excluded_nonpaper_treebanks = sorted(
                evaluation_treebank_ids - paper_scope_treebank_ids
            )
            evaluation_treebank_ids &= paper_scope_treebank_ids
            if excluded_nonpaper_treebanks:
                console.print(
                    "[blue]Paper sentence-evaluation scope:[/blue] excluding "
                    f"{len(excluded_nonpaper_treebanks)} non-paper treebank(s): "
                    f"{', '.join(excluded_nonpaper_treebanks[:8])}"
                    f"{'...' if len(excluded_nonpaper_treebanks) > 8 else ''}"
                )
            console.print(
                "[blue]Paper sentence-evaluation scope:[/blue] "
                f"{len(evaluation_treebank_ids)} mapped treebank(s) from "
                f"{PAPER_TREEBANK_MAPPING_PATH.name}"
            )
            if not evaluation_treebank_ids:
                raise ValueError(
                    "Paper-parity protocol has no evaluation targets after scope "
                    "restriction."
                )

        if fixed_partition_mode:
            from ud_genre_bootstrap.utils.sentence_split_map import (
                filter_embeddings_by_sentence_split_map,
                load_sentence_split_map,
            )

            default_anchor_partitions = [test_partition] if protocol_val == "paper_parity" else ["train", "dev"]
            anchor_partitions_val = [
                p.strip() for p in (anchor_partition or default_anchor_partitions) if p and p.strip()
            ]
            if not anchor_partitions_val:
                raise ValueError(
                    "Fixed-partition mode requires at least one --anchor-partition."
                )
            test_partition_val = (test_partition or "").strip()
            if not test_partition_val:
                raise ValueError(
                    "Fixed-partition mode requires a non-empty --test-partition."
                )

            if protocol_val == "paper_parity":
                if anchor_partitions_val != [test_partition_val]:
                    raise ValueError(
                        "Paper-parity protocol requires --anchor-partition to match "
                        f"--test-partition exactly ({test_partition_val})."
                    )
                console.print(
                    f"[blue]Paper-parity anchor source:[/blue] whole-treebank single-genre anchors "
                    f"from partition '{test_partition_val}'"
                )
            elif test_partition_val in anchor_partitions_val:
                console.print(
                    f"[yellow]Test partition '{test_partition_val}' also appears in "
                    "--anchor-partition; this can leak evaluation data.[/yellow]"
                )

            anchor_split_map = load_sentence_split_map(
                sentence_split_map,
                partitions=anchor_partitions_val,
            )
            test_split_map = load_sentence_split_map(
                sentence_split_map,
                partitions=[test_partition_val],
            )
            combined_split_map = load_sentence_split_map(
                sentence_split_map,
                partitions=anchor_partitions_val + [test_partition_val],
            )

            overlap_sentences = 0
            for split_key, anchor_sent_ids in anchor_split_map.split_to_sent_ids.items():
                test_sent_ids = test_split_map.split_to_sent_ids.get(split_key)
                if test_sent_ids is None:
                    continue
                overlap_sentences += len(anchor_sent_ids.intersection(test_sent_ids))
            if overlap_sentences > 0:
                if protocol_val == "paper_parity":
                    console.print(
                        f"[blue]Paper-parity partition overlap:[/blue] {overlap_sentences} sentence(s) "
                        "shared between anchor and test views by design."
                    )
                else:
                    console.print(
                        f"[yellow]Fixed-partition selection has {overlap_sentences} sentence(s) "
                        "present in both anchor and test partitions.[/yellow]"
                    )

            console.print(
                f"[blue]Fixed-partition mode:[/blue] anchors="
                f"{', '.join(anchor_partitions_val)}; test={test_partition_val}"
            )
            console.print(
                f"[blue]Anchor split keys:[/blue] "
                f"{len(anchor_split_map.split_to_sent_ids)}"
            )
            console.print(
                f"[blue]Test split keys:[/blue] "
                f"{len(test_split_map.split_to_sent_ids)}"
            )

            if uses_single_genre_anchors and anchor_mode_val == "parity":
                scan_treebank_data = all_treebank_data
            else:
                scan_treebank_data = [
                    tb for tb in all_treebank_data if tb["id"] in evaluation_treebank_ids
                ]
            console.print(
                f"[blue]Scanning {len(scan_treebank_data)} treebanks for fixed partitions...[/blue]"
            )

            test_multi_genre_treebanks = []
            parity_single_genre_treebanks = []
            sentence_metadata = {}
            train_treebank_keys = set()
            stats = {
                "checked": 0,
                "anchor_splits": 0,
                "test_splits": 0,
                "single_genre_test": 0,
                "multi_genre_test": 0,
                "load_errors": 0,
                "no_metadata_test": 0,
            }
            seen_anchor_split_keys = set()
            seen_test_split_keys = set()
            load_error_split_keys = set()
            no_metadata_test_split_keys = set()
            paper_test_split_keys_by_treebank: Dict[str, set[Tuple[str, str]]] = {}
            paper_anchor_split_keys_by_treebank: Dict[str, set[Tuple[str, str]]] = {}
            paper_test_metadata_counts_by_treebank: Dict[str, Dict[str, int]] = {}
            paper_test_sentence_counts_by_treebank: Dict[str, int] = {}
            paper_anchor_sentence_counts_by_treebank: Dict[str, int] = {}
            paper_treebank_languages: Dict[str, str] = {}

            with console.status(
                "[blue]Scanning sentence metadata for fixed partitions...[/blue]"
            ) as status:
                for tb in scan_treebank_data:
                    tb_code = tb["id"]
                    language = tb.get("language", tb_code.split("_")[0])
                    is_evaluation_target = tb_code in evaluation_treebank_ids
                    available_splits = bootstrapper.data_loader.get_available_splits(tb_code)
                    if not available_splits:
                        continue

                    for split_name in available_splits:
                        in_anchor_split = anchor_split_map.includes_split(tb_code, split_name)
                        in_test_split = test_split_map.includes_split(tb_code, split_name)
                        if not in_anchor_split and not in_test_split:
                            continue

                        if is_evaluation_target:
                            stats["checked"] += 1
                        status.update(
                            f"[blue]Scanning {tb_code}:{split_name} "
                            f"({stats['checked']} split(s) checked)...[/blue]"
                        )

                        split_key = (tb_code, split_name)
                        if in_anchor_split:
                            seen_anchor_split_keys.add(split_key)
                            paper_anchor_split_keys_by_treebank.setdefault(tb_code, set()).add(split_key)
                        if in_test_split:
                            seen_test_split_keys.add(split_key)
                            paper_test_split_keys_by_treebank.setdefault(tb_code, set()).add(split_key)
                        if in_anchor_split or in_test_split:
                            paper_treebank_languages[tb_code] = language

                        anchor_genre_counts = {}
                        test_genre_counts = {}
                        anchor_sentence_count = 0
                        test_sentence_count = 0

                        try:
                            sentence_iter = bootstrapper.data_loader.iter_treebank_sentences(
                                tb_code,
                                split_name,
                                metadata_only=True,
                            )
                            for idx, sentence in enumerate(sentence_iter):
                                sent_id = sentence.get(
                                    "sent_id",
                                    f"{tb_code}_{split_name}_{idx}",
                                )
                                in_anchor_sentence = (
                                    in_anchor_split
                                    and anchor_split_map.includes_sentence(
                                        tb_code, split_name, sent_id
                                    )
                                )
                                in_test_sentence = (
                                    in_test_split
                                    and test_split_map.includes_sentence(
                                        tb_code, split_name, sent_id
                                    )
                                    and (protocol_val == "paper_parity" or is_evaluation_target)
                                )
                                if not in_anchor_sentence and not in_test_sentence:
                                    continue

                                genres = genre_mapper.extract_genres_from_metadata(
                                    sentence, tb_code
                                )
                                if not genres:
                                    continue

                                primary_genre = genres[0]
                                sentence_metadata[(tb_code, split_name, sent_id)] = (
                                    primary_genre
                                )

                                if in_anchor_sentence:
                                    anchor_genre_counts[primary_genre] = (
                                        anchor_genre_counts.get(primary_genre, 0) + 1
                                    )
                                    anchor_sentence_count += 1

                                if in_test_sentence:
                                    test_genre_counts[primary_genre] = (
                                        test_genre_counts.get(primary_genre, 0) + 1
                                    )
                                    test_sentence_count += 1
                                    tb_counts = paper_test_metadata_counts_by_treebank.setdefault(
                                        tb_code, {}
                                    )
                                    tb_counts[primary_genre] = tb_counts.get(primary_genre, 0) + 1
                                    paper_test_sentence_counts_by_treebank[tb_code] = (
                                        paper_test_sentence_counts_by_treebank.get(tb_code, 0) + 1
                                    )
                        except Exception as e:
                            logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                            load_error_split_keys.add(split_key)
                            if is_evaluation_target:
                                stats["load_errors"] += 1
                            continue

                        if anchor_sentence_count > 0:
                            paper_anchor_sentence_counts_by_treebank[tb_code] = (
                                paper_anchor_sentence_counts_by_treebank.get(tb_code, 0)
                                + anchor_sentence_count
                            )
                            stats["anchor_splits"] += 1
                            if uses_train_virtual_anchors:
                                train_treebank_keys.add((tb_code, split_name))

                            anchor_unique_genres = list(anchor_genre_counts.keys())
                            if (
                                uses_single_genre_anchors
                                and len(anchor_unique_genres) == 1
                            ):
                                parity_single_genre_treebanks.append(
                                    {
                                        "treebank": tb_code,
                                        "split": split_name,
                                        "genres": anchor_unique_genres,
                                        "language": language,
                                        "sentence_count": anchor_sentence_count,
                                        "genre_counts": anchor_genre_counts,
                                    }
                                )

                        if test_sentence_count > 0:
                            stats["test_splits"] += 1
                            unique_test_genres = list(test_genre_counts.keys())
                            if len(unique_test_genres) >= 2:
                                stats["multi_genre_test"] += 1
                                test_multi_genre_treebanks.append(
                                    {
                                        "treebank": tb_code,
                                        "split": split_name,
                                        "genres": unique_test_genres,
                                        "language": language,
                                        "sentence_count": test_sentence_count,
                                        "genre_counts": test_genre_counts,
                                    }
                                )
                            elif len(unique_test_genres) == 1:
                                stats["single_genre_test"] += 1
                            else:
                                stats["no_metadata_test"] += 1
                                no_metadata_test_split_keys.add(split_key)

            if protocol_val == "paper_parity":
                paper_full_test_treebanks = []
                paper_scoring_treebanks = []
                for tb_code, split_keys in sorted(paper_test_split_keys_by_treebank.items()):
                    treebank_genres = sorted(
                        bootstrapper.data_loader.get_treebank_genres(tb_code) or []
                    )
                    genre_counts = dict(
                        sorted(paper_test_metadata_counts_by_treebank.get(tb_code, {}).items())
                    )
                    if not genre_counts:
                        continue

                    descriptor = {
                        "treebank": tb_code,
                        "split_keys": sorted(split_keys),
                        "genres": treebank_genres,
                        "observed_genres": sorted(genre_counts),
                        "language": paper_treebank_languages.get(tb_code, tb_code.split("_", 1)[0]),
                        "sentence_count": paper_test_sentence_counts_by_treebank.get(tb_code, 0),
                        "genre_counts": genre_counts,
                    }

                    if len(treebank_genres) >= 2:
                        paper_full_test_treebanks.append(descriptor)

                    if tb_code in evaluation_treebank_ids:
                        expected_genres = sorted((paper_treebank_genre_map or {}).get(tb_code, []))
                        if len(expected_genres) < 2:
                            continue
                        scoring_descriptor = dict(descriptor)
                        scoring_descriptor["genres"] = expected_genres
                        paper_scoring_treebanks.append(scoring_descriptor)

                paper_single_genre_anchor_treebanks = []
                for tb_code, split_keys in sorted(paper_anchor_split_keys_by_treebank.items()):
                    treebank_genres = list(bootstrapper.data_loader.get_treebank_genres(tb_code) or [])
                    if len(treebank_genres) != 1:
                        continue
                    anchor_genre = treebank_genres[0]
                    paper_single_genre_anchor_treebanks.append(
                        {
                            "treebank": tb_code,
                            "split_keys": sorted(split_keys),
                            "genres": [anchor_genre],
                            "language": paper_treebank_languages.get(tb_code, tb_code.split("_", 1)[0]),
                            "sentence_count": paper_anchor_sentence_counts_by_treebank.get(tb_code, 0),
                            "genre_counts": {anchor_genre: paper_anchor_sentence_counts_by_treebank.get(tb_code, 0)},
                        }
                    )

                test_multi_genre_treebanks = paper_full_test_treebanks
                parity_single_genre_treebanks = paper_single_genre_anchor_treebanks

            if len(test_multi_genre_treebanks) == 0:
                console.print(
                    "\n[bold yellow]⚠ No multi-genre test splits found for fixed-partition evaluation[/bold yellow]"
                )
                console.print("\n[bold]Summary:[/bold]")
                console.print(f"  • Checked: {stats['checked']} split(s)")
                console.print(f"  • Anchor splits: {stats['anchor_splits']}")
                console.print(f"  • Test splits: {stats['test_splits']}")
                console.print(f"  • Multi-genre test: {stats['multi_genre_test']}")
                console.print(f"  • Single-genre test: {stats['single_genre_test']}")
                if stats["no_metadata_test"] > 0:
                    console.print(
                        f"  • No metadata in test partition: {stats['no_metadata_test']}"
                    )
                if stats["load_errors"] > 0:
                    console.print(f"  • Load errors: {stats['load_errors']}")
                console.print("\n[blue]Evaluation skipped - no suitable test data[/blue]")
                raise typer.Exit(0)

            requested_anchor_split_keys = sorted(anchor_split_map.split_keys)
            requested_test_split_keys = sorted(test_split_map.split_keys)
            source_missing_anchor_split_keys = sorted(anchor_split_map.split_keys - seen_anchor_split_keys)
            source_missing_test_split_keys = sorted(test_split_map.split_keys - seen_test_split_keys)

            train_treebank_keys = sorted(train_treebank_keys)
            initial_train_treebank_keys = set(train_treebank_keys)
            initial_test_split_keys = collect_treebank_descriptor_split_keys(
                test_multi_genre_treebanks
            )
            initial_scoring_split_keys = collect_treebank_descriptor_split_keys(
                paper_scoring_treebanks or []
            )
            initial_single_anchor_split_keys = collect_treebank_descriptor_split_keys(
                parity_single_genre_treebanks
            )
            if uses_train_virtual_anchors and len(train_treebank_keys) == 0:
                raise ValueError(
                    "Fixed-partition evaluation requires at least one anchor split "
                    "with sentence-level metadata."
                )
            if (
                uses_single_genre_anchors
                and not uses_train_virtual_anchors
                and len(parity_single_genre_treebanks) == 0
            ):
                if protocol_val == "paper_parity":
                    raise ValueError(
                        "Paper-parity evaluation requires at least one whole-treebank "
                        "single-genre anchor."
                    )
                raise ValueError(
                    "Fixed-partition evaluation with anchor_pool_policy=single_genre "
                    "requires at least one single-genre anchor split."
                )

            summary_parts = []
            if uses_train_virtual_anchors:
                summary_parts.append(f"{len(train_treebank_keys)} virtual-anchor split(s)")
            if uses_single_genre_anchors:
                if protocol_val == "paper_parity":
                    summary_parts.append(
                        f"{len(parity_single_genre_treebanks)} single-genre anchor treebank(s) "
                        f"across {len(initial_single_anchor_split_keys)} split(s)"
                    )
                else:
                    summary_parts.append(
                        f"{len(parity_single_genre_treebanks)} single-genre anchor candidate split(s)"
                    )
            if protocol_val == "paper_parity":
                summary_parts.append(
                    f"{len(test_multi_genre_treebanks)} full-test clustering treebank(s) "
                    f"across {len(initial_test_split_keys)} split(s)"
                )
                summary_parts.append(
                    f"{len(paper_scoring_treebanks or [])} paper-scope scored treebank(s) "
                    f"across {len(initial_scoring_split_keys)} split(s)"
                )
            else:
                summary_parts.append(f"{len(test_multi_genre_treebanks)} multi-genre test split(s)")
            console.print(
                f"[blue]Fixed holdout summary:[/blue] {', '.join(summary_parts)}"
            )

            treebank_ids_to_embed = sorted(
                {
                    tb_code
                    for tb_code, _split_name in train_treebank_keys
                }.union({tb["treebank"] for tb in test_multi_genre_treebanks})
            )
            if uses_single_genre_anchors and parity_single_genre_treebanks:
                treebank_ids_to_embed = sorted(
                    set(treebank_ids_to_embed).union(
                        {tb["treebank"] for tb in parity_single_genre_treebanks}
                    )
                )

            console.print("\n[yellow]Generating/loading embeddings...[/yellow]")
            embeddings_by_tb = bootstrapper._generate_embeddings(
                treebank_filter=treebank_ids_to_embed
            )
            embeddings_by_tb, embedding_filter_stats = (
                filter_embeddings_by_sentence_split_map(
                    embeddings_by_tb,
                    combined_split_map,
                )
            )
            console.print(
                f"[blue]Split-map embedding filter:[/blue] kept "
                f"{embedding_filter_stats.kept_sentences} sentence(s) across "
                f"{embedding_filter_stats.kept_splits} split(s); dropped "
                f"{embedding_filter_stats.dropped_sentences} sentence(s) across "
                f"{embedding_filter_stats.dropped_splits} split(s)"
            )

            available_embedding_splits = set(embeddings_by_tb.keys())
            train_treebank_keys = [
                split_key
                for split_key in train_treebank_keys
                if split_key in available_embedding_splits
            ]
            test_multi_genre_treebanks = filter_treebank_descriptors_by_available_splits(
                test_multi_genre_treebanks,
                available_embedding_splits,
            )
            paper_scoring_treebanks = filter_treebank_descriptors_by_available_splits(
                paper_scoring_treebanks or [],
                available_embedding_splits,
            )
            parity_single_genre_treebanks = filter_treebank_descriptors_by_available_splits(
                parity_single_genre_treebanks,
                available_embedding_splits,
            )
            dropped_anchor_split_keys = sorted(
                initial_train_treebank_keys - set(train_treebank_keys)
            )
            dropped_test_split_keys = sorted(
                initial_test_split_keys
                - collect_treebank_descriptor_split_keys(test_multi_genre_treebanks)
            )
            dropped_scoring_split_keys = sorted(
                initial_scoring_split_keys
                - collect_treebank_descriptor_split_keys(paper_scoring_treebanks or [])
            )
            dropped_single_anchor_split_keys = sorted(
                initial_single_anchor_split_keys
                - collect_treebank_descriptor_split_keys(parity_single_genre_treebanks)
            )

            if uses_train_virtual_anchors and len(train_treebank_keys) == 0:
                raise ValueError(
                    "No anchor splits remained after split-map embedding filtering."
                )
            if (
                uses_single_genre_anchors
                and not uses_train_virtual_anchors
                and len(parity_single_genre_treebanks) == 0
            ):
                if protocol_val == "paper_parity":
                    raise ValueError(
                        "No whole-treebank single-genre anchors remained after "
                        "split-map embedding filtering."
                    )
                raise ValueError(
                    "No single-genre anchor splits remained after split-map embedding filtering."
                )
            if len(test_multi_genre_treebanks) == 0:
                if protocol_val == "paper_parity":
                    raise ValueError(
                        "No test-partition clustering treebanks remained after split-map embedding filtering."
                    )
                raise ValueError(
                    "No test splits remained after split-map embedding filtering."
                )
            if protocol_val == "paper_parity" and len(paper_scoring_treebanks or []) == 0:
                raise ValueError(
                    "No paper-scope scored treebanks remained after split-map embedding filtering."
                )

            console.print("\n[yellow]Running fixed-partition holdout evaluation...[/yellow]")
            set_results = evaluator.fixed_partition_validate(
                test_treebanks=test_multi_genre_treebanks,
                train_treebanks=train_treebank_keys,
                sentence_metadata=sentence_metadata,
                embeddings_by_tb=embeddings_by_tb,
                clusterer=bootstrapper.clusterer,
                single_genre_treebanks=(
                    parity_single_genre_treebanks
                    if uses_single_genre_anchors
                    else None
                ),
                scoring_treebanks=(
                    paper_scoring_treebanks
                    if protocol_val == "paper_parity"
                    else None
                ),
            )

            if (
                cfg.output.genres_path
                and "confusion_matrix" in set_results
                and "genre_labels" in set_results
            ):
                output_dir = Path(cfg.output.genres_path) / "evaluation"
                output_dir.mkdir(parents=True, exist_ok=True)
                result_label = f"fixed_{test_partition_val}"
                confusion_matrix_path = _save_clustering_confusion_matrix(
                    set_results,
                    output_dir=output_dir,
                    result_label=result_label,
                )
                if confusion_matrix_path is not None:
                    console.print(
                        f"[blue]Confusion matrix saved to:[/blue] "
                        f"{confusion_matrix_path}"
                    )

            set_results.update(
                build_fixed_partition_protocol_report(
                    protocol=protocol_val,
                    anchor_partitions=anchor_partitions_val,
                    test_partition=test_partition_val,
                    requested_anchor_split_keys=requested_anchor_split_keys,
                    requested_test_split_keys=requested_test_split_keys,
                    source_missing_anchor_split_keys=source_missing_anchor_split_keys,
                    source_missing_test_split_keys=source_missing_test_split_keys,
                    load_error_split_keys=sorted(load_error_split_keys),
                    no_metadata_test_split_keys=sorted(no_metadata_test_split_keys),
                    dropped_anchor_split_keys=dropped_anchor_split_keys,
                    dropped_test_split_keys=dropped_test_split_keys,
                    dropped_single_anchor_split_keys=dropped_single_anchor_split_keys,
                    missing_anchor_genres=set_results.get("missing_anchor_genres", []),
                    overlap_sentences=overlap_sentences,
                )
            )
            if protocol_val == "paper_parity":
                protocol_notes = list(set_results.get("protocol_notes", []))
                protocol_notes.append(
                    "Clustering scope: all multi-genre treebanks in the reconstructed test partition; "
                    "scoring scope: paper-mapped treebanks only."
                )
                if dropped_scoring_split_keys:
                    protocol_notes.append(
                        f"Paper scoring splits removed after embedding filtering: {len(dropped_scoring_split_keys)} split(s) "
                        f"({_format_protocol_split_keys(dropped_scoring_split_keys)})"
                    )
                set_results["protocol_notes"] = protocol_notes

            _display_evaluation_results(set_results)
            return

        # With broader parity anchors, scan all treebanks to build the anchor pool.
        # Otherwise, scan only explicit evaluation targets.
        if uses_single_genre_anchors and anchor_mode_val == "parity":
            scan_treebank_data = all_treebank_data
        else:
            scan_treebank_data = [
                tb for tb in all_treebank_data if tb["id"] in evaluation_treebank_ids
            ]

        if (
            uses_single_genre_anchors
            and anchor_mode_val == "parity"
            and len(scan_treebank_data) > len(evaluation_treebank_ids)
        ):
            console.print(
                f"[blue]Scanning {len(scan_treebank_data)} treebanks "
                f"({len(evaluation_treebank_ids)} evaluation target(s) + broader single-genre anchor pool)[/blue]"
            )
        else:
            console.print(
                f"[blue]Checking {len(scan_treebank_data)} treebanks for multi-genre datasets...[/blue]"
            )

        # Identify multi-genre treebanks and collect sentence metadata
        multi_genre_treebanks = []
        single_genre_treebanks = []
        sentence_metadata = {}  # Maps (tb_code, split, sent_id) -> genre

        # Track statistics
        stats = {
            'checked': 0,
            'single_genre': 0,
            'multi_genre': 0,
            'load_errors': 0,
            'no_metadata': 0,
        }
        scanned_splits = 0

        with console.status("[blue]Scanning sentence metadata...[/blue]") as status:
            for tb in scan_treebank_data:
                tb_code = tb['id']
                is_evaluation_target = tb_code in evaluation_treebank_ids
                available_splits = bootstrapper.data_loader.get_available_splits(tb_code)

                if not available_splits:
                    continue

                # Check each split for multi-genre content
                for split_name in available_splits:
                    if split_map_filter is not None and not split_map_filter.includes_split(
                        tb_code, split_name
                    ):
                        split_map_skipped_splits += 1
                        continue

                    scanned_splits += 1
                    if is_evaluation_target:
                        stats['checked'] += 1
                    status.update(
                        f"[blue]Scanning {tb_code}:{split_name} "
                        f"({scanned_splits} split(s) checked)...[/blue]"
                    )

                    # Count genres in this split
                    genre_counts = {}
                    sent_count = 0

                    try:
                        sentence_iter = bootstrapper.data_loader.iter_treebank_sentences(
                            tb_code,
                            split_name,
                            metadata_only=True,
                        )
                        for idx, sentence in enumerate(sentence_iter):
                            sent_id = sentence.get('sent_id', f'{tb_code}_{split_name}_{idx}')
                            if split_map_filter is not None and not split_map_filter.includes_sentence(
                                tb_code, split_name, sent_id
                            ):
                                split_map_skipped_sentences += 1
                                continue

                            genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)

                            if genres:
                                primary_genre = genres[0]
                                genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1
                                sentence_metadata[(tb_code, split_name, sent_id)] = primary_genre
                                sent_count += 1
                    except Exception as e:
                        logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                        if is_evaluation_target:
                            stats['load_errors'] += 1
                        continue

                    # Classify this split
                    unique_genres = list(genre_counts.keys())
                    if len(unique_genres) == 0:
                        if is_evaluation_target:
                            stats['no_metadata'] += 1
                    elif len(unique_genres) == 1:
                        if is_evaluation_target:
                            stats['single_genre'] += 1
                        if uses_single_genre_anchors:
                            single_genre_treebanks.append({
                                'treebank': tb_code,
                                'split': split_name,
                                'genres': unique_genres,
                                'language': tb['language'],
                                'sentence_count': sent_count,
                                'genre_counts': genre_counts,
                            })
                    elif len(unique_genres) >= 2:
                        if is_evaluation_target:
                            stats['multi_genre'] += 1
                            multi_genre_treebanks.append({
                                'treebank': tb_code,
                                'split': split_name,
                                'genres': unique_genres,
                                'language': tb['language'],
                                'sentence_count': sent_count,
                                'genre_counts': genre_counts,
                            })

        if split_map_filter is not None:
            console.print(
                f"[blue]Split-map scan filter:[/blue] skipped "
                f"{split_map_skipped_splits} split(s) and "
                f"{split_map_skipped_sentences} sentence row(s) outside selection"
            )

        if len(multi_genre_treebanks) == 0:
            console.print("\n[bold yellow]⚠ No multi-genre treebanks found for clustering evaluation[/bold yellow]")

            # Display summary
            console.print("\n[bold]Summary:[/bold]")
            console.print(f"  • Checked: {stats['checked']} treebank splits")
            console.print(f"  • Single-genre: {stats['single_genre']} (don't need clustering evaluation)")
            console.print(f"  • Multi-genre: {stats['multi_genre']} (suitable for evaluation)")
            if stats['no_metadata'] > 0:
                console.print(f"  • No sentence metadata: {stats['no_metadata']}")
            if stats['load_errors'] > 0:
                console.print(f"  • Load errors: {stats['load_errors']}")

            console.print("\n[dim]Clustering evaluation requires treebanks with:[/dim]")
            console.print("[dim]  • 2+ genres in the same treebank split[/dim]")
            console.print("[dim]  • Sentence-level genre metadata[/dim]")
            console.print("\n[dim]Suggestions:[/dim]")
            console.print("[dim]  • Use a config that includes multi-genre treebanks (e.g., PUD, PDTC)[/dim]")
            console.print("[dim]  • Single-genre treebanks are trivially labeled and don't need clustering[/dim]")
            console.print("\n[blue]Evaluation skipped - no suitable data available[/blue]")
            raise typer.Exit(0)

        available_multi_treebanks = sorted({tb["treebank"] for tb in multi_genre_treebanks})

        if explicit_eval_sets:
            evaluation_set_map: Dict[str, List[str]] = {}
            for set_name, set_treebanks in explicit_eval_sets.items():
                filtered_treebanks = [tb for tb in set_treebanks if tb in available_multi_treebanks]
                missing_treebanks = [tb for tb in set_treebanks if tb not in available_multi_treebanks]

                if missing_treebanks:
                    console.print(
                        f"[yellow]Set '{set_name}' ignored {len(missing_treebanks)} treebank(s) without "
                        f"multi-genre data: {', '.join(missing_treebanks)}[/yellow]"
                    )
                if not filtered_treebanks:
                    console.print(
                        f"[yellow]Set '{set_name}' has no usable multi-genre treebanks after filtering; skipping.[/yellow]"
                    )
                    continue

                evaluation_set_map[set_name] = filtered_treebanks
        elif progressive:
            evaluation_set_map = build_progressive_treebank_sets(
                multi_genre_treebanks=multi_genre_treebanks,
                min_size=n_folds_val,
                step=progressive_step,
            )
            if evaluation_set_map:
                console.print(
                    f"[blue]Generated {len(evaluation_set_map)} progressive evaluation set(s)[/blue]"
                )
        else:
            evaluation_set_map = {"default": available_multi_treebanks}

        if not evaluation_set_map:
            console.print("\n[bold yellow]⚠ No usable evaluation sets available[/bold yellow]")
            console.print("[blue]Evaluation skipped - all requested sets were empty after filtering[/blue]")
            raise typer.Exit(0)

        console.print(f"[blue]Found {len(multi_genre_treebanks)} multi-genre treebank splits for evaluation[/blue]")

        # Display multi-genre treebanks
        tb_table = Table(title="Multi-Genre Treebanks for Clustering Evaluation", show_header=True, header_style="bold magenta")
        tb_table.add_column("Treebank:Split", style="cyan")
        tb_table.add_column("Language", style="blue")
        tb_table.add_column("Genres", style="green")
        tb_table.add_column("Sentences", style="magenta", justify="right")

        for tb_info in sorted(multi_genre_treebanks, key=lambda x: (x['treebank'], x['split'])):
            tb_table.add_row(
                f"{tb_info['treebank']}:{tb_info['split']}",
                tb_info['language'],
                ", ".join(sorted(tb_info['genres'])),
                str(tb_info['sentence_count'])
            )

        console.print()
        console.print(tb_table)

        treebank_eval_stats = build_treebank_eval_stats(multi_genre_treebanks)

        if len(evaluation_set_map) > 1:
            plan_table = Table(
                title="Evaluation Set Plan",
                show_header=True,
                header_style="bold magenta",
            )
            plan_table.add_column("Set", style="cyan")
            plan_table.add_column("Treebanks", style="blue", justify="right")
            plan_table.add_column("Splits", style="green", justify="right")
            plan_table.add_column("Potential Virtual Splits", style="yellow", justify="right")
            plan_table.add_column("Status", style="magenta")

            for set_name, set_treebanks in evaluation_set_map.items():
                set_multi = [tb for tb in multi_genre_treebanks if tb["treebank"] in set_treebanks]
                can_run, reason = check_evaluation_fold_feasibility(
                    set_multi,
                    n_folds=n_folds_val,
                    group_by=group_by_val,
                )
                set_split_count = sum(
                    treebank_eval_stats[tb_code]["splits"]
                    for tb_code in set_treebanks
                    if tb_code in treebank_eval_stats
                )
                set_virtual_split_count = sum(
                    treebank_eval_stats[tb_code]["virtual_splits"]
                    for tb_code in set_treebanks
                    if tb_code in treebank_eval_stats
                )
                plan_table.add_row(
                    set_name,
                    str(len(set_treebanks)),
                    str(set_split_count),
                    str(set_virtual_split_count),
                    "ready" if can_run else f"skip ({reason})",
                )

            console.print()
            console.print(plan_table)

        # Generate embeddings for all multi-genre treebanks
        console.print(f"\n[yellow]Generating/loading embeddings...[/yellow]")
        treebank_ids_to_embed = sorted({
            tb_code
            for set_treebanks in evaluation_set_map.values()
            for tb_code in set_treebanks
        })
        if uses_single_genre_anchors and single_genre_treebanks:
            parity_anchor_treebanks = {tb["treebank"] for tb in single_genre_treebanks}
            treebank_ids_to_embed = sorted(set(treebank_ids_to_embed).union(parity_anchor_treebanks))
            console.print(
                f"[blue]Single-genre anchors:[/blue] {len(single_genre_treebanks)} split(s) "
                f"from {len(parity_anchor_treebanks)} treebank(s)"
            )
        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_ids_to_embed)
        if split_map_filter is not None:
            embeddings_by_tb, embedding_filter_stats = filter_embeddings_by_sentence_split_map(
                embeddings_by_tb,
                split_map_filter,
            )
            console.print(
                f"[blue]Split-map embedding filter:[/blue] kept "
                f"{embedding_filter_stats.kept_sentences} sentence(s) across "
                f"{embedding_filter_stats.kept_splits} split(s); dropped "
                f"{embedding_filter_stats.dropped_sentences} sentence(s) across "
                f"{embedding_filter_stats.dropped_splits} split(s)"
            )

            available_embedding_splits = set(embeddings_by_tb.keys())
            pre_filter_multi_count = len(multi_genre_treebanks)
            multi_genre_treebanks = [
                tb
                for tb in multi_genre_treebanks
                if (tb["treebank"], tb["split"]) in available_embedding_splits
            ]
            pre_filter_single_count = len(single_genre_treebanks)
            single_genre_treebanks = [
                tb
                for tb in single_genre_treebanks
                if (tb["treebank"], tb["split"]) in available_embedding_splits
            ]
            dropped_multi = pre_filter_multi_count - len(multi_genre_treebanks)
            dropped_single = pre_filter_single_count - len(single_genre_treebanks)
            if dropped_multi > 0 or dropped_single > 0:
                console.print(
                    f"[yellow]Split-map removed {dropped_multi} multi-genre split(s) "
                    f"and {dropped_single} single-genre anchor split(s) without "
                    "matching embeddings[/yellow]"
                )

            available_multi_treebanks = sorted({tb["treebank"] for tb in multi_genre_treebanks})
            filtered_evaluation_sets: Dict[str, List[str]] = {}
            for set_name, set_treebanks in evaluation_set_map.items():
                remaining_treebanks = [tb for tb in set_treebanks if tb in available_multi_treebanks]
                if not remaining_treebanks:
                    console.print(
                        f"[yellow]Set '{set_name}' has no multi-genre treebanks after "
                        "split-map embedding filtering; skipping.[/yellow]"
                    )
                    continue
                filtered_evaluation_sets[set_name] = remaining_treebanks
            evaluation_set_map = filtered_evaluation_sets

            if not evaluation_set_map:
                console.print(
                    "\n[bold yellow]⚠ No usable evaluation sets remain after split-map filtering[/bold yellow]"
                )
                console.print(
                    "[blue]Evaluation skipped - split map selection removed all candidate data[/blue]"
                )
                raise typer.Exit(0)

            treebank_eval_stats = build_treebank_eval_stats(multi_genre_treebanks)

        # Run clustering evaluation for each set
        console.print(f"\n[yellow]Running {n_folds_val}-fold clustering evaluation...[/yellow]")
        successful_results = {}

        for set_name, set_treebanks in evaluation_set_map.items():
            set_multi_treebanks = [
                tb_info for tb_info in multi_genre_treebanks
                if tb_info["treebank"] in set_treebanks
            ]
            can_run, reason = check_evaluation_fold_feasibility(
                set_multi_treebanks,
                n_folds=n_folds_val,
                group_by=group_by_val,
            )
            if not can_run:
                console.print(f"[yellow]Skipping set '{set_name}': {reason}[/yellow]")
                continue

            if len(evaluation_set_map) > 1 or set_name != "default":
                console.print(f"\n[bold cyan]Evaluation Set: {set_name}[/bold cyan]")
                console.print(
                    f"[blue]Treebanks:[/blue] {len(set_treebanks)}  "
                    f"[blue]Splits:[/blue] {len(set_multi_treebanks)}"
                )

            set_results = evaluator.k_fold_validate(
                multi_genre_treebanks=set_multi_treebanks,
                sentence_metadata=sentence_metadata,
                embeddings_by_tb=embeddings_by_tb,
                clusterer=bootstrapper.clusterer,
                single_genre_treebanks=single_genre_treebanks if uses_single_genre_anchors else None,
            )
            successful_results[set_name] = {
                "results": set_results,
                "treebanks": set_treebanks,
                "splits": len(set_multi_treebanks),
            }

            # Save confusion matrix as PNG if output path specified
            if cfg.output.genres_path and "confusion_matrix" in set_results and "genre_labels" in set_results:
                output_dir = Path(cfg.output.genres_path) / "evaluation"
                output_dir.mkdir(parents=True, exist_ok=True)
                result_label = None if len(evaluation_set_map) == 1 and set_name == "default" else set_name
                confusion_matrix_path = _save_clustering_confusion_matrix(
                    set_results,
                    output_dir=output_dir,
                    result_label=result_label,
                )
                if confusion_matrix_path is not None:
                    console.print(f"[blue]Confusion matrix saved to:[/blue] {confusion_matrix_path}")

            _display_evaluation_results(set_results)

        if not successful_results:
            console.print("\n[bold yellow]⚠ No evaluation sets satisfied fold constraints[/bold yellow]")
            console.print("[blue]Evaluation skipped - insufficient grouped data for requested n-fold setting[/blue]")
            raise typer.Exit(0)

        if len(successful_results) > 1:
            summary_table = Table(
                title="Evaluation Set Comparison",
                show_header=True,
                header_style="bold magenta",
            )
            summary_table.add_column("Set", style="cyan")
            summary_table.add_column("Treebanks", style="blue", justify="right")
            summary_table.add_column("Splits", style="green", justify="right")
            summary_table.add_column("Potential Virtual Splits", style="yellow", justify="right")
            summary_table.add_column("Overall Acc (Micro-F1)", style="magenta", justify="right")
            summary_table.add_column("Macro-F1", style="magenta", justify="right")
            summary_table.add_column("PUR", style="magenta", justify="right")
            summary_table.add_column("AGR (TB)", style="magenta", justify="right")
            summary_table.add_column("ΔBC (TB)", style="magenta", justify="right")
            summary_table.add_column("AGR (Split)", style="magenta", justify="right")
            summary_table.add_column("ΔBC (Split)", style="magenta", justify="right")
            summary_table.add_column("Mean Fold Acc (Micro-F1)", style="magenta", justify="right")

            def _fmt_metric(result: Dict, key: str) -> str:
                mean_key = f"mean_{key}"
                std_key = f"std_{key}"
                if mean_key in result and std_key in result:
                    return f"{result[mean_key]:.4f} +/- {result[std_key]:.4f}"
                value = result.get(key)
                if value is None:
                    return "n/a"
                return f"{value:.4f}"

            for set_name, payload in sorted(
                successful_results.items(),
                key=lambda item: item[1]["results"].get("overall_accuracy", -1.0),
                reverse=True,
            ):
                set_virtual_split_count = sum(
                    treebank_eval_stats[tb_code]["virtual_splits"]
                    for tb_code in payload["treebanks"]
                    if tb_code in treebank_eval_stats
                )
                set_results = payload["results"]
                summary_table.add_row(
                    set_name,
                    str(len(payload["treebanks"])),
                    str(payload["splits"]),
                    str(set_virtual_split_count),
                    _fmt_metric(set_results, "overall_accuracy"),
                    _fmt_metric(set_results, "macro_f1_instance"),
                    _fmt_metric(set_results, "purity"),
                    _fmt_metric(set_results, "agreement_treebank"),
                    _fmt_metric(set_results, "overlap_error_treebank"),
                    _fmt_metric(set_results, "agreement_split"),
                    _fmt_metric(set_results, "overlap_error_split"),
                    f"{set_results['mean_accuracy']:.4f} +/- {set_results['std_accuracy']:.4f}",
                )

            console.print()
            console.print(summary_table)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Evaluation failed")
        raise typer.Exit(1)


@app.command("build-sentence-split-map")
def build_sentence_split_map(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    ud_source: Optional[str] = typer.Option(
        None,
        "--ud-source",
        help=(
            "UD source URI (e.g., 'hf://universal-dependencies/universal_dependencies' or "
            "'local:///path/to/UD_repos'). Overrides config."
        ),
    ),
    ud_version: Optional[str] = typer.Option(
        None,
        "--ud-version",
        help="UD version/revision (used as HF revision). Overrides config.",
    ),
    metadata_path: Optional[Path] = typer.Option(
        None,
        "--metadata-path",
        help="Optional explicit metadata.json path. Overrides config.",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    split_pickle: Path = typer.Option(
        ...,
        "--split-pickle",
        help="Path to paper split pickle (e.g., 102-915-204.pkl).",
        exists=True,
        dir_okay=False,
        file_okay=True,
    ),
    output: Path = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output split map file (.parquet, .csv, or .tsv).",
    ),
    partition: Optional[List[str]] = typer.Option(
        None,
        "--partition",
        "-p",
        help=(
            "Partition(s) to export from split pickle (e.g., train/dev/test). "
            "Repeat for multiple partitions; defaults to all."
        ),
    ),
):
    """Convert global sentence-index split files into explicit sentence mapping rows."""
    console.print("\n[bold cyan]Build Sentence Split Map[/bold cyan]")
    console.print("=" * 60)

    try:
        from ud_genre_bootstrap.utils.paper_split_converter import (
            build_sentence_split_map_from_index_split,
        )

        cfg = load_config_from_path(config)
        ud_source_val = ud_source if ud_source is not None else cfg.ud_source
        ud_version_val = ud_version if ud_version is not None else cfg.ud_version
        metadata_path_val = metadata_path
        if metadata_path_val is None and cfg.metadata_path:
            metadata_path_val = Path(cfg.metadata_path)

        output.parent.mkdir(parents=True, exist_ok=True)
        stats = build_sentence_split_map_from_index_split(
            split_pickle_path=split_pickle,
            output_path=output,
            partitions=partition,
            ud_source=ud_source_val,
            ud_version=ud_version_val,
            metadata_path=metadata_path_val,
        )

        selected_partitions = ", ".join(stats.selected_partitions)
        available_partitions = ", ".join(stats.available_partitions)

        console.print(f"[blue]UD source:[/blue] {ud_source_val}")
        console.print(f"[blue]UD version:[/blue] {ud_version_val}")
        if metadata_path_val is not None:
            console.print(f"[blue]Metadata path:[/blue] {metadata_path_val}")
        console.print(f"[blue]Split pickle:[/blue] {split_pickle}")
        console.print(f"[blue]Output:[/blue] {stats.output_path}")
        console.print(f"[blue]Partitions:[/blue] {selected_partitions} (available: {available_partitions})")
        console.print(f"[blue]Scanned treebanks:[/blue] {stats.scanned_treebanks}")
        console.print(f"[blue]Scanned split units:[/blue] {stats.scanned_files}")
        console.print(f"[blue]Scanned sentences:[/blue] {stats.scanned_sentences}")
        console.print(f"[blue]Target indices:[/blue] {stats.target_indices}")
        console.print(f"[green]Matched indices:[/green] {stats.matched_target_indices}")
        console.print(f"[green]Rows written:[/green] {stats.rows_written}")

        if stats.rows_missing_sent_id > 0:
            console.print(
                f"[yellow]Rows skipped (missing sent_id):[/yellow] {stats.rows_missing_sent_id}"
            )
        if stats.load_errors > 0:
            treebanks = ", ".join(stats.load_error_treebanks)
            console.print(
                f"[yellow]Splits skipped (load errors):[/yellow] {stats.load_errors} "
                f"(treebanks: {treebanks})"
            )
        if stats.unsupported_files > 0:
            console.print(
                f"[yellow]Files skipped (non-standard filename):[/yellow] {stats.unsupported_files}"
            )
        if stats.unmatched_target_indices > 0:
            console.print(
                f"[bold yellow]⚠ Unmatched target indices:[/bold yellow] "
                f"{stats.unmatched_target_indices} "
                f"(max target index={stats.max_target_index}, scanned={stats.scanned_sentences})"
            )
        else:
            console.print("[bold green]✓ All selected target indices were matched[/bold green]")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Split-map conversion failed")
        raise typer.Exit(1)


@app.command()
def coverage(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration file"
    ),
    treebank: Optional[str] = typer.Option(
        None, "--treebank", "-t", help="Comma-separated list of treebank codes to analyze"
    ),
    threshold: float = typer.Option(
        0.95, "--threshold", help="Coverage threshold for 'fully covered' (0.0-1.0)"
    ),
    only_full: bool = typer.Option(
        False, "--only-full", help="Show only fully covered treebanks"
    ),
    only_partial: bool = typer.Option(
        False, "--only-partial", help="Show only partially covered treebanks"
    ),
    show_splits: bool = typer.Option(
        True, "--show-splits/--no-splits", help="Show per-split details"
    ),
    export: Optional[Path] = typer.Option(
        None, "--export", "-o", help="Export coverage data to JSON file"
    ),
):
    """Analyze sentence-level genre coverage across treebanks.

    Reports which treebanks have sufficient sentence-level genre metadata
    for evaluation and production use. Analyzes coverage across all splits
    (train/dev/test) and identifies fully/partially covered treebanks.

    Examples:
        # Analyze all treebanks from config
        ud-genre-bootstrap coverage --config myconfig.yaml

        # Analyze specific treebanks
        ud-genre-bootstrap coverage --treebank en_pud,de_pud,cs_pdtc

        # Show only fully covered treebanks
        ud-genre-bootstrap coverage --only-full --threshold 0.95

        # Export coverage data
        ud-genre-bootstrap coverage --export coverage.json
    """
    try:
        console.print("\n[bold]Genre Coverage Analysis[/bold]")
        console.print("=" * 60)

        # Load configuration
        cfg = load_config_from_path(config)
        console.print(f"Loading config from: {config}")

        # Initialize components
        from ud_genre_bootstrap.bootstrapping.bootstrapper import GenreBootstrapper
        from ud_genre_bootstrap.utils.genre_coverage import GenreCoverageAnalyzer

        console.print("\n[yellow]Initializing coverage analyzer...[/yellow]")
        bootstrapper = GenreBootstrapper(cfg)

        analyzer = GenreCoverageAnalyzer(
            data_loader=bootstrapper.data_loader,
            genre_mapper=bootstrapper.genre_mapper,
        )

        # Determine which treebanks to analyze
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
            console.print(f"[blue]Analyzing {len(treebank_filter)} specified treebanks[/blue]")
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            console.print(f"[blue]Analyzing {len(treebank_filter)} treebanks from config[/blue]")
        else:
            # Get all treebanks from metadata
            all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()
            treebank_filter = [tb['id'] for tb in all_treebank_data]
            console.print(f"[blue]Analyzing all {len(treebank_filter)} treebanks[/blue]")

        treebank_filter = apply_treebank_exclusions(
            cfg,
            bootstrapper.data_loader,
            treebank_filter,
        )
        console.print(f"[blue]Analyzing {len(treebank_filter)} treebanks after exclusions[/blue]")
        console.print(f"[blue]Coverage threshold: {threshold * 100:.0f}%[/blue]\n")

        # Analyze treebanks
        console.print("[yellow]Analyzing genre coverage...[/yellow]")
        results = analyzer.analyze_treebanks(treebank_filter)

        # Categorize treebanks
        fully_covered = []
        partially_covered = []
        not_covered = []
        ignored = []  # Treebanks with no successfully loaded splits

        for tb_code, coverage in results.items():
            if not coverage.splits:
                # No splits were successfully loaded
                ignored.append(tb_code)
            elif coverage.is_fully_covered(threshold):
                fully_covered.append((tb_code, coverage))
            elif coverage.is_partially_covered(threshold):
                partially_covered.append((tb_code, coverage))
            else:
                not_covered.append((tb_code, coverage))

        # Display summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"[green]✓ Fully covered:[/green] {len(fully_covered)} treebanks (all splits ≥{threshold*100:.0f}%)")
        console.print(f"[yellow]⚠ Partially covered:[/yellow] {len(partially_covered)} treebanks (some splits ≥{threshold*100:.0f}%, some <{threshold*100:.0f}%)")
        console.print(f"[red]✗ Not covered:[/red] {len(not_covered)} treebanks (all splits <{threshold*100:.0f}%)")
        if ignored:
            console.print(f"[dim]  Ignored: {len(ignored)} treebanks (load errors: {', '.join(sorted(ignored))})[/dim]")

        # Get canonical genres for marking non-canonical ones
        canonical_genres = set(analyzer.genre_mapper.canonical_genres) if analyzer.genre_mapper.canonical_genres else set()

        # Display detailed tables based on filters
        if only_full:
            _display_fully_covered_table(fully_covered, show_splits, canonical_genres)
        elif only_partial:
            _display_partially_covered_table(partially_covered, threshold, show_splits, canonical_genres)
        else:
            # Show all categories
            if fully_covered:
                _display_fully_covered_table(fully_covered, show_splits, canonical_genres)

            if partially_covered:
                _display_partially_covered_table(partially_covered, threshold, show_splits, canonical_genres)

            if not_covered and len(not_covered) <= 20:
                _display_not_covered_table(not_covered, canonical_genres)
            elif not_covered:
                console.print(f"\n[dim]({len(not_covered)} treebanks without sufficient coverage - use --only-partial to see details)[/dim]")

        # Export if requested
        if export:
            _export_coverage_data(results, export, threshold)
            console.print(f"\n[blue]Coverage data exported to:[/blue] {export}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Coverage analysis failed")
        raise typer.Exit(1)


def _display_fully_covered_table(fully_covered: list, show_splits: bool, canonical_genres: set):
    """Display table of fully covered treebanks."""
    from rich.table import Table

    if not fully_covered:
        return

    table = Table(title="\nFully Covered Treebanks", show_header=True, header_style="bold green")
    table.add_column("Treebank", style="cyan")
    table.add_column("Splits", style="blue")
    table.add_column("Sentences", style="magenta", justify="right")
    table.add_column("Coverage", style="green", justify="right")
    table.add_column("Genres", style="yellow")

    has_non_canonical = False
    for tb_code, coverage in sorted(fully_covered):
        splits_str = ", ".join(coverage.all_splits)
        # Mark non-canonical genres with asterisk
        marked_genres = []
        for genre in sorted(coverage.all_genres):
            if genre not in canonical_genres:
                marked_genres.append(f"*{genre}")
                has_non_canonical = True
            else:
                marked_genres.append(genre)
        genres_str = ", ".join(marked_genres)
        table.add_row(
            tb_code,
            splits_str,
            str(coverage.total_sentences),
            f"{coverage.overall_coverage * 100:.1f}%",
            genres_str,
        )

    console.print()
    console.print(table)
    if has_non_canonical:
        console.print("[dim]* Non-canonical genre[/dim]")


def _display_partially_covered_table(partially_covered: list, threshold: float, show_splits: bool, canonical_genres: set):
    """Display table of partially covered treebanks."""
    from rich.table import Table

    if not partially_covered:
        return

    table = Table(title="\nPartially Covered Treebanks", show_header=True, header_style="bold yellow")
    table.add_column("Treebank", style="cyan")
    table.add_column("Covered Splits", style="green")
    table.add_column("Uncovered Splits", style="red")
    table.add_column("Overall Coverage", style="yellow", justify="right")
    table.add_column("Genres", style="yellow")

    has_non_canonical = False
    for tb_code, coverage in sorted(partially_covered):
        covered = ", ".join(coverage.get_covered_splits(threshold))
        uncovered = ", ".join(coverage.get_uncovered_splits(threshold))
        # Mark non-canonical genres with asterisk
        marked_genres = []
        for genre in sorted(coverage.all_genres):
            if genre not in canonical_genres:
                marked_genres.append(f"*{genre}")
                has_non_canonical = True
            else:
                marked_genres.append(genre)
        genres_str = ", ".join(marked_genres)
        table.add_row(
            tb_code,
            covered or "—",
            uncovered or "—",
            f"{coverage.overall_coverage * 100:.1f}%",
            genres_str,
        )

    console.print()
    console.print(table)
    if has_non_canonical:
        console.print("[dim]* Non-canonical genre[/dim]")


def _display_not_covered_table(not_covered: list, canonical_genres: set):
    """Display table of treebanks without coverage."""
    from rich.table import Table

    if not not_covered:
        return

    table = Table(title="\nTreebanks Without Sufficient Coverage", show_header=True, header_style="bold red")
    table.add_column("Treebank", style="cyan")
    table.add_column("Splits", style="blue")
    table.add_column("Coverage", style="red", justify="right")
    table.add_column("Genres", style="yellow")

    has_non_canonical = False
    for tb_code, coverage in sorted(not_covered):
        splits_str = ", ".join(coverage.all_splits)
        # Mark non-canonical genres with asterisk
        marked_genres = []
        for genre in sorted(coverage.all_genres):
            if genre not in canonical_genres:
                marked_genres.append(f"*{genre}")
                has_non_canonical = True
            else:
                marked_genres.append(genre)
        genres_str = ", ".join(marked_genres)
        table.add_row(
            tb_code,
            splits_str,
            f"{coverage.overall_coverage * 100:.1f}%",
            genres_str,
        )

    console.print()
    console.print(table)
    if has_non_canonical:
        console.print("[dim]* Non-canonical genre[/dim]")


def _export_coverage_data(results: dict, output_path: Path, threshold: float):
    """Export coverage data to JSON file."""
    import json

    export_data = {
        'threshold': threshold,
        'treebanks': {},
    }

    for tb_code, coverage in results.items():
        export_data['treebanks'][tb_code] = {
            'splits': {
                split_name: {
                    'total_sentences': split_cov.total_sentences,
                    'sentences_with_genre': split_cov.sentences_with_genre,
                    'coverage': split_cov.coverage,
                    'genres': sorted(split_cov.genres),
                    'genre_counts': split_cov.genre_counts,
                }
                for split_name, split_cov in coverage.splits.items()
            },
            'overall': {
                'total_sentences': coverage.total_sentences,
                'sentences_with_genre': coverage.total_with_genre,
                'coverage': coverage.overall_coverage,
                'genres': sorted(coverage.all_genres),
                'fully_covered': coverage.is_fully_covered(threshold),
                'partially_covered': coverage.is_partially_covered(threshold),
                'covered_splits': coverage.get_covered_splits(threshold),
                'uncovered_splits': coverage.get_uncovered_splits(threshold),
            }
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)


@app.command()
def evaluate_xgenre(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to evaluate (e.g., en_ewt or en_ewt,de_gsd). If not specified, evaluates all bootstrap-derived sentences.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for X-GENRE evaluation results. Default: {output.genres_path}/xgenre_evaluation/",
    ),
):
    """Evaluate bootstrap quality using X-GENRE classifier.

    Compares bootstrap-derived sentences (multi-genre treebanks) against
    X-GENRE predictions. Only evaluates sentences labeled via bootstrap,
    not pre-existing metadata or single-genre treebanks.

    This provides an independent assessment of bootstrap labeling quality
    using a pre-trained multilingual genre classifier as ground truth.
    """
    console.print("\n[bold cyan]X-GENRE Bootstrap Evaluation[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Parse treebank filter
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
        elif cfg.include_treebanks:
            treebank_filter = cfg.include_treebanks
            console.print(f"[blue]Using treebanks from config:[/blue] {', '.join(treebank_filter)}")

        # Determine output directory
        if output:
            output_dir = Path(output)
        else:
            output_dir = Path(cfg.output.genres_path) / "xgenre_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load bootstrap results
        bootstrap_file = Path(cfg.output.genres_path) / "all_genres.parquet"
        if not bootstrap_file.exists():
            console.print(f"[bold red]✗ Error:[/bold red] Bootstrap results not found at {bootstrap_file}")
            console.print("[yellow]Run 'label' or 'run' command first to generate bootstrap labels[/yellow]")
            raise typer.Exit(1)

        console.print(f"[blue]Loading bootstrap results from:[/blue] {bootstrap_file}")
        import pandas as pd
        df_bootstrap = pd.read_parquet(bootstrap_file)

        # Filter to only bootstrap-derived sentences
        # (exclude metadata and single-genre treebanks)
        console.print("\n[yellow]Filtering to bootstrap-derived sentences only...[/yellow]")
        bootstrap_methods = {'bootstrap-labeled', 'bootstrap-inferred'}
        df_bootstrap_only = df_bootstrap[df_bootstrap['method'].isin(bootstrap_methods)].copy()

        if len(df_bootstrap_only) == 0:
            console.print("[bold red]✗ Error:[/bold red] No bootstrap-derived sentences found")
            console.print(
                "[yellow]Expected method tags: bootstrap-labeled / bootstrap-inferred[/yellow]"
            )
            raise typer.Exit(1)

        export_has_sentence_key = {"treebank", "split", "sent_id"}.issubset(df_bootstrap_only.columns)

        # Apply treebank filter if specified
        if treebank_filter:
            if export_has_sentence_key:
                df_bootstrap_only = df_bootstrap_only[df_bootstrap_only['treebank'].isin(treebank_filter)]
            else:
                # Legacy schema fallback: infer treebank from sent_id string.
                df_bootstrap_only['treebank'] = df_bootstrap_only['sent_id'].str.extract(
                    r'^([a-z]{2,3}_[a-z]+)',
                    expand=False,
                )
                df_bootstrap_only = df_bootstrap_only[df_bootstrap_only['treebank'].isin(treebank_filter)]

            if len(df_bootstrap_only) == 0:
                console.print(f"[bold red]✗ Error:[/bold red] No bootstrap-derived sentences found for specified treebanks")
                raise typer.Exit(1)

        console.print(f"[green]✓ Found {len(df_bootstrap_only)} bootstrap-derived sentences[/green]")

        # Show breakdown by bootstrap method
        method_counts = df_bootstrap.groupby('method').size()
        console.print("\n[cyan]Label method breakdown (all sentences):[/cyan]")
        for method, count in method_counts.items():
            console.print(f"  {method}: {count} ({count/len(df_bootstrap):.1%})")
        console.print(f"[yellow]→ Evaluating only the {len(df_bootstrap_only)} bootstrap-derived sentences[/yellow]")

        # Load sentences from data loader to get text
        console.print("\n[yellow]Loading sentence texts...[/yellow]")
        from ud_genre_bootstrap.utils.data_loader import UDDataLoader

        data_loader = UDDataLoader(
            ud_source=cfg.ud_source,
            ud_version=cfg.ud_version,
            metadata_path=Path(cfg.metadata_path) if cfg.metadata_path else None,
        )

        # Build sentence-ref -> text mapping
        sent_id_to_text = {}

        for tb_code, split, dataset in data_loader.iter_all_treebanks(treebank_filter=treebank_filter):
            for sentence in dataset:
                sent_id = sentence.get('sent_id')
                text = sentence.get('text', '')
                if sent_id and text:
                    if export_has_sentence_key:
                        sent_ref = qualify_sentence_ref(tb_code, split, sent_id)
                        sent_id_to_text[sent_ref] = text
                    else:
                        sent_id_to_text[sent_id] = text

        # Filter df to only sentences we have text for
        if export_has_sentence_key:
            df_bootstrap_only['_sent_ref'] = [
                qualify_sentence_ref(tb, split, sent_id)
                for tb, split, sent_id in zip(
                    df_bootstrap_only['treebank'],
                    df_bootstrap_only['split'],
                    df_bootstrap_only['sent_id'],
                )
            ]
            df_bootstrap_only = df_bootstrap_only[df_bootstrap_only['_sent_ref'].isin(sent_id_to_text)]
        else:
            df_bootstrap_only = df_bootstrap_only[df_bootstrap_only['sent_id'].isin(sent_id_to_text)]
        console.print(f"[green]✓ Loaded texts for {len(df_bootstrap_only)} sentences[/green]")

        if len(df_bootstrap_only) == 0:
            console.print("[bold red]✗ Error:[/bold red] No sentence texts found")
            raise typer.Exit(1)

        # Load X-GENRE classifier
        console.print(f"\n[yellow]Loading X-GENRE classifier: {cfg.xgenre_evaluation.model}[/yellow]")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        # Determine device
        device_str = cfg.xgenre_evaluation.device.lower()
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_str == "cuda":
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        console.print(f"[blue]Using device:[/blue] {device}")

        tokenizer = AutoTokenizer.from_pretrained(cfg.xgenre_evaluation.model)
        model = AutoModelForSequenceClassification.from_pretrained(cfg.xgenre_evaluation.model)
        model.to(device)
        model.eval()

        # X-GENRE label mapping
        xgenre_labels = ['Other', 'Information/Explanation', 'News', 'Instruction',
                         'Opinion/Argumentation', 'Forum', 'Prose/Lyrical', 'Legal', 'Promotion']

        # Run X-GENRE predictions
        console.print("\n[yellow]Running X-GENRE predictions...[/yellow]")
        xgenre_predictions = []
        batch_size = cfg.xgenre_evaluation.batch_size

        if export_has_sentence_key:
            texts = [sent_id_to_text[sent_ref] for sent_ref in df_bootstrap_only['_sent_ref']]
        else:
            texts = [sent_id_to_text[sid] for sid in df_bootstrap_only['sent_id']]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Classifying sentences...", total=len(texts))

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]

                inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model(**inputs)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    xgenre_predictions.extend([xgenre_labels[p.item()] for p in predictions])

                progress.update(task, advance=len(batch_texts))

        # Map X-GENRE labels to UD genres
        console.print("\n[yellow]Mapping X-GENRE labels to UD genres...[/yellow]")
        xgenre_mapped = [cfg.xgenre_evaluation.genre_mapping.get(label) for label in xgenre_predictions]

        # Add predictions to dataframe
        df_bootstrap_only['xgenre_label'] = xgenre_predictions
        df_bootstrap_only['xgenre_mapped'] = xgenre_mapped

        # Filter out sentences where X-GENRE predicted "Other" (no UD equivalent)
        df_eval = df_bootstrap_only[df_bootstrap_only['xgenre_mapped'].notna()].copy()
        console.print(f"[blue]Evaluating {len(df_eval)} sentences (excluded {len(df_bootstrap_only) - len(df_eval)} 'Other' predictions)[/blue]")

        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

        y_true = df_eval['xgenre_mapped'].values
        y_pred = df_eval['genre'].values

        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, labels=sorted(set(y_true)))

        # Confusion matrix
        labels = sorted(set(y_true) | set(y_pred))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=labels)

        # Display results
        console.print("\n[bold cyan]Evaluation Results[/bold cyan]")
        console.print("=" * 60)
        console.print(f"[green]Overall Accuracy:[/green] {accuracy:.3f}")
        console.print(f"[blue]Evaluated sentences:[/blue] {len(df_eval)}")

        # Per-genre results table
        console.print("\n[bold]Per-Genre Results:[/bold]")
        from rich.table import Table as RichTable

        table = RichTable(show_header=True, header_style="bold magenta")
        table.add_column("Genre", style="cyan")
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall", justify="right", style="yellow")
        table.add_column("F1-Score", justify="right", style="blue")
        table.add_column("Support", justify="right")

        for i, genre in enumerate(sorted(set(y_true))):
            table.add_row(
                genre,
                f"{precision[i]:.3f}",
                f"{recall[i]:.3f}",
                f"{f1[i]:.3f}",
                str(support[i])
            )

        console.print(table)

        # Confusion matrix
        console.print("\n[bold]Confusion Matrix:[/bold]")
        console.print("[dim]Rows = X-GENRE (ground truth), Columns = Bootstrap (predictions)[/dim]")

        cm_table = RichTable(show_header=True, header_style="bold magenta")
        cm_table.add_column("X-GENRE \\ Bootstrap", style="cyan")

        for label in labels:
            cm_table.add_column(label, justify="right", style="yellow")

        for i, true_label in enumerate(labels):
            row = [true_label]
            for j in range(len(labels)):
                count = conf_matrix[i][j]
                if i == j:
                    row.append(f"[bold green]{count}[/bold green]")
                else:
                    row.append(str(count))
            cm_table.add_row(*row)

        console.print(cm_table)

        # Save results
        console.print(f"\n[yellow]Saving results to {output_dir}...[/yellow]")

        # Save detailed predictions
        predictions_file = output_dir / "xgenre_predictions.parquet"
        if '_sent_ref' in df_eval.columns:
            df_eval = df_eval.drop(columns=['_sent_ref'])
        df_eval.to_parquet(predictions_file, index=False)
        console.print(f"[green]✓ Saved predictions:[/green] {predictions_file}")

        # Save metrics as JSON
        import json
        metrics = {
            "accuracy": float(accuracy),
            "num_sentences": int(len(df_eval)),
            "per_genre": {
                genre: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }
                for i, genre in enumerate(sorted(set(y_true)))
            },
            "confusion_matrix": conf_matrix.tolist(),
            "labels": labels,
        }

        metrics_file = output_dir / "xgenre_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        console.print(f"[green]✓ Saved metrics:[/green] {metrics_file}")

        # Save confusion matrix as PNG
        console.print("\n[yellow]Generating confusion matrix visualization...[/yellow]")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Bootstrap Prediction')
        plt.ylabel('X-GENRE Ground Truth')
        plt.title('Bootstrap vs X-GENRE Comparison')
        plt.tight_layout()

        cm_file = output_dir / "confusion_matrix.png"
        plt.savefig(cm_file, dpi=150, bbox_inches='tight')
        plt.close()
        console.print(f"[green]✓ Saved confusion matrix:[/green] {cm_file}")

        console.print(f"\n[bold green]✓ X-GENRE evaluation complete![/bold green]")
        console.print(f"[blue]Results saved to:[/blue] {output_dir}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("X-GENRE evaluation failed")
        raise typer.Exit(1)


@app.command()
def test_genres(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to test (e.g., en_ewt or en_ewt,de_gsd). If not specified, tests all.",
    ),
    split: Optional[str] = typer.Option(
        "train",
        "--split",
        "-s",
        help="Split to test (train, dev, test)",
    ),
    limit: int = typer.Option(
        100,
        "--limit",
        "-n",
        help="Number of sentences to process (0 for all)",
    ),
    show_examples: bool = typer.Option(
        True,
        "--examples/--no-examples",
        help="Show example sentences for each pattern match",
    ),
):
    """Test and debug genre extraction patterns.

    Applies genre extraction to treebank(s) and shows detailed statistics
    about pattern matches, extracted genres, and unmatched sentences.
    """
    console.print("\n[bold cyan]Genre Extraction Test & Debug[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        from pathlib import Path
        from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
        from ud_genre_bootstrap.utils.data_loader import UDDataLoader

        # Initialize components
        data_loader = UDDataLoader(
            ud_source=cfg.ud_source,
            ud_version=cfg.ud_version,
            metadata_path=Path(cfg.metadata_path) if cfg.metadata_path else None,
        )

        # Initialize genre mapper with patterns
        mapping_path = None
        patterns_path = None
        if cfg.genre_extraction.mapping_path:
            mapping_path = Path(cfg.genre_extraction.mapping_path)
        if cfg.genre_extraction.patterns_path:
            # Handle both string and list of strings
            if isinstance(cfg.genre_extraction.patterns_path, list):
                patterns_path = [Path(p) for p in cfg.genre_extraction.patterns_path]
            else:
                patterns_path = Path(cfg.genre_extraction.patterns_path)

        genre_mapper = GenreMapper(
            genre_mapping_path=mapping_path,
            metadata_patterns_path=patterns_path,
            canonical_genres=cfg.genre_extraction.canonical_genres,
            data_loader=data_loader,
        )

        # Determine which treebanks to test
        if treebank:
            # Parse comma-separated treebank list from CLI
            treebanks_to_test = [tb.strip() for tb in treebank.split(",")]
        elif cfg.include_treebanks:
            # Use treebanks from config if specified
            treebanks_to_test = cfg.include_treebanks
            console.print(f"[blue]Testing {len(treebanks_to_test)} treebanks from config[/blue]\n")
        else:
            # If patterns are defined, only test treebanks with patterns
            if genre_mapper.metadata_patterns:
                # Get treebanks that have patterns defined
                treebanks_with_patterns = list(genre_mapper.metadata_patterns.keys())
                # Also include all treebanks if we want to show coverage
                all_treebanks = data_loader.get_treebank_codes()

                # Prioritize treebanks with patterns
                treebanks_to_test = [
                    tb for tb in all_treebanks if tb in treebanks_with_patterns
                ][:10]

                if treebanks_to_test:
                    console.print(
                        f"[yellow]Testing {len(treebanks_to_test)} treebanks with patterns defined. "
                        f"Use --treebank to test specific one.[/yellow]\n"
                    )
                else:
                    # Fallback to first 10 if no patterns match
                    treebanks_to_test = all_treebanks[:10]
                    console.print(
                        f"[yellow]No patterns match available treebanks. "
                        f"Testing first 10 treebanks.[/yellow]\n"
                    )
            else:
                treebanks_to_test = data_loader.get_treebank_codes()[:10]
                console.print(f"[yellow]No patterns defined. Testing first 10 treebanks.[/yellow]\n")

        # Apply exclusions from config (unless specific treebank requested via CLI)
        if cfg.exclude_treebanks and not treebank:
            original_count = len(treebanks_to_test)
            treebanks_to_test = [tb for tb in treebanks_to_test if tb not in cfg.exclude_treebanks]
            excluded_count = original_count - len(treebanks_to_test)
            if excluded_count > 0:
                console.print(f"[dim]Excluded {excluded_count} treebanks from config exclusion list[/dim]\n")

        # Process each treebank
        for tb_code in treebanks_to_test:
            # Check if the split exists for this treebank
            available_splits = data_loader.get_available_splits(tb_code)
            if split not in available_splits:
                console.print(
                    f"\n[yellow]⚠ Skipping {tb_code}: split '{split}' not available. "
                    f"Available: {', '.join(available_splits)}[/yellow]"
                )
                continue

            _test_treebank_genres(
                data_loader,
                genre_mapper,
                tb_code,
                split,
                limit,
                show_examples,
                console,
            )

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Genre extraction test failed")
        raise typer.Exit(1)


@app.command()
def info(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
    ),
):
    """Display configuration and system information."""
    console.print("\n[bold cyan]UD Genre Bootstrap - Configuration Info[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Create configuration table
        table = Table(title="Configuration", show_header=True, header_style="bold magenta")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("UD Version", cfg.ud_version)
        table.add_row("UD Source", cfg.ud_source)
        table.add_row("Metadata Path", cfg.metadata_path or "auto")
        table.add_row("", "")
        table.add_row("[bold]Embeddings[/bold]", "")
        table.add_row("  Model", cfg.embeddings.model)
        table.add_row("  Pooling", cfg.embeddings.pooling)
        table.add_row("  Batch Size", str(cfg.embeddings.batch_size))
        table.add_row("  Device", cfg.embeddings.device)
        table.add_row("", "")
        table.add_row("[bold]Clustering[/bold]", "")
        table.add_row("  Method", cfg.clustering.method)
        table.add_row("  Level", cfg.clustering.level)
        table.add_row("  Seed", str(cfg.clustering.seed))
        table.add_row("", "")
        table.add_row("[bold]Bootstrapping[/bold]", "")
        table.add_row("  Min Confidence", str(cfg.bootstrapping.min_confidence))
        table.add_row("  Min Margin", str(cfg.bootstrapping.min_margin))
        table.add_row("  Reference Weighting", str(cfg.bootstrapping.reference_weighting))
        table.add_row("  Max Iterations", str(cfg.bootstrapping.max_iterations))
        table.add_row("  Fail on Incomplete", str(cfg.bootstrapping.fail_on_incomplete))
        table.add_row(
            "  Virtual Split Coverage",
            str(cfg.evaluation.metadata_validation.coverage_threshold),
        )
        table.add_row(
            "  Virtual Split Min Genre Sentences",
            str(cfg.evaluation.metadata_validation.min_genre_sentences),
        )
        table.add_row("", "")
        table.add_row("[bold]Output[/bold]", "")
        table.add_row("  Genres Path", cfg.output.genres_path)
        table.add_row("  Push to Hub", str(cfg.output.push_to_hub))

        console.print(table)

        # Display HF Hub settings if configured
        if cfg.output.push_to_hub:
            console.print("\n[bold cyan]HuggingFace Hub Settings[/bold cyan]")
            console.print(f"  Embeddings Repo: {cfg.output.embeddings_hf_repo}")
            console.print(f"  Genres Repo: {cfg.output.genres_hf_repo}")
            console.print(f"  Revision: {cfg.output.embeddings_revision}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise typer.Exit(1)


@app.command()
def visualize_clusters(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration YAML file (provides default paths)",
        exists=True,
        dir_okay=False,
    ),
    clusters: Optional[Path] = typer.Option(
        None,
        "--clusters",
        "-c",
        help="Path to cluster output directory (contains cluster_assignments.parquet). If not specified, uses output.genres_path/clusters/ from config.",
    ),
    embeddings: Optional[Path] = typer.Option(
        None,
        "--embeddings",
        "-e",
        help="Path to embeddings cache directory. If not specified, uses config.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for visualization (PNG/HTML). Default: clusters/visualization.html",
    ),
    method: str = typer.Option(
        "umap",
        "--method",
        help="Dimensionality reduction method: 'umap' or 'tsne'",
    ),
    n_jobs: int = typer.Option(
        -1,
        "--n-jobs",
        help="Number of parallel jobs for UMAP (-1 uses all CPUs, only works without --random-state)",
    ),
    random_state: Optional[int] = typer.Option(
        None,
        "--random-state",
        help="Random seed for reproducibility (disables parallelism in UMAP)",
    ),
    use_gpu: Optional[bool] = typer.Option(
        None,
        "--use-gpu/--no-gpu",
        help="Use GPU acceleration if available (requires cuML for UMAP). If not specified, uses config clustering.device setting.",
    ),
    treebank: Optional[str] = typer.Option(
        None,
        "--treebank",
        "-t",
        help="Specific treebank(s) to visualize (e.g., en_ewt or en_ewt,de_gsd). If not specified, visualizes all.",
    ),
    color_by: str = typer.Option(
        "genre",
        "--color-by",
        help="Attribute to color points by: 'genre', 'cluster', 'treebank'",
    ),
):
    """Visualize cluster assignments using dimensionality reduction.

    Creates interactive plots showing how sentences cluster together.
    By default, colors points by genre to evaluate clustering quality.

    Usage:
        # Simple: just provide config (uses default paths)
        uv run ud-genre-bootstrap visualize-clusters --config configs/default.yaml

        # Override specific paths if needed
        uv run ud-genre-bootstrap visualize-clusters --config configs/default.yaml --clusters output/custom/

    """
    console.print("\n[bold cyan]Cluster Visualization[/bold cyan]")
    console.print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import json
        from pathlib import Path as PathLib

        # Load config to get default paths
        cfg = None
        if config:
            cfg = load_config_from_path(config)

        # Determine clusters directory - CLI overrides config
        if clusters is None:
            if cfg is None:
                console.print("[bold red]✗ Error:[/bold red] Either --config or --clusters must be specified")
                raise typer.Exit(1)

            # Use default from config
            clusters_dir = PathLib(cfg.output.genres_path) / "clusters"
            console.print(f"[blue]Using default cluster directory from config:[/blue] {clusters_dir}")
        else:
            clusters_dir = PathLib(clusters)

        # Validate clusters directory exists
        if not clusters_dir.exists():
            console.print(f"[bold red]✗ Error:[/bold red] Cluster directory does not exist: {clusters_dir}")
            raise typer.Exit(1)

        # Load cluster assignments
        assignments_file = clusters_dir / "cluster_assignments.parquet"
        if not assignments_file.exists():
            console.print(f"[bold red]✗ Error:[/bold red] Cluster assignments not found at {assignments_file}")
            raise typer.Exit(1)

        console.print(f"[blue]Loading cluster assignments from:[/blue] {assignments_file}")
        df_clusters = pd.read_parquet(assignments_file)

        # Try to load sentence-level genre assignments from bootstrap labeling
        # Check both in the clusters directory and in the parent output directory
        genre_labels_file = None
        possible_paths = [
            clusters_dir.parent / "all_genres.parquet",  # Parent of clusters dir
            clusters_dir / "all_genres.parquet",  # In clusters dir itself
        ]

        for path in possible_paths:
            if path.exists():
                genre_labels_file = path
                break

        genre_lookup = {}
        genres_use_sentence_key = False
        if genre_labels_file:
            console.print(f"[blue]Loading sentence-level genre assignments from:[/blue] {genre_labels_file}")
            df_genres = pd.read_parquet(genre_labels_file)
            genre_lookup, genres_use_sentence_key = build_exported_genre_lookup(df_genres)
            console.print(f"[green]✓ Loaded {len(genre_lookup)} sentence-level genre assignments[/green]")
        else:
            console.print(f"[yellow]⚠ Warning: all_genres.parquet not found[/yellow]")
            console.print("[yellow]Will use treebank-level metadata genres (may show multiple genres per sentence)[/yellow]")
            console.print("[yellow]Run 'label' command first to get sentence-level genre assignments[/yellow]")

            # Fallback: Load cluster statistics to get treebank-level genre information
            stats_file = clusters_dir / "cluster_statistics.json"
            if stats_file.exists():
                console.print(f"[blue]Loading cluster statistics from:[/blue] {stats_file}")
                with open(stats_file, 'r') as f:
                    cluster_stats = json.load(f)

                # Create mapping from treebank to genres (fallback only)
                treebank_genres = {}
                for key, stats in cluster_stats.items():
                    tb_code = stats['treebank']
                    split = stats['split']
                    genres = stats['genres']
                    treebank_genres[(tb_code, split)] = genres
            else:
                treebank_genres = {}

        # Filter by treebank if specified
        if treebank:
            # Parse comma-separated treebank list
            treebank_list = [tb.strip() for tb in treebank.split(",")]
            df_clusters = df_clusters[df_clusters['treebank'].isin(treebank_list)]
            if len(df_clusters) == 0:
                console.print(f"[bold red]✗ Error:[/bold red] No clusters found for treebank(s) '{treebank}'")
                raise typer.Exit(1)
            if len(treebank_list) == 1:
                console.print(f"[blue]Filtered to treebank:[/blue] {treebank_list[0]}")
            else:
                console.print(f"[blue]Filtered to {len(treebank_list)} treebanks:[/blue] {', '.join(treebank_list)}")

        # Determine embeddings directory - CLI overrides config
        if embeddings:
            emb_dir = PathLib(embeddings)
        elif cfg and cfg.embeddings.cache_dir:
            emb_dir = PathLib(cfg.embeddings.cache_dir)
            console.print(f"[blue]Using default embeddings directory from config:[/blue] {emb_dir}")
        else:
            console.print("[bold red]✗ Error:[/bold red] Must specify --embeddings or provide --config with embeddings.cache_dir")
            raise typer.Exit(1)

        console.print(f"[blue]Loading embeddings from:[/blue] {emb_dir}")

        # Determine GPU usage - CLI flag overrides config
        should_use_gpu = False
        if use_gpu is not None:
            # Explicit flag provided
            should_use_gpu = use_gpu
            if should_use_gpu:
                console.print("[blue]GPU acceleration enabled via --use-gpu flag[/blue]")
            else:
                console.print("[blue]GPU acceleration disabled via --no-gpu flag[/blue]")
        elif cfg and hasattr(cfg, 'clustering'):
            # Use config device setting
            device = cfg.clustering.device.lower()
            if device == "cuda":
                should_use_gpu = True
                console.print("[blue]GPU acceleration enabled via config (clustering.device: cuda)[/blue]")
            elif device == "auto":
                # Auto-detect GPU
                try:
                    import cupy as cp
                    cp.cuda.Device(0).compute_capability
                    should_use_gpu = True
                    console.print("[blue]GPU acceleration enabled via auto-detection (clustering.device: auto)[/blue]")
                except (ImportError, Exception):
                    should_use_gpu = False
                    console.print("[blue]GPU not available, using CPU (clustering.device: auto)[/blue]")
            else:
                should_use_gpu = False
                console.print(f"[blue]GPU acceleration disabled via config (clustering.device: {device})[/blue]")

        # Load embeddings for each treebank/split
        embeddings_list = []
        labels_list = []
        treebank_list = []
        sent_ids_list = []
        genre_list = []

        for (tb, split), group in df_clusters.groupby(['treebank', 'split']):
            emb_file = emb_dir / f"{tb}-{split}.npy"
            ids_file = emb_dir / f"{tb}-{split}_ids.txt"

            if not emb_file.exists() or not ids_file.exists():
                console.print(f"[yellow]⚠ Skipping {tb} {split}: embeddings not found[/yellow]")
                continue

            # Load embeddings and IDs
            embeddings = np.load(emb_file)
            with open(ids_file, 'r') as f:
                sent_ids = [line.strip() for line in f.readlines()]

            # Create mapping from sent_id to embedding index
            sent_id_to_idx = {sid: i for i, sid in enumerate(sent_ids)}

            # Get embeddings for sentences in this cluster group
            for _, row in group.iterrows():
                if row['sent_id'] in sent_id_to_idx:
                    idx = sent_id_to_idx[row['sent_id']]
                    sent_id = row['sent_id']

                    # Get genre for this specific sentence
                    if genre_lookup:
                        # Use sentence-level genre assignment from bootstrap labeling
                        if genres_use_sentence_key:
                            genre_str = genre_lookup.get(
                                qualify_sentence_ref(tb, split, sent_id),
                                "unlabeled",
                            )
                        else:
                            genre_str = genre_lookup.get(sent_id, "unlabeled")
                    else:
                        # Fallback: use treebank-level metadata (may be multiple genres)
                        genres = treebank_genres.get((tb, split), [])
                        genre_str = ", ".join(genres) if genres else "unknown"

                    embeddings_list.append(embeddings[idx])
                    labels_list.append(f"{tb}:{split}:c{row['cluster_id']}")
                    treebank_list.append(f"{tb}_{split}")
                    sent_ids_list.append(sent_id)
                    genre_list.append(genre_str)

        if len(embeddings_list) == 0:
            console.print("[bold red]✗ Error:[/bold red] No embeddings found for clustered sentences")
            raise typer.Exit(1)

        embeddings_array = np.array(embeddings_list)
        console.print(f"[blue]Loaded {len(embeddings_array)} embeddings[/blue]")

        # Report on genre information type
        if genre_lookup:
            n_genres = len(set(genre_list))
            console.print(f"[green]✓ Using sentence-level genre assignments ({n_genres} unique genres)[/green]")
        else:
            console.print(f"[yellow]⚠ Using treebank-level metadata (genres may not be sentence-specific)[/yellow]")

        # Perform dimensionality reduction
        console.print(f"\n[yellow]Performing {method.upper()} dimensionality reduction...[/yellow]")

        if method == "umap":
            try:
                if should_use_gpu:
                    # Try to use cuML for GPU acceleration
                    try:
                        from cuml import UMAP as cumlUMAP
                        console.print("[blue]Using GPU-accelerated UMAP (cuML)[/blue]")
                        # cuML UMAP doesn't support n_jobs parameter
                        reducer = cumlUMAP(
                            n_neighbors=15,
                            min_dist=0.1,
                            metric='cosine',
                            random_state=random_state,
                        )
                        embeddings_2d = reducer.fit_transform(embeddings_array)
                    except ImportError:
                        console.print("[yellow]⚠ cuML not available, falling back to CPU UMAP[/yellow]")
                        console.print("[blue]Install cuML for GPU support: conda install -c rapidsai -c conda-forge cuml[/blue]")
                        should_use_gpu = False

                if not should_use_gpu:
                    # CPU UMAP
                    from umap import UMAP

                    # Build UMAP parameters
                    umap_params = {
                        'n_neighbors': 15,
                        'min_dist': 0.1,
                        'metric': 'cosine',
                    }

                    # Add random_state if specified (disables parallelism)
                    if random_state is not None:
                        umap_params['random_state'] = random_state
                        console.print(f"[blue]Using random_state={random_state} (disables parallelism)[/blue]")
                    else:
                        umap_params['n_jobs'] = n_jobs
                        if n_jobs == -1:
                            console.print("[blue]Using all available CPUs for parallelization[/blue]")
                        else:
                            console.print(f"[blue]Using {n_jobs} parallel jobs[/blue]")

                    reducer = UMAP(**umap_params)
                    embeddings_2d = reducer.fit_transform(embeddings_array)

            except ImportError:
                console.print("[bold red]✗ Error:[/bold red] UMAP not installed. Install with: uv pip install umap-learn")
                raise typer.Exit(1)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            tsne_params = {'n_components': 2, 'metric': 'cosine'}
            if random_state is not None:
                tsne_params['random_state'] = random_state
            reducer = TSNE(**tsne_params)
            embeddings_2d = reducer.fit_transform(embeddings_array)
        else:
            console.print(f"[bold red]✗ Error:[/bold red] Unknown method '{method}'. Use 'umap' or 'tsne'")
            raise typer.Exit(1)

        console.print("[green]✓ Dimensionality reduction complete[/green]")

        # Create visualization
        console.print("\n[yellow]Creating visualization...[/yellow]")

        try:
            import plotly.express as px

            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'x': embeddings_2d[:, 0],
                'y': embeddings_2d[:, 1],
                'cluster': labels_list,
                'treebank_split': treebank_list,
                'sent_id': sent_ids_list,
                'genre': genre_list,
            })

            # Validate color_by parameter
            if color_by not in ['genre', 'cluster', 'treebank_split', 'treebank']:
                console.print(f"[yellow]⚠ Invalid color_by value '{color_by}', using 'genre'[/yellow]")
                color_by = 'genre'

            # Map 'treebank' to 'treebank_split' for backwards compatibility
            if color_by == 'treebank':
                color_by = 'treebank_split'

            # Prepare hover data - include all attributes except the one being used for color
            hover_cols = [col for col in ['genre', 'cluster', 'treebank_split', 'sent_id'] if col != color_by]

            # Create interactive plot
            fig = px.scatter(
                plot_df,
                x='x',
                y='y',
                color=color_by,
                hover_data=hover_cols,
                title=f'Cluster Visualization by {color_by.replace("_", " ").title()} ({method.upper()})',
                labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'},
            )

            fig.update_traces(marker=dict(size=5, opacity=0.7))
            fig.update_layout(
                width=1200,
                height=800,
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )

            # Save plot
            if output:
                output_file = PathLib(output)
            else:
                output_file = clusters_dir / "visualization.html"

            fig.write_html(str(output_file))
            console.print(f"[bold green]✓ Visualization saved to:[/bold green] {output_file}")
            console.print(f"[blue]Open in browser to view interactive plot[/blue]")

        except ImportError:
            console.print("[yellow]⚠ Plotly not installed, falling back to matplotlib[/yellow]")

            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            # Validate color_by parameter
            if color_by not in ['genre', 'cluster', 'treebank_split', 'treebank']:
                console.print(f"[yellow]⚠ Invalid color_by value '{color_by}', using 'genre'[/yellow]")
                color_by = 'genre'

            # Select the data to color by
            if color_by == 'genre':
                color_data = genre_list
            elif color_by == 'cluster':
                color_data = labels_list
            else:  # treebank or treebank_split
                color_data = treebank_list

            # Create static plot with matplotlib
            unique_values = list(set(color_data))
            colors = cm.rainbow(np.linspace(0, 1, len(unique_values)))
            value_to_color = {value: colors[i] for i, value in enumerate(unique_values)}

            plt.figure(figsize=(12, 8))
            for value in unique_values:
                mask = [v == value for v in color_data]
                plt.scatter(
                    embeddings_2d[mask, 0],
                    embeddings_2d[mask, 1],
                    c=[value_to_color[value]],
                    label=value,
                    alpha=0.6,
                    s=20
                )

            plt.xlabel(f'{method.upper()} 1')
            plt.ylabel(f'{method.upper()} 2')
            plt.title(f'Cluster Visualization by {color_by.replace("_", " ").title()} ({method.upper()})')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            # Save plot
            if output:
                output_file = PathLib(output)
            else:
                output_file = clusters_dir / "visualization.png"

            plt.savefig(str(output_file), dpi=300, bbox_inches='tight')
            console.print(f"[bold green]✓ Visualization saved to:[/bold green] {output_file}")

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Visualization failed")
        raise typer.Exit(1)


def _display_results(results: dict):
    """Display pipeline results in a formatted table."""
    table = Table(title="Pipeline Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    for key, value in results.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        else:
            table.add_row(key, str(value))

    console.print(table)


def _format_split_label(split_name: str) -> str:
    """Normalize internal split placeholders for user-facing output."""
    if split_name == "__combined__":
        return "combined"
    return split_name


def _display_cluster_stats(treebank_clusters: dict):
    """Display clustering statistics."""
    table = Table(title="Clustering Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Treebank", style="cyan")
    table.add_column("Split", style="blue")
    table.add_column("Genres", style="yellow")
    table.add_column("Clusters", style="green")

    for tb_key, info in treebank_clusters.items():
        # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
        if len(tb_key) == 3:
            tb_code, split, genre_tag = tb_key
            display_split = f"{_format_split_label(split)}:{genre_tag}"
        else:
            tb_code, split = tb_key
            display_split = _format_split_label(split)

        genres = ", ".join(info["genres"])
        n_clusters = len(info["cluster_result"]["clusters"])
        table.add_row(tb_code, display_split, genres, str(n_clusters))

    console.print(table)


def _display_schedule_summary(schedule: list):
    """Display bootstrap schedule summary."""
    from rich.table import Table

    console.print("\n[bold cyan]Bootstrap Schedule Summary[/bold cyan]")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Env", style="cyan", justify="center")
    table.add_column("Known Genres", style="green")
    table.add_column("Can Predict", style="yellow", justify="center")
    table.add_column("Still Disjunct", style="red", justify="center")
    table.add_column("New Genres", style="blue")

    prev_known = set()
    for i, env in enumerate(schedule, 1):
        known = set(env['known'])
        new_genres = known - prev_known
        prev_known = known

        known_str = ", ".join(sorted(env['known'][:5]))
        if len(env['known']) > 5:
            known_str += f" ... (+{len(env['known'])-5})"

        new_str = ", ".join(sorted(new_genres)) if new_genres else "-"

        table.add_row(
            str(i),
            known_str,
            str(len(env['predict'])),
            str(len(env['disjunct'])),
            new_str
        )

    console.print(table)

    # Show final status
    final_env = schedule[-1]
    if len(final_env['disjunct']) == 0:
        console.print("\n[bold green]✓ All genre combinations can be resolved![/bold green]")
    else:
        console.print(f"\n[bold red]✗ {len(final_env['disjunct'])} genre combinations remain unresolved[/bold red]")
        console.print("[yellow]These combinations have no overlap with single-genre treebanks:[/yellow]")
        for combo in final_env['disjunct'][:5]:
            console.print(f"  - {combo}")
        if len(final_env['disjunct']) > 5:
            console.print(f"  ... and {len(final_env['disjunct'])-5} more")


def _save_cluster_results(bootstrapper, embeddings_by_tb: dict, output_path: str):
    """Save cluster assignments and statistics to disk.

    Args:
        bootstrapper: GenreBootstrapper instance with treebank_clusters
        embeddings_by_tb: Dictionary of embeddings by (treebank, split)
        output_path: Base output path from config
    """
    from pathlib import Path as PathLib
    import json
    import pandas as pd

    output_dir = PathLib(output_path) / "clusters"
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[yellow]Saving cluster results to {output_dir}...[/yellow]")

    sent_split_lookup: Dict[Tuple[str, object], str] = {}
    for (tb_code, split_name), emb_data in embeddings_by_tb.items():
        for sent_id in emb_data.get("sent_id", []):
            sent_split_lookup[(tb_code, sent_id)] = split_name

    # Save cluster assignments as parquet
    cluster_assignments = []
    for tb_key, info in bootstrapper.treebank_clusters.items():
        # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
        if len(tb_key) == 3:
            tb_code, split, genre_tag = tb_key
        else:
            tb_code, split = tb_key

        cluster_result = info['cluster_result']

        for cluster_id, cluster_info in cluster_result['clusters'].items():
            for sent_id in cluster_info['sent_ids']:
                resolved_split = sent_split_lookup.get((tb_code, sent_id), split)
                if isinstance(sent_id, tuple) and len(sent_id) == 3:
                    ref_tb_code, ref_split, raw_sent_id = sent_id
                    export_treebank = str(ref_tb_code)
                    export_split = str(ref_split)
                    export_sent_id = str(raw_sent_id)
                else:
                    export_treebank = tb_code
                    export_split = resolved_split
                    export_sent_id = sent_id
                cluster_assignments.append({
                    'treebank': export_treebank,
                    'split': _format_split_label(export_split),
                    'sent_id': export_sent_id,
                    'cluster_id': cluster_id,
                    'confidence': cluster_info.get('confidence', None),
                })

    if cluster_assignments:
        df_assignments = pd.DataFrame(cluster_assignments)
        assignments_file = output_dir / "cluster_assignments.parquet"
        df_assignments.to_parquet(assignments_file, index=False)
        console.print(f"[green]✓ Saved cluster assignments:[/green] {assignments_file}")

    # Save cluster statistics as JSON
    cluster_stats = {}
    for tb_key, info in bootstrapper.treebank_clusters.items():
        # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
        if len(tb_key) == 3:
            tb_code, split, genre_tag = tb_key
            key = f"{tb_code}_{split}_{genre_tag}"
        else:
            tb_code, split = tb_key
            key = f"{tb_code}_{split}"

        cluster_result = info['cluster_result']

        cluster_stats[key] = {
            'treebank': tb_code,
            'split': _format_split_label(split),
            'genres': list(info['genres']),
            'n_clusters': len(cluster_result['clusters']),
            'n_sentences': sum(len(c['sent_ids']) for c in cluster_result['clusters'].values()),
            'clusters': {
                str(cid): {
                    'size': len(cinfo['sent_ids']),
                    'confidence': float(cinfo.get('confidence', 0.0)),
                }
                for cid, cinfo in cluster_result['clusters'].items()
            },
            'metrics': cluster_result.get('metrics', {}),
        }

    stats_file = output_dir / "cluster_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(cluster_stats, f, indent=2)
    console.print(f"[green]✓ Saved cluster statistics:[/green] {stats_file}")

    # Save cluster state (for label command)
    import pickle
    cluster_state = {
        'treebank_clusters': bootstrapper.treebank_clusters,
        'embeddings_by_tb': embeddings_by_tb,
    }
    state_file = output_dir / "cluster_state.pkl"
    with open(state_file, 'wb') as f:
        pickle.dump(cluster_state, f)
    console.print(f"[green]✓ Saved cluster state:[/green] {state_file}")


def _save_clustering_confusion_matrix(
    results: dict,
    output_dir: Path,
    result_label: Optional[str] = None,
) -> Optional[Path]:
    """Save clustering confusion matrix heatmap to disk.

    Returns:
        Path to saved PNG if successful, else None.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
    except ImportError as e:
        console.print(
            "[yellow]Warning: Could not save confusion matrix PNG. "
            "Install visualization dependencies with: uv pip install .[viz][/yellow]"
        )
        logger.warning(f"Failed to import plotting dependencies: {e}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    conf_matrix = np.array(results["confusion_matrix"])
    genre_labels = results["genre_labels"]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=genre_labels,
        yticklabels=genre_labels,
        cbar_kws={'label': 'Count'},
    )
    plt.xlabel('Predicted Genre')
    plt.ylabel('True Genre')

    title = f'Clustering Evaluation - {results["num_folds"]}-Fold CV (Sentence-Level)'
    if result_label:
        title += f"\nSet: {result_label}"
    plt.title(title)
    plt.tight_layout()

    if result_label:
        filename = f"confusion_matrix_clustering_{safe_label_for_filename(result_label)}.png"
    else:
        filename = "confusion_matrix_clustering.png"

    confusion_matrix_path = output_dir / filename
    plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()

    return confusion_matrix_path


def _display_evaluation_results(results: dict):
    """Display evaluation results."""
    evaluation_mode = results.get("evaluation_mode", "cross_validation")
    title = (
        "Fixed-Partition Evaluation Results"
        if evaluation_mode == "fixed_partition"
        else "Cross-Validation Results"
    )
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    console.print("=" * 60)

    console.print(
        f"\nMean Fold Acc (Micro-F1): "
        f"{results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}"
    )
    console.print(f"Overall Acc (Micro-F1): {results['overall_accuracy']:.4f}")
    console.print(f"Number of Folds: {results['num_folds']}")
    metric_specs = [
        ("macro_f1_instance", "Macro-F1 (instance-labeled)"),
        ("purity", "Purity (PUR)"),
        ("agreement_treebank", "Agreement (AGR, treebank-level)"),
        ("overlap_error_treebank", "Overlap Error (ΔBC, treebank-level)"),
        ("agreement_split", "Agreement (AGR, split-level diagnostic)"),
        ("overlap_error_split", "Overlap Error (ΔBC, split-level diagnostic)"),
    ]
    for key, label in metric_specs:
        if key not in results:
            continue
        mean_key = f"mean_{key}"
        std_key = f"std_{key}"
        if mean_key in results and std_key in results:
            console.print(
                f"{label}: {results[key]:.4f} (fold mean +/- std: "
                f"{results[mean_key]:.4f} +/- {results[std_key]:.4f})"
            )
        else:
            console.print(f"{label}: {results[key]:.4f}")
    if "instance_labeled_treebanks_treebank" in results:
        console.print(
            f"Instance-labeled Treebanks (treebank-level): "
            f"{results['instance_labeled_treebanks_treebank']}"
        )
    if "instance_labeled_treebanks_split" in results:
        console.print(
            f"Instance-labeled Treebank Splits (diagnostic): "
            f"{results['instance_labeled_treebanks_split']}"
        )
    if "evaluation_mode" in results:
        console.print(f"Evaluation Mode: {results['evaluation_mode']}")
    if "evaluation_protocol" in results:
        console.print(f"Evaluation Protocol: {results['evaluation_protocol']}")
    if "protocol_scope" in results:
        protocol_scope = results["protocol_scope"] or {}
        anchor_parts = ", ".join(protocol_scope.get("anchor_partitions", [])) or "none"
        console.print(
            "Protocol Scope: "
            f"anchors={anchor_parts} "
            f"({protocol_scope.get('requested_anchor_split_keys', 0)} split key(s)); "
            f"test={protocol_scope.get('test_partition', 'none')} "
            f"({protocol_scope.get('requested_test_split_keys', 0)} split key(s))"
        )
    if "protocol_notes" in results:
        protocol_notes = results["protocol_notes"] or []
        if protocol_notes:
            console.print("Protocol Notes:")
            for note in protocol_notes:
                console.print(f"  - {note}")
    if "protocol_deviations" in results:
        protocol_deviations = results["protocol_deviations"] or []
        if protocol_deviations:
            console.print("Protocol Deviations:")
            for deviation in protocol_deviations:
                console.print(f"  - {deviation}")
        else:
            console.print("Protocol Deviations: none")
    if "anchor_policy" in results:
        console.print(f"Anchor Policy: {results['anchor_policy']}")
    if "anchor_counts_by_genre" in results:
        anchor_counts = results["anchor_counts_by_genre"] or {}
        if anchor_counts:
            anchor_summary = ", ".join(
                f"{genre}={count}" for genre, count in sorted(anchor_counts.items())
            )
            console.print(f"Anchors by Genre: {anchor_summary}")
        else:
            console.print("Anchors by Genre: [yellow]none[/yellow]")
    if "missing_anchor_genres" in results:
        missing_anchor_genres = results["missing_anchor_genres"] or []
        if missing_anchor_genres:
            console.print(
                f"Missing Anchor Genres: [yellow]{', '.join(sorted(missing_anchor_genres))}[/yellow]"
            )
        else:
            console.print("Missing Anchor Genres: [green]none[/green]")

    # Fold accuracies
    console.print("\n[bold]Per-Fold Accuracies:[/bold]")
    for i, acc in enumerate(results['fold_accuracies'], 1):
        console.print(f"  Fold {i}: {acc:.4f}")

    # Classification report (if available)
    if "classification_report" in results:
        console.print("\n[bold]Classification Report:[/bold]")
        report = results["classification_report"]

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Genre", style="cyan")
        table.add_column("Precision", style="green")
        table.add_column("Recall", style="green")
        table.add_column("F1-Score", style="green")
        table.add_column("Support", style="yellow")

        for genre, metrics in report.items():
            if genre not in ["accuracy", "macro avg", "weighted avg"]:
                table.add_row(
                    genre,
                    f"{metrics['precision']:.3f}",
                    f"{metrics['recall']:.3f}",
                    f"{metrics['f1-score']:.3f}",
                    str(metrics['support']),
                )

        console.print(table)

    # Confusion Matrix (if available)
    if "confusion_matrix" in results and "genre_labels" in results:
        console.print("\n[bold]Confusion Matrix:[/bold]")

        conf_matrix = results["confusion_matrix"]
        genre_labels = results["genre_labels"]

        # Create Rich table
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("True \\ Predicted", style="cyan")

        # Add column for each predicted genre
        for pred_genre in genre_labels:
            table.add_column(pred_genre, justify="right", style="yellow")

        # Add row for each true genre
        for i, true_genre in enumerate(genre_labels):
            row = [true_genre]
            for j in range(len(genre_labels)):
                count = conf_matrix[i][j]
                # Highlight diagonal (correct predictions)
                if i == j:
                    row.append(f"[bold green]{count}[/bold green]")
                else:
                    row.append(str(count))
            table.add_row(*row)

        console.print(table)


def _test_treebank_genres(
    data_loader,
    genre_mapper,
    treebank_code: str,
    split: str,
    limit: int,
    show_examples: bool,
    console,
):
    """Test genre extraction for a single treebank."""
    from collections import defaultdict, Counter

    console.print(f"\n[bold yellow]Testing: {treebank_code} ({split})[/bold yellow]")

    try:
        # Load treebank data
        dataset = data_loader.load_treebank(treebank_code, split)

        # Get expected genres from metadata
        expected_genres = data_loader.get_treebank_genres(treebank_code)
        if expected_genres:
            console.print(f"[blue]Expected genres from metadata:[/blue] {', '.join(expected_genres)}")
        else:
            console.print("[yellow]⚠ No genre metadata available for this treebank[/yellow]")

        # Statistics tracking
        stats = {
            'total_sentences': 0,
            'sentences_with_genre': 0,
            'sentences_without_genre': 0,
            'genres_extracted': Counter(),
            'methods_used': Counter(),
            'pattern_matches': defaultdict(list),
            'examples': defaultdict(list),
            'no_match_examples': [],
        }

        # Process sentences
        num_sentences = limit if limit > 0 else len(dataset)
        for i, sentence in enumerate(dataset):
            if limit > 0 and i >= limit:
                break

            stats['total_sentences'] += 1

            # Extract genres
            genres = genre_mapper.extract_genres_from_metadata(sentence, treebank_code)

            if genres:
                stats['sentences_with_genre'] += 1
                for genre in genres:
                    stats['genres_extracted'][genre] += 1

                # Track which method was used
                if 'genre' in sentence:
                    stats['methods_used']['direct_field'] += 1
                elif 'comments' in sentence:
                    # Check if it was a pattern match or standard comment
                    for comment in sentence['comments']:
                        if 'newdoc genre' in comment or 'genre =' in comment:
                            stats['methods_used']['standard_comment'] += 1
                            break
                    else:
                        stats['methods_used']['pattern_match'] += 1

                # Store examples
                if show_examples and len(stats['examples'][genres[0]]) < 3:
                    example = {
                        'sent_id': sentence.get('sent_id', f'sentence_{i}'),
                        'text': sentence.get('text', ''),
                        'comments': sentence.get('comments', [])[:3],
                        'genres': genres,
                    }
                    stats['examples'][genres[0]].append(example)
            else:
                stats['sentences_without_genre'] += 1

                # Store examples of no match
                if show_examples and len(stats['no_match_examples']) < 3:
                    example = {
                        'sent_id': sentence.get('sent_id', f'sentence_{i}'),
                        'text': sentence.get('text', ''),
                        'comments': sentence.get('comments', [])[:3],
                    }
                    stats['no_match_examples'].append(example)

        # Display statistics
        _display_genre_test_results(stats, show_examples, console)

    except Exception as e:
        console.print(f"[red]✗ Failed to test {treebank_code}: {e}[/red]")
        logger.exception(f"Failed to test {treebank_code}")


def _display_genre_test_results(stats, show_examples, console):
    """Display genre extraction test results."""
    # Summary statistics
    table = Table(title="Extraction Statistics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Sentences", str(stats['total_sentences']))
    table.add_row("Sentences with Genre", str(stats['sentences_with_genre']))
    table.add_row("Sentences without Genre", str(stats['sentences_without_genre']))

    if stats['total_sentences'] > 0:
        coverage = (stats['sentences_with_genre'] / stats['total_sentences']) * 100
        table.add_row("Coverage", f"{coverage:.1f}%")

    console.print(table)

    # Genre distribution
    if stats['genres_extracted']:
        console.print("\n[bold]Extracted Genres:[/bold]")
        genre_table = Table(show_header=True, header_style="bold magenta")
        genre_table.add_column("Genre", style="cyan")
        genre_table.add_column("Count", style="green")
        genre_table.add_column("Percentage", style="yellow")

        for genre, count in stats['genres_extracted'].most_common():
            pct = (count / stats['sentences_with_genre']) * 100
            genre_table.add_row(genre, str(count), f"{pct:.1f}%")

        console.print(genre_table)

    # Methods used
    if stats['methods_used']:
        console.print("\n[bold]Extraction Methods:[/bold]")
        for method, count in stats['methods_used'].most_common():
            console.print(f"  • {method}: {count}")

    # Show examples
    if show_examples:
        # Matched examples
        if stats['examples']:
            console.print("\n[bold green]Example Matches:[/bold green]")
            for genre, examples in list(stats['examples'].items())[:3]:
                console.print(f"\n[cyan]Genre: {genre}[/cyan]")
                for ex in examples:
                    console.print(f"  sent_id: {ex['sent_id']}")
                    if ex['text']:
                        console.print(f"  text: {ex['text'][:80]}...")
                    if ex['comments']:
                        console.print(f"  comments: {ex['comments'][0]}")
                    console.print()

        # No match examples
        if stats['no_match_examples']:
            console.print("\n[bold yellow]Examples Without Genre Match:[/bold yellow]")
            for ex in stats['no_match_examples']:
                console.print(f"  sent_id: {ex['sent_id']}")
                if ex['text']:
                    console.print(f"  text: {ex['text'][:80]}...")
                if ex['comments']:
                    console.print(f"  comments: {ex['comments']}")
                else:
                    console.print(f"  comments: [none]")
                console.print()


if __name__ == "__main__":
    app()
