"""Command-line interface for ud-genre-bootstrap."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from ud_genre_bootstrap.bootstrapping import GenreBootstrapper
from ud_genre_bootstrap.evaluation import CrossValidator
from ud_genre_bootstrap.utils.config import Config

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
        return load_config(config_path)
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

        # Save cluster results using shared helper function
        # Note: Need to get embeddings_by_tb from cache since fit() doesn't return it
        embeddings_by_tb = {}
        for tb_key, info in bootstrapper.treebank_clusters.items():
            # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
            if len(tb_key) == 3:
                tb_code, split, genre_tag = tb_key
                # Virtual splits use parent treebank's embeddings
            else:
                tb_code, split = tb_key

            cache_key = f"{tb_code}_{split}"
            if hasattr(bootstrapper.embedding_generator, 'embedding_cache') and cache_key in bootstrapper.embedding_generator.embedding_cache:
                embeddings_by_tb[(tb_code, split)] = bootstrapper.embedding_generator.embedding_cache[cache_key]

        _save_cluster_results(bootstrapper, embeddings_by_tb, cfg.output.genres_path)

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
            # Identify which treebanks were re-clustered (those we have new embeddings for)
            reclustered_keys = set(embeddings_by_tb.keys())

            # Restore embeddings for all non-reclustered treebanks
            # (clusters are already in bootstrapper.treebank_clusters from the loaded state)
            for key in bootstrapper.treebank_clusters.keys():
                if key not in reclustered_keys and key in existing_embeddings_by_tb:
                    embeddings_by_tb[key] = existing_embeddings_by_tb[key]

            n_kept = len(bootstrapper.treebank_clusters) - len(reclustered_keys)
            n_updated = len(reclustered_keys)
            console.print(f"[blue]Updated {n_updated} treebank(s), kept {n_kept} unchanged[/blue]")

        console.print(f"\n[bold green]✓ Clustered {len(bootstrapper.treebank_clusters)} treebank splits[/bold green]")

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

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Labeling failed")
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
):
    """Evaluate clustering + labeling on multi-genre treebanks.

    Tests the actual problem the framework solves: clustering mixed sentences
    and assigning genres. Reports sentence-level accuracy against ground truth
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
        n_folds_val = n_folds if n_folds is not None else eval_cfg.k
        group_by_val = group_by if group_by is not None else eval_cfg.group_by

        # Parse treebank filter (comma-separated)
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
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

        console.print(f"[blue]CV settings:[/blue] {n_folds_val}-fold, group_by={group_by_val}")
        console.print(f"[blue]Min confidence:[/blue] {cfg.bootstrapping.min_confidence}")
        console.print(f"[blue]Min margin:[/blue] {cfg.bootstrapping.min_margin}")

        # Initialize clustering evaluator
        from ud_genre_bootstrap.evaluation.validator import ClusteringEvaluator

        evaluator = ClusteringEvaluator(
            n_folds=n_folds_val,
            group_by=group_by_val,
            min_confidence=cfg.bootstrapping.min_confidence,
            min_margin=cfg.bootstrapping.min_margin,
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

        # Filter by treebank if specified
        if treebank_filter:
            all_treebank_data = [
                tb for tb in all_treebank_data if tb['id'] in treebank_filter
            ]

        console.print(f"[blue]Checking {len(all_treebank_data)} treebanks for multi-genre datasets...[/blue]")

        # Identify multi-genre treebanks and collect sentence metadata
        multi_genre_treebanks = []
        sentence_metadata = {}  # Maps (tb_code, split, sent_id) -> genre

        # Track statistics
        stats = {
            'checked': 0,
            'single_genre': 0,
            'multi_genre': 0,
            'load_errors': 0,
            'no_metadata': 0,
        }

        for tb in all_treebank_data:
            tb_code = tb['id']
            available_splits = bootstrapper.data_loader.get_available_splits(tb_code)

            if not available_splits:
                continue

            # Check each split for multi-genre content
            for split_name in available_splits:
                stats['checked'] += 1

                try:
                    dataset = bootstrapper.data_loader.load_treebank(tb_code, split_name)
                except Exception as e:
                    logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                    stats['load_errors'] += 1
                    continue

                # Count genres in this split
                genre_counts = {}
                sent_count = 0

                for idx, sentence in enumerate(dataset):
                    sent_id = sentence.get('sent_id', f'{tb_code}_{split_name}_{idx}')
                    genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)

                    if genres:
                        primary_genre = genres[0]
                        genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1
                        sentence_metadata[(tb_code, split_name, sent_id)] = primary_genre
                        sent_count += 1

                # Classify this split
                unique_genres = list(genre_counts.keys())
                if len(unique_genres) == 0:
                    stats['no_metadata'] += 1
                elif len(unique_genres) == 1:
                    stats['single_genre'] += 1
                elif len(unique_genres) >= 2:
                    stats['multi_genre'] += 1
                    multi_genre_treebanks.append({
                        'treebank': tb_code,
                        'split': split_name,
                        'genres': unique_genres,
                        'language': tb['language'],
                        'sentence_count': sent_count,
                        'genre_counts': genre_counts,
                    })

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

        if len(multi_genre_treebanks) < n_folds_val:
            console.print(
                f"\n[bold yellow]⚠ Insufficient data for {n_folds_val}-fold cross-validation[/bold yellow]"
            )
            console.print(f"[dim]Found {len(multi_genre_treebanks)} multi-genre treebanks, need at least {n_folds_val}[/dim]")

            # Display summary
            console.print("\n[bold]Summary:[/bold]")
            console.print(f"  • Checked: {stats['checked']} treebank splits")
            console.print(f"  • Single-genre: {stats['single_genre']} (skipped)")
            console.print(f"  • Multi-genre: {stats['multi_genre']} (found, but need {n_folds_val} for {n_folds_val}-fold CV)")
            if stats['no_metadata'] > 0:
                console.print(f"  • No sentence metadata: {stats['no_metadata']}")
            if stats['load_errors'] > 0:
                console.print(f"  • Load errors: {stats['load_errors']}")

            console.print("\n[dim]Suggestions:[/dim]")
            console.print(f"[dim]  • Reduce n-folds: --n-folds {len(multi_genre_treebanks)}[/dim]")
            console.print("[dim]  • Include more treebanks in the config[/dim]")
            console.print("[dim]  • Remove treebank filter to evaluate all available treebanks[/dim]")
            console.print("\n[blue]Evaluation skipped - insufficient data for cross-validation[/blue]")
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

        # Generate embeddings for all multi-genre treebanks
        console.print(f"\n[yellow]Generating/loading embeddings...[/yellow]")
        treebank_ids_to_embed = list(set(tb['treebank'] for tb in multi_genre_treebanks))
        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_ids_to_embed)

        # Run clustering evaluation
        console.print(f"\n[yellow]Running {n_folds_val}-fold clustering evaluation...[/yellow]")
        results = evaluator.k_fold_validate(
            multi_genre_treebanks=multi_genre_treebanks,
            sentence_metadata=sentence_metadata,
            embeddings_by_tb=embeddings_by_tb,
            clusterer=bootstrapper.clusterer,
        )


        # Save confusion matrix as PNG if output path specified
        if cfg.output.genres_path and "confusion_matrix" in results and "genre_labels" in results:
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                import numpy as np

                output_dir = Path(cfg.output.genres_path) / "evaluation"
                output_dir.mkdir(parents=True, exist_ok=True)

                conf_matrix = np.array(results["confusion_matrix"])
                genre_labels = results["genre_labels"]

                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    conf_matrix,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    xticklabels=genre_labels,
                    yticklabels=genre_labels,
                    cbar_kws={'label': 'Count'}
                )
                plt.xlabel('Predicted Genre')
                plt.ylabel('True Genre')
                plt.title(f'Clustering Evaluation - {results["num_folds"]}-Fold CV (Sentence-Level)')
                plt.tight_layout()

                # Save
                confusion_matrix_path = output_dir / "confusion_matrix_clustering.png"
                plt.savefig(confusion_matrix_path, dpi=150, bbox_inches='tight')
                plt.close()

                console.print(f"[blue]Confusion matrix saved to:[/blue] {confusion_matrix_path}")
            except ImportError as e:
                console.print(f"[yellow]Warning: Could not save confusion matrix PNG. Install visualization dependencies with: uv pip install .[viz][/yellow]")
                logger.warning(f"Failed to save confusion matrix: {e}")

        # Display results
        _display_evaluation_results(results)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Evaluation failed")
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

        # Apply treebank filter if specified
        if treebank_filter:
            # Extract treebank codes from sent_id (format: tb_code:split:sent_id or similar)
            # We need to match against treebank codes in the sent_id
            df_bootstrap_only['treebank'] = df_bootstrap_only['sent_id'].str.extract(r'^([a-z]{2,3}_[a-z]+)', expand=False)
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
        )

        # Build sent_id -> text mapping
        sent_id_to_text = {}
        sent_id_to_treebank = {}

        for tb_code, split, dataset in data_loader.iter_all_treebanks(treebank_filter=treebank_filter):
            for sentence in dataset:
                sent_id = sentence.get('sent_id')
                text = sentence.get('text', '')
                if sent_id and text:
                    sent_id_to_text[sent_id] = text
                    sent_id_to_treebank[sent_id] = (tb_code, split)

        # Filter df to only sentences we have text for
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

        sent_id_to_genre = {}
        if genre_labels_file:
            console.print(f"[blue]Loading sentence-level genre assignments from:[/blue] {genre_labels_file}")
            df_genres = pd.read_parquet(genre_labels_file)
            # Create mapping from sent_id to genre
            sent_id_to_genre = dict(zip(df_genres['sent_id'], df_genres['genre']))
            console.print(f"[green]✓ Loaded {len(sent_id_to_genre)} sentence-level genre assignments[/green]")
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
                    if sent_id_to_genre:
                        # Use sentence-level genre assignment from bootstrap labeling
                        genre_str = sent_id_to_genre.get(sent_id, "unlabeled")
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
        if sent_id_to_genre:
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
            display_split = f"{split}:{genre_tag}"
        else:
            tb_code, split = tb_key
            display_split = split

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
                cluster_assignments.append({
                    'treebank': tb_code,
                    'split': split,
                    'sent_id': sent_id,
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
            'split': split,
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


def _display_evaluation_results(results: dict):
    """Display cross-validation results."""
    console.print("\n[bold cyan]Cross-Validation Results[/bold cyan]")
    console.print("=" * 60)

    console.print(f"\nMean Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    console.print(f"Overall Accuracy: {results['overall_accuracy']:.4f}")
    console.print(f"Number of Folds: {results['num_folds']}")

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
