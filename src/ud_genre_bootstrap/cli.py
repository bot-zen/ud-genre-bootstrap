"""Command-line interface for ud-genre-bootstrap."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
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

        # Run pipeline
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running bootstrap pipeline...", total=None)
            results = bootstrapper.fit()

        # Display results
        console.print("\n[bold green]✓ Pipeline complete![/bold green]")
        _display_results(results)

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
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]

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
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu",
        help="Use GPU acceleration for GMM clustering (requires cuML)",
    ),
):
    """Cluster treebank sentences into genre groups.

    Uses GMM clustering to group sentences based on embeddings.
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
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]

        # Apply config exclusions
        treebank_filter = apply_treebank_exclusions(cfg, bootstrapper.data_loader, treebank_filter)

        if treebank_filter:
            if len(treebank_filter) == 1:
                console.print(f"\n[yellow]Clustering {treebank_filter[0]}...[/yellow]")
            else:
                console.print(f"\n[yellow]Clustering {len(treebank_filter)} treebanks: {', '.join(treebank_filter[:5])}{'...' if len(treebank_filter) > 5 else ''}[/yellow]")
        else:
            console.print("\n[yellow]Clustering all treebanks...[/yellow]")

        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_filter)

        # Cluster treebanks
        console.print("\n[yellow]Clustering treebanks...[/yellow]")
        bootstrapper._cluster_treebanks(embeddings_by_tb)

        console.print(f"\n[bold green]✓ Clustered {len(bootstrapper.treebank_clusters)} treebank splits[/bold green]")

        # Display cluster statistics
        _display_cluster_stats(bootstrapper.treebank_clusters)

        # Save cluster results
        from pathlib import Path as PathLib
        import json
        import pandas as pd

        output_dir = PathLib(cfg.output.genres_path) / "clusters"
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"\n[yellow]Saving cluster results to {output_dir}...[/yellow]")

        # Save cluster assignments as parquet
        cluster_assignments = []
        for (tb_code, split), info in bootstrapper.treebank_clusters.items():
            cluster_result = info['cluster_result']
            emb_data = embeddings_by_tb[(tb_code, split)]

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
        for (tb_code, split), info in bootstrapper.treebank_clusters.items():
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

        # Save embeddings for visualization (optional, can be large)
        console.print(f"[blue]Embeddings already cached at:[/blue] {cfg.embeddings.cache_dir if cfg.embeddings.cache_dir else 'not configured'}")

        console.print(f"\n[bold green]✓ Cluster results saved to {output_dir}[/bold green]")
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
):
    """Label clusters using bootstrap algorithm.

    Applies the bootstrapping schedule to assign genre labels to clusters.
    """
    console.print("\n[bold cyan]Bootstrap Genre Labeling[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        bootstrapper = GenreBootstrapper(cfg)

        # Run through clustering
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

        console.print("\n[yellow]Labeling clusters...[/yellow]")
        bootstrapper._label_clusters(schedule)

        # Generate cross-lingual report
        console.print("\n[yellow]Generating cross-lingual assignment report...[/yellow]")
        bootstrapper._generate_cross_lingual_report()

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
    coverage_threshold: Optional[float] = typer.Option(
        None,
        "--coverage-threshold",
        help="Minimum sentence-level metadata coverage to include treebank (overrides config)",
    ),
    stratify: Optional[str] = typer.Option(
        None,
        "--stratify",
        help="Variable to stratify on - ensures each fold has similar distribution of this variable (default: 'genre', overrides config)",
    ),
    group_by: Optional[str] = typer.Option(
        None,
        "--group-by",
        help="Variable to group by: 'treebank', 'language', or None (overrides config)",
    ),
):
    """Evaluate bootstrap quality using cross-validation.

    Performs k-fold cross-validation to assess genre prediction accuracy.

    Uses ALL available splits (train, dev, test) from all treebanks with
    sentence-level genre metadata. Since we only use text content and genre
    labels (not UD annotations), there's no restriction on using test splits.

    Creates virtual single-genre splits from each treebank split with sufficient
    sentences per genre, then performs k-fold CV over these virtual splits.
    """
    console.print("\n[bold cyan]Bootstrap Evaluation (Cross-Validation)[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Get evaluation config values (CLI overrides config)
        eval_cfg = cfg.evaluation.metadata_validation
        n_folds_val = n_folds if n_folds is not None else eval_cfg.k
        coverage_threshold_val = coverage_threshold if coverage_threshold is not None else eval_cfg.coverage_threshold
        stratify_val = stratify if stratify is not None else eval_cfg.stratify_by
        group_by_val = group_by if group_by is not None else eval_cfg.group_by

        # Parse treebank filter (comma-separated)
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
            if len(treebank_filter) == 1:
                console.print(f"[blue]Evaluating:[/blue] {treebank_filter[0]}")
            else:
                console.print(f"[blue]Evaluating {len(treebank_filter)} treebanks:[/blue] {', '.join(treebank_filter)}")
        else:
            console.print("[blue]Evaluating all treebanks[/blue]")

        console.print(f"[blue]CV settings:[/blue] {n_folds_val}-fold, group_by={group_by_val}, coverage>={coverage_threshold_val:.0%}")

        # Initialize validator
        validator = CrossValidator(
            n_folds=n_folds_val,
            stratify_by=stratify_val,
            group_by=group_by_val,
        )

        # Load treebank metadata and genre mapper
        console.print("\n[yellow]Loading treebank metadata and checking coverage...[/yellow]")
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
        )

        # Get all treebank metadata
        all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()

        # Apply exclusions from config
        if cfg.exclude_treebanks:
            excluded_count = len([tb for tb in all_treebank_data if tb['id'] in cfg.exclude_treebanks])
            all_treebank_data = [
                tb for tb in all_treebank_data if tb['id'] not in cfg.exclude_treebanks
            ]
            if excluded_count > 0:
                console.print(f"[yellow]Excluded {excluded_count} treebanks from config: {', '.join(cfg.exclude_treebanks)}[/yellow]")

        # Filter by treebank if specified
        if treebank_filter:
            all_treebank_data = [
                tb for tb in all_treebank_data if tb['id'] in treebank_filter
            ]

        console.print(f"[blue]Checking {len(all_treebank_data)} treebanks...[/blue]")

        # Get minimum sentences threshold
        min_genre_sentences = eval_cfg.min_genre_sentences

        # Check sentence-level coverage and create virtual single-genre splits
        # A treebank can contribute multiple virtual splits (one per genre)
        virtual_splits = []  # List of dicts with virtual split info
        virtual_split_sentence_ids = {}  # Maps (tb_code, split_name, genre) -> list of sent_ids

        for tb in all_treebank_data:
            tb_code = tb['id']

            # Get all available splits for this treebank
            available_splits = bootstrapper.data_loader.get_available_splits(tb_code)

            if not available_splits:
                logger.warning(f"No splits found for {tb_code}, skipping")
                continue

            # Use all available splits (train, dev, test) - no restrictions
            # We're only using text content and genre metadata, not UD annotations
            splits_to_check = available_splits

            # Process each available split to create virtual splits
            for split_name in splits_to_check:
                try:
                    dataset = bootstrapper.data_loader.load_treebank(tb_code, split_name)
                except Exception as e:
                    logger.warning(f"Could not load {tb_code}:{split_name}: {e}")
                    continue

                # Extract genres for all sentences and group by genre
                total_sentences = len(dataset)
                genre_sentences = {}  # genre -> list of sent_ids

                for idx, sentence in enumerate(dataset):
                    sent_id = sentence.get('sent_id', f'{tb_code}_{split_name}_{idx}')
                    genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)

                    # A sentence can have multiple genres, but we'll use the first one for simplicity
                    # (most sentences should have exactly one genre after extraction)
                    if genres:
                        primary_genre = genres[0]
                        if primary_genre not in genre_sentences:
                            genre_sentences[primary_genre] = []
                        genre_sentences[primary_genre].append(sent_id)

                # Create virtual splits for each genre with sufficient sentences
                for genre, sent_ids in genre_sentences.items():
                    num_sentences = len(sent_ids)

                    if num_sentences >= min_genre_sentences:
                        # Calculate coverage for this genre in this split
                        coverage = num_sentences / total_sentences

                        # Create virtual split ID that includes the split type
                        virtual_split_id = f"{tb_code}:{split_name}:{genre}"

                        virtual_splits.append({
                            'id': virtual_split_id,
                            'treebank': tb_code,
                            'split': split_name,
                            'genre': genre,
                            'genres': [genre],  # Single genre for this virtual split
                            'language': tb['language'],
                            'coverage': coverage,
                            'sentence_count': num_sentences,
                        })

                        # Store sentence IDs for this virtual split
                        virtual_split_sentence_ids[(tb_code, split_name, genre)] = sent_ids

        console.print(f"[blue]Created {len(virtual_splits)} virtual single-genre splits from {len(all_treebank_data)} treebanks[/blue]")
        console.print(f"[blue]Minimum {min_genre_sentences} sentences per genre required[/blue]")

        if len(virtual_splits) < n_folds_val:
            console.print(
                f"\n[bold red]✗ Error:[/bold red] Not enough virtual splits "
                f"({len(virtual_splits)}) for {n_folds_val}-fold cross-validation"
            )
            raise typer.Exit(1)

        # Count splits by type
        split_counts = {}
        for s in virtual_splits:
            split_type = s['split']
            split_counts[split_type] = split_counts.get(split_type, 0) + 1

        console.print(f"[blue]  Split distribution: {dict(sorted(split_counts.items()))}[/blue]")

        # Display virtual split summary
        tb_table = Table(title="Virtual Single-Genre Splits for Evaluation", show_header=True, header_style="bold magenta")
        tb_table.add_column("Virtual Split", style="cyan")
        tb_table.add_column("Split", style="blue")
        tb_table.add_column("Genre", style="green")
        tb_table.add_column("Coverage", style="yellow", justify="right")
        tb_table.add_column("Sentences", style="magenta", justify="right")

        for split in sorted(virtual_splits, key=lambda x: x['id']):
            tb_table.add_row(
                f"{split['treebank']}:{split['genre']}",
                split['split'],
                split['genre'],
                f"{split['coverage']:.1%}",
                str(split['sentence_count'])
            )

        console.print()
        console.print(tb_table)

        # Create list of all unique treebank IDs being evaluated (for embedding generation)
        evaluated_treebank_ids = list(set(split['treebank'] for split in virtual_splits))

        # Create bootstrapper function for cross-validation
        def run_bootstrap(visible_split_ids: List[str]) -> Dict[str, str]:
            """Run bootstrap with only visible virtual splits and predict hidden ones.

            Args:
                visible_split_ids: List of virtual split IDs to use for training (e.g., ['en_ewt:blog', 'de_pud:news'])

            Returns:
                Dict mapping virtual_split_id to predicted genre
            """
            # Get embeddings for all treebanks being evaluated
            embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=evaluated_treebank_ids)

            # Filter embeddings to create virtual split embeddings
            # For each virtual split, we need to filter the embeddings to only include sentences from that genre
            virtual_embeddings_by_tb = {}
            virtual_treebank_clusters = {}

            for split_info in virtual_splits:
                tb_code = split_info['treebank']
                split_name = split_info['split']
                genre = split_info['genre']
                virtual_split_id = split_info['id']

                # Get the sentence IDs for this virtual split
                allowed_sent_ids = set(virtual_split_sentence_ids[(tb_code, split_name, genre)])

                # Filter embeddings for this treebank/split
                if (tb_code, split_name) in embeddings_by_tb:
                    emb_data = embeddings_by_tb[(tb_code, split_name)]

                    # Find indices of sentences that belong to this virtual split
                    indices = [
                        i for i, sent_id in enumerate(emb_data['sent_id'])
                        if sent_id in allowed_sent_ids
                    ]

                    if indices:
                        # Create filtered embedding data
                        import numpy as np
                        filtered_embeddings = emb_data['embedding'][indices]
                        filtered_sent_ids = [emb_data['sent_id'][i] for i in indices]

                        # Store with virtual split key
                        virtual_key = (virtual_split_id, split_name)
                        virtual_embeddings_by_tb[virtual_key] = {
                            'embedding': filtered_embeddings,
                            'sent_id': filtered_sent_ids,
                        }

                        # Cluster this virtual split (it's single-genre, so n_genres=1)
                        cluster_result = bootstrapper.clusterer.cluster_treebank(
                            embeddings=filtered_embeddings,
                            sent_ids=filtered_sent_ids,
                            n_genres=1,  # Virtual splits are single-genre
                        )

                        virtual_treebank_clusters[virtual_key] = {
                            'genres': [genre],
                            'cluster_result': cluster_result,
                        }

            # Build genre_combination_clusters using virtual splits
            from collections import defaultdict
            genre_combination_clusters = defaultdict(dict)

            for virtual_key, cluster_info in virtual_treebank_clusters.items():
                genres = cluster_info['genres']
                genre_combination = tuple(sorted(genres))

                # Compute cluster embeddings
                emb_data = virtual_embeddings_by_tb[virtual_key]
                cluster_result = cluster_info['cluster_result']

                cluster_list = []
                for cluster_id, cluster_info_inner in cluster_result['clusters'].items():
                    sent_ids = cluster_info_inner['sent_ids']
                    indices = [
                        i for i, sid in enumerate(emb_data['sent_id'])
                        if sid in sent_ids
                    ]

                    if indices:
                        import numpy as np
                        cluster_emb = np.mean(emb_data['embedding'][indices], axis=0)
                        cluster_list.append({
                            'cluster_id': cluster_id,
                            'sent_ids': sent_ids,
                            'embedding': cluster_emb,
                            'confidence': cluster_info_inner.get('confidence', 1.0),
                        })

                genre_combination_clusters[genre_combination][virtual_key] = cluster_list

            # Filter to only visible splits
            visible_clusters = {}
            for genre_comb, splits_dict in genre_combination_clusters.items():
                visible_split_data = {
                    k: v for k, v in splits_dict.items()
                    if k[0] in visible_split_ids
                }
                if visible_split_data:
                    visible_clusters[genre_comb] = visible_split_data

            # Temporarily override bootstrapper state
            original_clusters = bootstrapper.genre_combination_clusters
            original_treebank_clusters = bootstrapper.treebank_clusters
            bootstrapper.genre_combination_clusters = visible_clusters
            bootstrapper.treebank_clusters = virtual_treebank_clusters

            try:
                # Create schedule with visible splits only
                schedule = bootstrapper._create_schedule()

                # Label clusters (only labels multi-genre combinations if any)
                bootstrapper._label_clusters(schedule)

                # Compute mean embeddings for known genres from visible splits
                from scipy.spatial import distance
                known_genre_embeddings = {}

                for genre_comb, splits_dict in visible_clusters.items():
                    if len(genre_comb) == 1:  # Single-genre
                        genre = genre_comb[0]
                        # Collect all cluster embeddings for this genre from visible splits
                        all_embeddings = []
                        for cluster_list in splits_dict.values():
                            for cluster in cluster_list:
                                all_embeddings.append(cluster['embedding'])

                        if all_embeddings:
                            import numpy as np
                            known_genre_embeddings[genre] = np.mean(all_embeddings, axis=0)

                # Debug: Check what's visible vs hidden
                num_visible = len(visible_split_ids)
                num_total = len(virtual_treebank_clusters)
                num_hidden = num_total - num_visible
                console.print(f"[blue]  Fold stats: {num_visible} visible (train), {num_hidden} hidden (test) splits[/blue]")
                console.print(f"[blue]  Known genres from training: {list(known_genre_embeddings.keys())}[/blue]")

                # Predict genres for all virtual splits
                predictions = {}
                num_correct = 0
                num_incorrect = 0
                all_similarities = []  # Track all similarity scores for diagnostics

                for virtual_key, info in virtual_treebank_clusters.items():
                    virtual_split_id = virtual_key[0]
                    true_genre = info['genres'][0]

                    # If this is a visible split, use its ground truth genre
                    if virtual_split_id in visible_split_ids:
                        predictions[virtual_split_id] = true_genre
                        continue

                    # For hidden splits, predict based on cluster embedding similarity
                    # WITHOUT using ground truth genre information
                    if virtual_key in virtual_embeddings_by_tb:
                        # Get embeddings for this hidden split
                        emb_data = virtual_embeddings_by_tb[virtual_key]

                        # Compute mean embedding for all sentences in this split
                        import numpy as np
                        split_mean_embedding = np.mean(emb_data['embedding'], axis=0)

                        # Find most similar known genre
                        best_genre = None
                        best_similarity = -1
                        similarities = {}

                        for genre, genre_emb in known_genre_embeddings.items():
                            similarity = 1 - distance.cosine(split_mean_embedding, genre_emb)
                            similarities[genre] = similarity
                            if similarity > best_similarity:
                                best_similarity = similarity
                                best_genre = genre

                        if best_genre:
                            predictions[virtual_split_id] = best_genre

                            # Track accuracy and similarity margins
                            if best_genre == true_genre:
                                num_correct += 1
                                correct_sim = similarities[best_genre]
                                # Calculate margin (difference from second-best)
                                sorted_sims = sorted(similarities.values(), reverse=True)
                                margin = sorted_sims[0] - sorted_sims[1] if len(sorted_sims) > 1 else 1.0
                                all_similarities.append((correct_sim, margin, True))
                            else:
                                num_incorrect += 1
                                # Show first few errors
                                if num_incorrect <= 3:
                                    console.print(f"[yellow]  Misclassified: {virtual_split_id} predicted as {best_genre} (true: {true_genre}), sims: {similarities}[/yellow]")

                # Show statistics
                console.print(f"[blue]  Test predictions: {num_correct} correct, {num_incorrect} incorrect out of {num_hidden} hidden[/blue]")

                if all_similarities:
                    import numpy as np
                    avg_sim = np.mean([s[0] for s in all_similarities])
                    avg_margin = np.mean([s[1] for s in all_similarities])
                    min_margin = np.min([s[1] for s in all_similarities])
                    console.print(f"[blue]  Avg similarity: {avg_sim:.3f}, Avg margin: {avg_margin:.3f}, Min margin: {min_margin:.3f}[/blue]")

                return predictions

            finally:
                # Restore original clusters
                bootstrapper.genre_combination_clusters = original_clusters
                bootstrapper.treebank_clusters = original_treebank_clusters

        # Run cross-validation
        console.print(f"\n[yellow]Running {n_folds_val}-fold cross-validation...[/yellow]")
        results = validator.k_fold_validate(virtual_splits, run_bootstrap)

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
                plt.title(f'Confusion Matrix - {results["num_folds"]}-Fold Cross-Validation')
                plt.tight_layout()

                # Save
                confusion_matrix_path = output_dir / "confusion_matrix.png"
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
        )

        # Determine which treebanks to test
        if treebank:
            # Parse comma-separated treebank list
            treebanks_to_test = [tb.strip() for tb in treebank.split(",")]
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
        table.add_row("  Max Iterations", str(cfg.bootstrapping.max_iterations))
        table.add_row("  Fail on Incomplete", str(cfg.bootstrapping.fail_on_incomplete))
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
    clusters: Path = typer.Option(
        ...,
        "--clusters",
        "-c",
        help="Path to cluster output directory (contains cluster_assignments.parquet)",
        exists=True,
        dir_okay=True,
    ),
    embeddings: Optional[Path] = typer.Option(
        None,
        "--embeddings",
        "-e",
        help="Path to embeddings cache directory. If not specified, uses config.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        help="Path to configuration YAML file",
        exists=True,
        dir_okay=False,
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
    use_gpu: bool = typer.Option(
        False,
        "--use-gpu/--no-gpu",
        help="Use GPU acceleration if available (requires cuML for UMAP)",
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
    """
    console.print("\n[bold cyan]Cluster Visualization[/bold cyan]")
    console.print("=" * 60)

    try:
        import pandas as pd
        import numpy as np
        import json
        from pathlib import Path as PathLib

        # Load cluster assignments
        assignments_file = PathLib(clusters) / "cluster_assignments.parquet"
        if not assignments_file.exists():
            console.print(f"[bold red]✗ Error:[/bold red] Cluster assignments not found at {assignments_file}")
            raise typer.Exit(1)

        console.print(f"[blue]Loading cluster assignments from:[/blue] {assignments_file}")
        df_clusters = pd.read_parquet(assignments_file)

        # Try to load sentence-level genre assignments from bootstrap labeling
        # Check both in the clusters directory and in the parent output directory
        genre_labels_file = None
        possible_paths = [
            PathLib(clusters).parent / "all_genres.parquet",  # Parent of clusters dir
            PathLib(clusters) / "all_genres.parquet",  # In clusters dir itself
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
            stats_file = PathLib(clusters) / "cluster_statistics.json"
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

        # Determine embeddings directory
        if embeddings:
            emb_dir = PathLib(embeddings)
        elif config:
            cfg = load_config_from_path(config)
            if cfg.embeddings.cache_dir:
                emb_dir = PathLib(cfg.embeddings.cache_dir)
            else:
                console.print("[bold red]✗ Error:[/bold red] No embeddings cache directory configured")
                raise typer.Exit(1)
        else:
            console.print("[bold red]✗ Error:[/bold red] Must specify --embeddings or --config")
            raise typer.Exit(1)

        console.print(f"[blue]Loading embeddings from:[/blue] {emb_dir}")

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
                if use_gpu:
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
                        use_gpu = False

                if not use_gpu:
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
                output_file = PathLib(clusters) / "visualization.html"

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
                output_file = PathLib(clusters) / "visualization.png"

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

    for (tb_code, split), info in treebank_clusters.items():
        genres = ", ".join(info["genres"])
        n_clusters = len(info["cluster_result"]["clusters"])
        table.add_row(tb_code, split, genres, str(n_clusters))

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
