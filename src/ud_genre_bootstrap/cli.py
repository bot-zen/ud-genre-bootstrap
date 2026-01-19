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
            if len(treebank_filter) == 1:
                console.print(f"\n[yellow]Generating embeddings for {treebank_filter[0]}...[/yellow]")
            else:
                console.print(f"\n[yellow]Generating embeddings for {len(treebank_filter)} treebanks: {', '.join(treebank_filter)}...[/yellow]")
        else:
            console.print("\n[yellow]Generating embeddings for all treebanks...[/yellow]")

        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_filter)

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
):
    """Cluster treebank sentences into genre groups.

    Uses GMM clustering to group sentences based on embeddings.
    """
    console.print("\n[bold cyan]UD Treebank Clustering[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        bootstrapper = GenreBootstrapper(cfg)

        # Parse treebank filter (comma-separated)
        treebank_filter = None
        if treebank:
            treebank_filter = [tb.strip() for tb in treebank.split(",")]
            if len(treebank_filter) == 1:
                console.print(f"\n[yellow]Generating embeddings for {treebank_filter[0]}...[/yellow]")
            else:
                console.print(f"\n[yellow]Generating embeddings for {len(treebank_filter)} treebanks: {', '.join(treebank_filter)}...[/yellow]")
        else:
            console.print("\n[yellow]Generating embeddings for all treebanks...[/yellow]")

        embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=treebank_filter)

        # Cluster treebanks
        console.print("\n[yellow]Clustering treebanks...[/yellow]")
        bootstrapper._cluster_treebanks(embeddings_by_tb)

        console.print(f"\n[bold green]✓ Clustered {len(bootstrapper.treebank_clusters)} treebank splits[/bold green]")

        # Display cluster statistics
        _display_cluster_stats(bootstrapper.treebank_clusters)

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

        console.print("\n[yellow]Labeling clusters...[/yellow]")
        bootstrapper._label_clusters(schedule)

        console.print("\n[bold green]✓ Labeling complete![/bold green]")

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
        help="Variable to stratify on (overrides config)",
    ),
    group_by: Optional[str] = typer.Option(
        None,
        "--group-by",
        help="Variable to group by: 'treebank', 'language', or None (overrides config)",
    ),
):
    """Evaluate bootstrap quality using cross-validation.

    Performs k-fold cross-validation to assess genre prediction accuracy.
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
        )

        # Get all treebank metadata
        all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()

        # Filter by treebank if specified
        if treebank_filter:
            all_treebank_data = [
                tb for tb in all_treebank_data if tb['id'] in treebank_filter
            ]

        console.print(f"[blue]Checking {len(all_treebank_data)} treebanks...[/blue]")

        # Check sentence-level coverage for each treebank
        treebank_data = []
        for tb in all_treebank_data:
            tb_code = tb['id']

            # Load train split for coverage check (most representative)
            try:
                dataset = bootstrapper.data_loader.load_treebank(tb_code, 'train')
            except Exception:
                # Try dev or test if train doesn't exist
                try:
                    dataset = bootstrapper.data_loader.load_treebank(tb_code, 'dev')
                except Exception:
                    try:
                        dataset = bootstrapper.data_loader.load_treebank(tb_code, 'test')
                    except Exception:
                        logger.warning(f"Could not load any split for {tb_code}, skipping")
                        continue

            # Extract genres for all sentences
            total_sentences = len(dataset)
            sentences_with_genre = 0
            genre_counts = {}

            for sentence in dataset:
                genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                if genres:
                    sentences_with_genre += 1
                    for genre in genres:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

            coverage = sentences_with_genre / total_sentences if total_sentences > 0 else 0.0

            # Include if coverage meets threshold
            if coverage >= coverage_threshold_val:
                # Determine the dominant genre (for stratification)
                if genre_counts:
                    dominant_genre = max(genre_counts, key=genre_counts.get)
                    treebank_data.append({
                        'id': tb_code,
                        'genres': [dominant_genre],  # Use dominant genre for stratification
                        'language': tb['language'],
                        'coverage': coverage,
                        'sentence_count': total_sentences,
                    })

        console.print(f"[blue]Treebanks with >= {coverage_threshold_val:.0%} coverage: {len(treebank_data)}/{len(all_treebank_data)}[/blue]")

        if len(treebank_data) < n_folds_val:
            console.print(
                f"\n[bold red]✗ Error:[/bold red] Not enough treebanks with sufficient coverage "
                f"({len(treebank_data)}) for {n_folds_val}-fold cross-validation"
            )
            raise typer.Exit(1)

        # Display treebank summary
        tb_table = Table(title="Treebanks for Evaluation", show_header=True, header_style="bold magenta")
        tb_table.add_column("Treebank", style="cyan")
        tb_table.add_column("Dominant Genre", style="green")
        tb_table.add_column("Coverage", style="yellow", justify="right")
        tb_table.add_column("Sentences", style="blue", justify="right")

        for tb in sorted(treebank_data, key=lambda x: x['id']):
            tb_table.add_row(
                tb['id'],
                tb['genres'][0],
                f"{tb['coverage']:.1%}",
                str(tb['sentence_count'])
            )

        console.print()
        console.print(tb_table)

        # Create list of all treebank IDs being evaluated
        evaluated_treebank_ids = [tb['id'] for tb in treebank_data]

        # Create bootstrapper function for cross-validation
        def run_bootstrap(visible_treebanks: List[str]) -> Dict[str, str]:
            """Run bootstrap with only visible treebanks and predict hidden ones.

            Args:
                visible_treebanks: List of treebank IDs to use for training

            Returns:
                Dict mapping treebank_id to predicted genre
            """
            # Get embeddings for all treebanks being evaluated (not all treebanks in dataset)
            # We need embeddings for all evaluated treebanks for clustering, not just visible ones
            embeddings_by_tb = bootstrapper._generate_embeddings(treebank_filter=evaluated_treebank_ids)

            # Cluster all treebanks
            bootstrapper._cluster_treebanks(embeddings_by_tb)

            # Compute cluster embeddings
            bootstrapper._compute_cluster_embeddings(embeddings_by_tb)

            # Create schedule using only visible treebanks
            # Filter genre_combination_clusters to only include visible treebanks
            visible_clusters = {}
            for genre_comb, treebanks_dict in bootstrapper.genre_combination_clusters.items():
                visible_tbs = {
                    k: v for k, v in treebanks_dict.items()
                    if k[0] in visible_treebanks
                }
                if visible_tbs:
                    visible_clusters[genre_comb] = visible_tbs

            # Temporarily replace clusters
            original_clusters = bootstrapper.genre_combination_clusters
            bootstrapper.genre_combination_clusters = visible_clusters

            try:
                # Create schedule with visible treebanks only
                schedule = bootstrapper._create_schedule()

                # Label clusters
                bootstrapper._label_clusters(schedule)

                # Predict genres for all treebanks based on their sentences
                predictions = {}
                for (tb_code, split), info in bootstrapper.treebank_clusters.items():
                    # Get sentence predictions for this treebank
                    cluster_result = info['cluster_result']
                    genre_counts = {}

                    for cluster_id, cluster_info in cluster_result['clusters'].items():
                        # Get predicted genres for sentences in this cluster
                        for sent_id in cluster_info['sent_ids']:
                            if sent_id in bootstrapper.final_labels:
                                predicted_genre, _, _ = bootstrapper.final_labels[sent_id]
                                genre_counts[predicted_genre] = genre_counts.get(predicted_genre, 0) + 1

                    # Majority vote for treebank-level prediction
                    if genre_counts:
                        predicted_genre = max(genre_counts, key=genre_counts.get)
                        predictions[tb_code] = predicted_genre

                return predictions

            finally:
                # Restore original clusters
                bootstrapper.genre_combination_clusters = original_clusters

        # Run cross-validation
        console.print(f"\n[yellow]Running {n_folds}-fold cross-validation...[/yellow]")
        results = validator.k_fold_validate(treebank_data, run_bootstrap)

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
        help="Specific treebank to test (e.g., en_ewt). If not specified, tests all.",
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
        )

        # Determine which treebanks to test
        if treebank:
            treebanks_to_test = [treebank]
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
