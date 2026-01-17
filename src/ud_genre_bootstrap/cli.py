"""Command-line interface for ud-genre-bootstrap."""

import logging
from pathlib import Path
from typing import Optional

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
        help="Specific treebank to embed (e.g., en_ewt). If not specified, embeds all.",
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

        # Generate embeddings
        console.print("\n[yellow]Generating embeddings...[/yellow]")
        embeddings_by_tb = bootstrapper._generate_embeddings()

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
        help="Specific treebank to cluster",
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

        # Generate embeddings first
        console.print("\n[yellow]Generating embeddings...[/yellow]")
        embeddings_by_tb = bootstrapper._generate_embeddings()

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
        console.print("\n[yellow]Generating embeddings...[/yellow]")
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
    n_folds: int = typer.Option(
        5,
        "--n-folds",
        "-k",
        help="Number of folds for cross-validation",
    ),
    stratify: str = typer.Option(
        "genre",
        "--stratify",
        help="Variable to stratify on",
    ),
    group_by: Optional[str] = typer.Option(
        "language",
        "--group-by",
        help="Variable to group by (e.g., language)",
    ),
):
    """Evaluate bootstrap quality using cross-validation.

    Performs k-fold cross-validation to assess genre prediction accuracy.
    """
    console.print("\n[bold cyan]Bootstrap Evaluation (Cross-Validation)[/bold cyan]")
    console.print("=" * 60)

    try:
        cfg = load_config_from_path(config)

        # Initialize validator
        validator = CrossValidator(
            n_folds=n_folds,
            stratify_by=stratify,
            group_by=group_by,
        )

        # Load treebank metadata
        bootstrapper = GenreBootstrapper(cfg)

        # TODO: Need to extract treebank metadata with genres
        console.print("\n[yellow]Loading treebank metadata...[/yellow]")
        console.print("[red]Evaluation not yet fully implemented[/red]")

        # This would look like:
        # treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()
        #
        # def run_bootstrap(visible_treebanks):
        #     # Run bootstrap with only visible treebanks
        #     # Return predictions for hidden treebanks
        #     pass
        #
        # results = validator.k_fold_validate(treebank_data, run_bootstrap)
        # _display_evaluation_results(results)

    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        logger.exception("Evaluation failed")
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


if __name__ == "__main__":
    app()
