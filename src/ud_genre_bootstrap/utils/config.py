"""Configuration management for UD Genre Bootstrap."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class EmbeddingsConfig:
    """Configuration for embeddings generation."""

    model: str = "xlm-roberta-base"
    pooling: str = "mean"
    batch_size: int = 64
    layer: int = -1
    device: str = "auto"
    cache_dir: Optional[str] = None  # Directory to cache embeddings


@dataclass
class ClusteringConfig:
    """Configuration for clustering."""

    method: str = "gmm"
    level: str = "treebank"
    seed: int = 42


@dataclass
class BootstrappingConfig:
    """Configuration for bootstrapping."""

    min_confidence: float = 0.8
    max_iterations: int = 10
    fail_on_incomplete: bool = False
    unresolved_handling: str = "null"


@dataclass
class GenreExtractionConfig:
    """Configuration for genre extraction."""

    mapping_path: Optional[str] = None
    patterns_path: Optional[Union[str, List[str]]] = None


@dataclass
class MetadataValidationConfig:
    """Configuration for metadata validation."""

    method: str = "kfold"
    k: int = 5
    stratify_by: str = "genre"
    group_by: str = "language"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    enabled: bool = True
    metadata_validation: MetadataValidationConfig = field(
        default_factory=MetadataValidationConfig
    )
    cluster_metrics: List[str] = field(
        default_factory=lambda: ["silhouette", "calinski_harabasz", "davies_bouldin"]
    )
    convergence_metrics: List[str] = field(
        default_factory=lambda: [
            "resolution_rate",
            "disjunct_combinations",
            "confidence_distribution",
        ]
    )


@dataclass
class OutputConfig:
    """Configuration for output."""

    genres_path: str = "output/ud-v2.17/genres/"
    embeddings_hf_repo: str = "commul/ud-embeddings-xlm-roberta-base"
    embeddings_revision: str = "2.17"
    genres_hf_repo: str = "commul/ud-genres"
    genres_revision: str = "2.17"
    push_to_hub: bool = False
    hf_token: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration for UD Genre Bootstrap."""

    ud_version: str = "2.17"
    ud_source: str = "hf://commul/universal_dependencies"
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    bootstrapping: BootstrappingConfig = field(default_factory=BootstrappingConfig)
    genre_extraction: GenreExtractionConfig = field(default_factory=GenreExtractionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Parse nested configs
        embeddings = EmbeddingsConfig(**config_dict.get("embeddings", {}))
        clustering = ClusteringConfig(**config_dict.get("clustering", {}))
        bootstrapping = BootstrappingConfig(**config_dict.get("bootstrapping", {}))
        genre_extraction = GenreExtractionConfig(**config_dict.get("genre_extraction", {}))

        # Parse evaluation config
        eval_dict = config_dict.get("evaluation", {})
        metadata_val = MetadataValidationConfig(
            **eval_dict.get("metadata_validation", {})
        )
        evaluation = EvaluationConfig(
            enabled=eval_dict.get("enabled", True),
            metadata_validation=metadata_val,
            cluster_metrics=eval_dict.get("cluster_metrics", []),
            convergence_metrics=eval_dict.get("convergence_metrics", []),
        )

        output = OutputConfig(**config_dict.get("output", {}))
        logging_cfg = LoggingConfig(**config_dict.get("logging", {}))

        return cls(
            ud_version=config_dict.get("ud_version", "2.17"),
            ud_source=config_dict.get("ud_source", "hf://commul/universal_dependencies"),
            embeddings=embeddings,
            clustering=clustering,
            bootstrapping=bootstrapping,
            genre_extraction=genre_extraction,
            evaluation=evaluation,
            output=output,
            logging=logging_cfg,
        )


def load_config(config_path: Path | str) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config_dict = yaml.safe_load(f)

    # Expand environment variables
    config_dict = _expand_env_vars(config_dict)

    return Config.from_dict(config_dict)


def _expand_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively expand environment variables in config dictionary."""
    result = {}
    for key, value in config_dict.items():
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            result[key] = os.environ.get(env_var)
        elif isinstance(value, dict):
            result[key] = _expand_env_vars(value)
        else:
            result[key] = value
    return result
