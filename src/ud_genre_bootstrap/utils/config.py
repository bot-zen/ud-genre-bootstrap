"""Configuration management for UD Genre Bootstrap."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

from ud_genre_bootstrap.utils.release_identity import validate_artifact_id


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
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Shared clustering parameters
    max_iter: int = 300  # Maximum iterations for GMM and K-Means
    fit_sample_size: Optional[int] = None  # Optional subsample size for fitting large datasets
    # GMM-specific parameters
    reg_covar: float = 1e-4  # Covariance regularization for GMM


@dataclass
class BootstrappingConfig:
    """Configuration for bootstrapping."""

    min_confidence: float = 0.8
    min_margin: float = 0.05
    reference_weighting: str = "sentence_count"  # "sentence_count" or "uniform"
    max_iterations: int = 10
    fail_on_incomplete: bool = False
    unresolved_handling: str = "null"


@dataclass
class GenreExtractionConfig:
    """Configuration for genre extraction."""

    mapping_path: Optional[str] = None
    patterns_path: Optional[Union[str, List[str]]] = None
    canonical_genres: Optional[List[str]] = None  # Override default canonical genre set


@dataclass
class MetadataValidationConfig:
    """Configuration for metadata validation."""

    protocol: str = "generalization"  # "generalization" or "paper_parity"
    method: str = "kfold"
    k: int = 5
    stratify_by: str = "genre"
    group_by: str = "language"  # "language", "treebank", or None
    anchor_mode: str = "strict"  # "strict" (fold-train anchors only) or "parity" (plus broader single-genre anchors)
    anchor_pool_policy: str = "auto"  # "auto", "train_virtual", "single_genre", or "combined"
    coverage_threshold: float = 0.95  # Minimum sentence-level metadata coverage
    min_genre_sentences: int = 100  # Minimum sentences per genre for virtual splits


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""

    enabled: bool = True
    metadata_validation: MetadataValidationConfig = field(
        default_factory=MetadataValidationConfig
    )
    treebank_sets: Dict[str, List[str]] = field(default_factory=dict)
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
class XGenreEvaluationConfig:
    """Configuration for X-GENRE classifier evaluation."""

    model: str = "classla/xlm-roberta-base-multilingual-text-genre-classifier"
    batch_size: int = 32
    device: str = "auto"  # "auto", "cuda", "cpu"
    # Mapping from X-GENRE labels to UD canonical genres
    genre_mapping: Dict[str, Optional[str]] = field(default_factory=lambda: {
        "News": "news",
        "Legal": "legal",
        "Information/Explanation": "wiki",
        "Forum": "social",
        "Prose/Lyrical": "fiction",
        "Opinion/Argumentation": "reviews",
        "Instruction": "nonfiction",
        "Promotion": "web",
        "Other": None,  # No UD equivalent
    })


@dataclass
class OutputConfig:
    """Configuration for output."""

    genres_path: str = "output/ud-v2.17/genres/"
    embeddings_hf_repo: str = "commul/ud-embeddings-xlm-roberta-base"
    embeddings_revision: str = "2.17"
    genres_hf_repo: str = "commul/ud_genre"
    genres_revision: str = "2.17"
    config_name: Optional[str] = None
    run_id: Optional[str] = None
    ud_source_revision: Optional[str] = None
    baseline_summary_path: Optional[str] = None
    push_to_hub: bool = False
    hf_token: Optional[str] = None


@dataclass
class ReleaseConfig:
    """Public identity for promoted genre artifacts."""

    artifact_id: Optional[str] = None
    scope: str = "full"
    label_schema: str = "ud"
    artifact_version: str = "v1"
    hf_repo: Optional[str] = None
    hf_branches: List[str] = field(default_factory=list)
    hf_tag: Optional[str] = None
    hf_default_branch: str = "main"
    hf_revisions: List[str] = field(default_factory=list)
    source_repo: Optional[str] = None
    source_branch: Optional[str] = None
    source_tag: Optional[str] = None
    source_commit: Optional[str] = None
    git_branch: Optional[str] = None
    git_tag: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging."""

    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class Config:
    """Main configuration for UD Genre Bootstrap."""

    ud_version: str = "2.17"
    ud_source: str = "hf://universal-dependencies/universal_dependencies"
    metadata_path: Optional[str] = None  # Optional path to metadata.json
    include_treebanks: Optional[List[str]] = None  # Treebank codes to include (None = all)
    exclude_treebanks: List[str] = field(default_factory=list)  # Treebank codes to exclude
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    bootstrapping: BootstrappingConfig = field(default_factory=BootstrappingConfig)
    genre_extraction: GenreExtractionConfig = field(default_factory=GenreExtractionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    xgenre_evaluation: XGenreEvaluationConfig = field(default_factory=XGenreEvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    release: ReleaseConfig = field(default_factory=ReleaseConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @staticmethod
    def _parse_optional_int(value: Any, field_name: str) -> Optional[int]:
        """Parse optional integer config values from YAML-friendly inputs."""
        if value is None:
            return None

        if isinstance(value, bool):
            raise ValueError(f"{field_name} must be an integer or null, got boolean")

        if isinstance(value, int):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "null", "none"}:
                return None
            try:
                return int(normalized)
            except ValueError as exc:
                raise ValueError(
                    f"{field_name} must be an integer or null, got '{value}'"
                ) from exc

        raise ValueError(
            f"{field_name} must be an integer or null, got {type(value).__name__}"
        )

    @staticmethod
    def _parse_anchor_mode(value: Any, field_name: str) -> str:
        """Parse and validate evaluation anchor mode."""
        if value is None:
            return "strict"

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string ('strict' or 'parity'), got {type(value).__name__}"
            )

        normalized = value.strip().lower()
        if normalized in {"strict", "parity"}:
            return normalized

        raise ValueError(
            f"{field_name} must be 'strict' or 'parity', got '{value}'"
        )

    @staticmethod
    def _parse_evaluation_protocol(value: Any, field_name: str) -> str:
        """Parse and validate evaluation protocol."""
        if value is None:
            return "generalization"

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string ('generalization' or 'paper_parity'), got {type(value).__name__}"
            )

        normalized = value.strip().lower()
        alias_map = {
            "paper": "paper_parity",
            "parity": "paper_parity",
            "generalisation": "generalization",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized in {"generalization", "paper_parity"}:
            return normalized

        raise ValueError(
            f"{field_name} must be 'generalization' or 'paper_parity', got '{value}'"
        )

    @staticmethod
    def _parse_anchor_pool_policy(value: Any, field_name: str) -> str:
        """Parse and validate evaluation anchor-pool policy."""
        if value is None:
            return "auto"

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string ('auto', 'train_virtual', 'single_genre', or 'combined'), got {type(value).__name__}"
            )

        normalized = value.strip().lower()
        alias_map = {
            "train_virtual_only": "train_virtual",
            "single_genre_only": "single_genre",
            "virtual_only": "train_virtual",
        }
        normalized = alias_map.get(normalized, normalized)
        if normalized in {"auto", "train_virtual", "single_genre", "combined"}:
            return normalized

        raise ValueError(
            f"{field_name} must be 'auto', 'train_virtual', 'single_genre', or 'combined', got '{value}'"
        )

    @staticmethod
    def _parse_reference_weighting(value: Any, field_name: str) -> str:
        """Parse and validate reference embedding weighting mode."""
        if value is None:
            return "sentence_count"

        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a string ('sentence_count' or 'uniform'), got {type(value).__name__}"
            )

        normalized = value.strip().lower()
        if normalized in {"sentence_count", "uniform"}:
            return normalized

        raise ValueError(
            f"{field_name} must be 'sentence_count' or 'uniform', got '{value}'"
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        # Parse nested configs
        embeddings = EmbeddingsConfig(**config_dict.get("embeddings", {}))
        clustering_dict = dict(config_dict.get("clustering", {}))
        if "fit_sample_size" in clustering_dict:
            clustering_dict["fit_sample_size"] = cls._parse_optional_int(
                clustering_dict["fit_sample_size"],
                "clustering.fit_sample_size",
            )
        clustering = ClusteringConfig(**clustering_dict)
        bootstrapping_dict = dict(config_dict.get("bootstrapping", {}))
        if "reference_weighting" in bootstrapping_dict:
            bootstrapping_dict["reference_weighting"] = cls._parse_reference_weighting(
                bootstrapping_dict["reference_weighting"],
                "bootstrapping.reference_weighting",
            )
        bootstrapping = BootstrappingConfig(**bootstrapping_dict)
        genre_extraction = GenreExtractionConfig(**config_dict.get("genre_extraction", {}))

        # Parse evaluation config
        eval_dict = config_dict.get("evaluation", {})
        metadata_val_dict = dict(eval_dict.get("metadata_validation", {}))
        if "protocol" in metadata_val_dict:
            metadata_val_dict["protocol"] = cls._parse_evaluation_protocol(
                metadata_val_dict["protocol"],
                "evaluation.metadata_validation.protocol",
            )
        if "anchor_mode" in metadata_val_dict:
            metadata_val_dict["anchor_mode"] = cls._parse_anchor_mode(
                metadata_val_dict["anchor_mode"],
                "evaluation.metadata_validation.anchor_mode",
            )
        if "anchor_pool_policy" in metadata_val_dict:
            metadata_val_dict["anchor_pool_policy"] = cls._parse_anchor_pool_policy(
                metadata_val_dict["anchor_pool_policy"],
                "evaluation.metadata_validation.anchor_pool_policy",
            )
        metadata_val = MetadataValidationConfig(**metadata_val_dict)
        evaluation = EvaluationConfig(
            enabled=eval_dict.get("enabled", True),
            metadata_validation=metadata_val,
            treebank_sets=eval_dict.get("treebank_sets", {}),
            cluster_metrics=eval_dict.get("cluster_metrics", []),
            convergence_metrics=eval_dict.get("convergence_metrics", []),
        )

        output = OutputConfig(**config_dict.get("output", {}))
        release = ReleaseConfig(**config_dict.get("release", {}))
        if release.artifact_id:
            validate_artifact_id(
                release.artifact_id,
                ud_version=str(config_dict.get("ud_version", "2.17")),
                scope=release.scope,
                label_schema=release.label_schema,
                artifact_version=release.artifact_version,
            )
        logging_cfg = LoggingConfig(**config_dict.get("logging", {}))
        xgenre_evaluation = XGenreEvaluationConfig(**config_dict.get("xgenre_evaluation", {}))

        return cls(
            ud_version=config_dict.get("ud_version", "2.17"),
            ud_source=config_dict.get("ud_source", "hf://universal-dependencies/universal_dependencies"),
            metadata_path=config_dict.get("metadata_path"),
            include_treebanks=config_dict.get("include_treebanks", None),
            exclude_treebanks=config_dict.get("exclude_treebanks", []),
            embeddings=embeddings,
            clustering=clustering,
            bootstrapping=bootstrapping,
            genre_extraction=genre_extraction,
            evaluation=evaluation,
            xgenre_evaluation=xgenre_evaluation,
            output=output,
            release=release,
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

    # Expand config variable references
    config_dict = _expand_config_variables(config_dict)

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


def _flatten_config(config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dictionary for variable expansion.

    Args:
        config_dict: Nested config dictionary
        prefix: Current prefix for nested keys

    Returns:
        Flattened dictionary with dot-separated keys

    Example:
        {"embeddings": {"model": "xlm"}} -> {"embeddings.model": "xlm"}
    """
    result = {}
    for key, value in config_dict.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(_flatten_config(value, full_key))
        else:
            result[full_key] = value
    return result


def _expand_config_vars(value: str, flat_config: Dict[str, Any]) -> str:
    """Expand config variable references in a string.

    Args:
        value: String that may contain {variable} placeholders
        flat_config: Flattened config dictionary for lookups

    Returns:
        String with variables expanded

    Example:
        "output/{ud_version}/data" with {"ud_version": "2.17"} -> "output/2.17/data"
    """
    import re

    # Find all {variable} patterns
    pattern = r'\{([^}]+)\}'

    def replace_var(match):
        var_name = match.group(1)
        if var_name in flat_config:
            val = flat_config[var_name]
            # Convert to string and handle special characters in paths
            return str(val).replace('/', '_') if '/' in str(val) and 'model' in var_name else str(val)
        return match.group(0)  # Return original if not found

    return re.sub(pattern, replace_var, value)


def _expand_config_variables(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Expand config variable references in specific fields.

    This is applied after the config is loaded to expand references like:
    - {ud_version}
    - {embeddings.model}
    - {clustering.method}
    etc.

    Args:
        config_dict: Config dictionary with potential variable references

    Returns:
        Config dictionary with variables expanded in specific fields
    """
    # Flatten config for easy lookup
    flat_config = _flatten_config(config_dict)

    # Fields that support variable expansion
    expandable_fields = [
        ("embeddings", "cache_dir"),
        ("output", "genres_path"),
    ]

    for section, field in expandable_fields:
        if section in config_dict and field in config_dict[section]:
            value = config_dict[section][field]
            if isinstance(value, str) and '{' in value:
                config_dict[section][field] = _expand_config_vars(value, flat_config)

    return config_dict
