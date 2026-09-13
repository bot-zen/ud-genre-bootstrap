"""Utilities for exploratory reduced-genre schema analysis."""

from __future__ import annotations

import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from ud_genre_bootstrap.utils.sentence_refs import extract_sentence_ref_parts

METADATA_DERIVED_METHODS = {"single-genre-treebank", "virtual-split"}


@dataclass(frozen=True)
class CandidateSchema:
    """A complete projection from source genres to a reduced target inventory."""

    name: str
    description: str
    mapping: dict[str, str]
    target_genres: list[str]
    rationale: dict[str, str]


@dataclass(frozen=True)
class ProjectionMetrics:
    """Metrics for one candidate reduced-genre projection."""

    target_genre_count: int
    label_counts: dict[str, int]
    cluster_purity: float
    cluster_purity_gain: float
    evaluation_micro_f1: Optional[float]
    evaluation_macro_f1: Optional[float]


def load_candidate_schema_config(path: Path) -> tuple[list[str], dict[str, CandidateSchema]]:
    """Load and validate reduced-schema candidates from YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    source_genres = payload.get("source_genres") or []
    if not source_genres or not all(isinstance(genre, str) for genre in source_genres):
        raise ValueError("Candidate schema config must define string list 'source_genres'.")

    raw_candidates = payload.get("candidate_schemas") or payload.get("candidates") or {}
    if not isinstance(raw_candidates, dict) or not raw_candidates:
        raise ValueError("Candidate schema config must define at least one candidate schema.")

    candidates: dict[str, CandidateSchema] = {}
    for name, raw_candidate in raw_candidates.items():
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"Candidate schema {name!r} must be a mapping.")
        mapping = raw_candidate.get("mapping") or {}
        target_genres = raw_candidate.get("target_genres") or []
        validate_candidate_mapping(
            mapping=mapping,
            source_genres=source_genres,
            target_genres=target_genres,
            schema_name=str(name),
        )
        candidates[str(name)] = CandidateSchema(
            name=str(name),
            description=str(raw_candidate.get("description") or ""),
            mapping={str(key): str(value) for key, value in mapping.items()},
            target_genres=[str(genre) for genre in target_genres],
            rationale={
                str(key): str(value)
                for key, value in (raw_candidate.get("rationale") or {}).items()
            },
        )

    return [str(genre) for genre in source_genres], candidates


def validate_candidate_mapping(
    mapping: Mapping[str, str],
    source_genres: Iterable[str],
    *,
    target_genres: Optional[Iterable[str]] = None,
    schema_name: str = "candidate",
) -> None:
    """Validate that a projection maps each source genre exactly once."""
    if not isinstance(mapping, Mapping):
        raise ValueError(f"{schema_name}: mapping must be a dictionary.")

    source_set = set(source_genres)
    mapping_keys = set(mapping)
    missing = sorted(source_set - mapping_keys)
    extra = sorted(mapping_keys - source_set)
    if missing:
        raise ValueError(f"{schema_name}: missing source genre mapping(s): {', '.join(missing)}")
    if extra:
        raise ValueError(f"{schema_name}: unknown source genre mapping(s): {', '.join(extra)}")

    blank_targets = sorted(
        str(source)
        for source, target in mapping.items()
        if not isinstance(target, str) or not target
    )
    if blank_targets:
        raise ValueError(
            f"{schema_name}: blank target genre(s) for source genre(s): "
            f"{', '.join(blank_targets)}"
        )

    if target_genres is not None:
        target_set = set(target_genres)
        if not target_set:
            raise ValueError(f"{schema_name}: target_genres must not be empty.")
        unknown_targets = sorted(set(mapping.values()) - target_set)
        if unknown_targets:
            raise ValueError(
                f"{schema_name}: unknown target genre(s): {', '.join(unknown_targets)}"
            )


def project_labels(labels: Sequence[str], mapping: Mapping[str, str]) -> list[str]:
    """Apply a source-to-target genre mapping to a label sequence."""
    projected = []
    for label in labels:
        if label not in mapping:
            raise ValueError(f"No target genre configured for source genre {label!r}.")
        projected.append(mapping[label])
    return projected


def project_confusion_matrix(
    confusion_matrix: Sequence[Sequence[int]],
    source_labels: Sequence[str],
    mapping: Mapping[str, str],
    target_labels: Optional[Sequence[str]] = None,
) -> tuple[list[list[int]], list[str]]:
    """Aggregate a source-label confusion matrix under a reduced schema."""
    if len(confusion_matrix) != len(source_labels):
        raise ValueError("Confusion matrix row count must match source_labels.")
    for row in confusion_matrix:
        if len(row) != len(source_labels):
            raise ValueError("Confusion matrix must be square over source_labels.")

    targets = list(target_labels or sorted({mapping[label] for label in source_labels}))
    target_index = {label: idx for idx, label in enumerate(targets)}
    projected = np.zeros((len(targets), len(targets)), dtype=np.int64)

    for row_idx, true_label in enumerate(source_labels):
        if true_label not in mapping:
            raise ValueError(f"No target genre configured for source genre {true_label!r}.")
        projected_true = mapping[true_label]
        if projected_true not in target_index:
            raise ValueError(f"Projected true label {projected_true!r} is not in target labels.")
        for col_idx, pred_label in enumerate(source_labels):
            if pred_label not in mapping:
                raise ValueError(f"No target genre configured for source genre {pred_label!r}.")
            projected_pred = mapping[pred_label]
            if projected_pred not in target_index:
                raise ValueError(
                    f"Projected predicted label {projected_pred!r} is not in target labels."
                )
            projected[target_index[projected_true], target_index[projected_pred]] += int(
                confusion_matrix[row_idx][col_idx]
            )

    return projected.tolist(), targets


def classification_metrics_from_confusion(
    confusion_matrix: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Compute micro-F1-equivalent accuracy and macro-F1 from a confusion matrix."""
    matrix = np.asarray(confusion_matrix, dtype=np.float64)
    if matrix.size == 0 or matrix.sum() == 0:
        return {"micro_f1": 0.0, "macro_f1": 0.0}

    micro_f1 = float(np.trace(matrix) / matrix.sum())
    f1_scores = []
    for idx in range(matrix.shape[0]):
        tp = matrix[idx, idx]
        fp = matrix[:, idx].sum() - tp
        fn = matrix[idx, :].sum() - tp
        denom = (2 * tp) + fp + fn
        f1_scores.append(float((2 * tp) / denom) if denom else 0.0)

    return {"micro_f1": micro_f1, "macro_f1": float(np.mean(f1_scores))}


def analyze_genre_schema(
    *,
    release_dir: Path,
    candidate_config: Path,
    output_dir: Path,
    evaluation_results: Optional[Path] = None,
    cluster_state: Optional[Path] = None,
    ud_version: Optional[str] = None,
    top_pairs: int = 40,
) -> dict[str, Any]:
    """Run reduced-schema analysis and write report artifacts."""
    labels_path = release_dir / "all_genres.parquet"
    clusters_path = release_dir / "clusters" / "cluster_assignments.parquet"
    if not labels_path.exists():
        raise FileNotFoundError(f"Genre release file not found: {labels_path}")
    if not clusters_path.exists():
        raise FileNotFoundError(f"Cluster assignment file not found: {clusters_path}")

    source_genres, candidate_schemas = load_candidate_schema_config(candidate_config)
    labels_df = pd.read_parquet(
        labels_path,
        columns=["treebank", "split", "sent_id", "genre", "confidence", "method"],
    )
    observed_genres = sorted(labels_df["genre"].dropna().unique().tolist())
    validate_observed_genres(observed_genres, source_genres)

    clusters_df = pd.read_parquet(
        clusters_path,
        columns=["treebank", "split", "sent_id", "cluster_id"],
    )
    raw_cluster_assignment_rows = int(len(clusters_df))
    clusters_df = clusters_df.drop_duplicates(
        ["treebank", "split", "sent_id", "cluster_id"]
    )
    cluster_assignment_summary = {
        "raw_rows": raw_cluster_assignment_rows,
        "deduplicated_rows": int(len(clusters_df)),
        "exact_duplicate_rows_removed": raw_cluster_assignment_rows - int(len(clusters_df)),
        "sentences_with_multiple_cluster_contexts": int(
            (
                clusters_df.groupby(["treebank", "split", "sent_id"], observed=True)[
                    "cluster_id"
                ].nunique()
                > 1
            ).sum()
        ),
    }
    joined_df = clusters_df.merge(
        labels_df,
        on=["treebank", "split", "sent_id"],
        how="inner",
        validate="many_to_one",
    )

    evaluation_payload = load_evaluation_results(evaluation_results)
    centroid_similarity = (
        build_embedding_centroid_similarity(labels_df, cluster_state)
        if cluster_state is not None
        else None
    )
    support = build_support_summary(labels_df)
    pair_rows = build_pair_similarity_table(
        labels_df=labels_df,
        joined_df=joined_df,
        evaluation_payload=evaluation_payload,
        centroid_similarity=centroid_similarity,
    )
    metadata_joined_df = joined_df[joined_df["method"].isin(METADATA_DERIVED_METHODS)]
    cluster_purity_df = metadata_joined_df if not metadata_joined_df.empty else joined_df
    cluster_purity_basis = (
        "metadata-derived" if not metadata_joined_df.empty else "all-labels"
    )
    source_cluster_purity = compute_weighted_cluster_purity(
        cluster_purity_df,
        label_column="genre",
    )

    projection_scores = {
        name: score_candidate_schema(
            candidate,
            labels_df=labels_df,
            joined_df=cluster_purity_df,
            source_cluster_purity=source_cluster_purity,
            evaluation_payload=evaluation_payload,
            cluster_purity_basis=cluster_purity_basis,
        )
        for name, candidate in sorted(candidate_schemas.items())
    }
    data_driven = build_data_driven_candidates(
        pair_rows=pair_rows,
        source_genres=source_genres,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "candidate_mappings.json",
        {
            "source_genres": source_genres,
            "cluster_assignment_summary": cluster_assignment_summary,
            "candidate_schemas": {
                name: {
                    "description": candidate.description,
                    "target_genres": candidate.target_genres,
                    "mapping": candidate.mapping,
                    "rationale": candidate.rationale,
                }
                for name, candidate in sorted(candidate_schemas.items())
            },
            "data_driven_candidates": data_driven,
        },
    )
    write_pair_table(output_dir / "genre_similarity.tsv", pair_rows)
    write_json(output_dir / "projection_scores.json", projection_scores)
    write_report(
        output_dir / "report.md",
        ud_version=ud_version,
        release_dir=release_dir,
        candidate_config=candidate_config,
        support=support,
        cluster_assignment_summary=cluster_assignment_summary,
        cluster_purity_basis=cluster_purity_basis,
        centroid_available=centroid_similarity is not None,
        pair_rows=pair_rows[:top_pairs],
        projection_scores=projection_scores,
        data_driven=data_driven,
        evaluation_available=evaluation_payload is not None,
    )
    write_figures(output_dir=output_dir, source_genres=source_genres, pair_rows=pair_rows)

    return {
        "output_dir": str(output_dir),
        "observed_genres": observed_genres,
        "candidate_schema_count": len(candidate_schemas),
        "top_pair_count": min(top_pairs, len(pair_rows)),
        "evaluation_available": evaluation_payload is not None,
        "centroid_available": centroid_similarity is not None,
        "cluster_assignment_summary": cluster_assignment_summary,
    }


def validate_observed_genres(observed_genres: Sequence[str], source_genres: Sequence[str]) -> None:
    """Ensure candidate mappings cover the release labels being analyzed."""
    source_set = set(source_genres)
    unknown = sorted(set(observed_genres) - source_set)
    if unknown:
        raise ValueError(
            "Release contains genre(s) not listed in candidate config source_genres: "
            + ", ".join(unknown)
        )


def load_evaluation_results(path: Optional[Path]) -> Optional[dict[str, Any]]:
    """Load an optional full evaluation result export."""
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Evaluation results file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "results" in payload and "confusion_matrix" in payload["results"]:
        return payload["results"]
    if "sets" in payload:
        sets = payload["sets"] or {}
        if "default" in sets:
            return sets["default"].get("results")
        first_set = sorted(sets)[0] if sets else None
        if first_set:
            return sets[first_set].get("results")
    if "confusion_matrix" in payload:
        return payload
    return None


def build_support_summary(labels_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize source genre support, provenance, and confidence."""
    total = int(len(labels_df))
    method_table = (
        labels_df.groupby(["genre", "method"], observed=True)
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    confidence = (
        labels_df.groupby("genre", observed=True)["confidence"]
        .agg(["count", "mean", "min"])
        .sort_values("count")
    )
    genre_counts = labels_df["genre"].value_counts().sort_values(ascending=False)
    return {
        "total_sentences": total,
        "genre_counts": {str(k): int(v) for k, v in genre_counts.items()},
        "genre_percentages": {
            str(k): round(float(v) * 100.0 / total, 4) for k, v in genre_counts.items()
        },
        "method_counts_by_genre": _dataframe_to_nested_ints(method_table),
        "confidence_by_genre": {
            str(index): {
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 6),
                "min": round(float(row["min"]), 6),
            }
            for index, row in confidence.iterrows()
        },
    }


def build_pair_similarity_table(
    *,
    labels_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    evaluation_payload: Optional[dict[str, Any]],
    centroid_similarity: Optional[Mapping[tuple[str, str], float]] = None,
) -> list[dict[str, Any]]:
    """Build pairwise genre merge evidence from clusters, co-occurrence, and evaluation."""
    genre_counts = Counter(labels_df["genre"].tolist())
    metadata_joined = joined_df[joined_df["method"].isin(METADATA_DERIVED_METHODS)]

    metadata_cluster = _cluster_pair_scores(metadata_joined, genre_counts)
    all_cluster = _cluster_pair_scores(joined_df, genre_counts)
    same_split = _group_pair_scores(labels_df, ["treebank", "split"], genre_counts)
    eval_confusion = _evaluation_confusion_scores(evaluation_payload)

    pairs = sorted(_all_pairs(sorted(genre_counts)))
    rows: list[dict[str, Any]] = []
    for genre_a, genre_b in pairs:
        evidence = {
            "metadata_cluster_cooccurrence": metadata_cluster.get((genre_a, genre_b), 0.0),
            "all_label_cluster_cooccurrence": all_cluster.get((genre_a, genre_b), 0.0),
            "same_split_cooccurrence": same_split.get((genre_a, genre_b), 0.0),
            "evaluation_confusion": eval_confusion.get((genre_a, genre_b)),
            "embedding_centroid_similarity": (
                None
                if centroid_similarity is None
                else centroid_similarity.get((genre_a, genre_b), 0.0)
            ),
        }
        combined_score = _combine_pair_evidence(evidence)
        cluster_merge_score = _combine_cluster_merge_evidence(evidence)
        rows.append(
            {
                "genre_a": genre_a,
                "genre_b": genre_b,
                "combined_score": round(combined_score, 6),
                "cluster_merge_score": round(cluster_merge_score, 6),
                "metadata_cluster_cooccurrence": round(
                    evidence["metadata_cluster_cooccurrence"], 6
                ),
                "all_label_cluster_cooccurrence": round(
                    evidence["all_label_cluster_cooccurrence"], 6
                ),
                "same_split_cooccurrence": round(evidence["same_split_cooccurrence"], 6),
                "evaluation_confusion": (
                    None
                    if evidence["evaluation_confusion"] is None
                    else round(evidence["evaluation_confusion"], 6)
                ),
                "embedding_centroid_similarity": (
                    None
                    if evidence["embedding_centroid_similarity"] is None
                    else round(evidence["embedding_centroid_similarity"], 6)
                ),
                "genre_a_count": int(genre_counts[genre_a]),
                "genre_b_count": int(genre_counts[genre_b]),
            }
        )

    return sorted(rows, key=lambda row: row["cluster_merge_score"], reverse=True)


def build_embedding_centroid_similarity(
    labels_df: pd.DataFrame,
    cluster_state_path: Path,
) -> dict[tuple[str, str], float]:
    """Compute genre-centroid cosine similarity from metadata-derived embeddings."""
    if not cluster_state_path.exists():
        raise FileNotFoundError(f"Cluster state file not found: {cluster_state_path}")

    metadata_labels = labels_df[labels_df["method"].isin(METADATA_DERIVED_METHODS)]
    label_lookup = {
        (str(row.treebank), str(row.split), str(row.sent_id)): str(row.genre)
        for row in metadata_labels.itertuples(index=False)
    }

    with cluster_state_path.open("rb") as handle:
        cluster_state = pickle.load(handle)

    sums: dict[str, np.ndarray] = {}
    counts: Counter[str] = Counter()
    for (tb_code, split_name), emb_data in (
        cluster_state.get("embeddings_by_tb") or {}
    ).items():
        sent_ids = emb_data.get("sent_id") or []
        embeddings = np.asarray(emb_data.get("embedding"))
        if len(sent_ids) == 0 or len(embeddings) == 0:
            continue
        for sent_ref, embedding in zip(sent_ids, embeddings):
            ref_tb, ref_split, raw_sent_id = extract_sentence_ref_parts(
                sent_ref,
                tb_code=str(tb_code),
                split_name=str(split_name),
            )
            genre = label_lookup.get((ref_tb, ref_split, raw_sent_id))
            if genre is None:
                continue
            vector = np.asarray(embedding, dtype=np.float64)
            if genre not in sums:
                sums[genre] = np.zeros_like(vector, dtype=np.float64)
            sums[genre] += vector
            counts[genre] += 1

    centroids = {
        genre: vector_sum / counts[genre]
        for genre, vector_sum in sums.items()
        if counts[genre] > 0
    }
    scores: dict[tuple[str, str], float] = {}
    for genre_a, genre_b in _all_pairs(sorted(centroids)):
        scores[(genre_a, genre_b)] = _cosine_similarity_01(
            centroids[genre_a],
            centroids[genre_b],
        )
    return scores


def score_candidate_schema(
    candidate: CandidateSchema,
    *,
    labels_df: pd.DataFrame,
    joined_df: pd.DataFrame,
    source_cluster_purity: float,
    evaluation_payload: Optional[dict[str, Any]],
    cluster_purity_basis: str,
) -> dict[str, Any]:
    """Score a source-to-target genre projection."""
    projected_labels = labels_df["genre"].map(candidate.mapping)
    target_counts = projected_labels.value_counts().sort_values(ascending=False)

    projected_clusters = joined_df.copy()
    projected_clusters["projected_genre"] = projected_clusters["genre"].map(candidate.mapping)
    projected_cluster_purity = compute_weighted_cluster_purity(
        projected_clusters,
        label_column="projected_genre",
    )

    projected_evaluation = None
    if evaluation_payload:
        source_labels = evaluation_payload.get("genre_labels") or []
        matrix = evaluation_payload.get("confusion_matrix") or []
        if source_labels and matrix:
            projected_matrix, target_labels = project_confusion_matrix(
                matrix,
                source_labels,
                candidate.mapping,
                candidate.target_genres,
            )
            projected_evaluation = {
                "target_labels": target_labels,
                "confusion_matrix": projected_matrix,
                **classification_metrics_from_confusion(projected_matrix),
            }

    metrics = ProjectionMetrics(
        target_genre_count=len(target_counts),
        label_counts={str(k): int(v) for k, v in target_counts.items()},
        cluster_purity=round(projected_cluster_purity, 6),
        cluster_purity_gain=round(projected_cluster_purity - source_cluster_purity, 6),
        evaluation_micro_f1=(
            None
            if projected_evaluation is None
            else round(float(projected_evaluation["micro_f1"]), 6)
        ),
        evaluation_macro_f1=(
            None
            if projected_evaluation is None
            else round(float(projected_evaluation["macro_f1"]), 6)
        ),
    )
    return {
        "description": candidate.description,
        "target_genre_count": metrics.target_genre_count,
        "target_genres": candidate.target_genres,
        "label_counts": metrics.label_counts,
        "min_target_count": min(metrics.label_counts.values()) if metrics.label_counts else 0,
        "cluster_purity_basis": cluster_purity_basis,
        "source_cluster_purity": round(source_cluster_purity, 6),
        "projected_cluster_purity": metrics.cluster_purity,
        "cluster_purity_gain": metrics.cluster_purity_gain,
        "evaluation": projected_evaluation,
        "evaluation_micro_f1": metrics.evaluation_micro_f1,
        "evaluation_macro_f1": metrics.evaluation_macro_f1,
    }


def compute_weighted_cluster_purity(df: pd.DataFrame, *, label_column: str) -> float:
    """Compute weighted majority-label purity over treebank-local clusters."""
    if df.empty:
        return 0.0

    total = 0
    correct = 0
    for _cluster_key, group in df.groupby(["treebank", "split", "cluster_id"], observed=True):
        counts = group[label_column].value_counts()
        cluster_total = int(counts.sum())
        total += cluster_total
        correct += int(counts.max()) if cluster_total else 0

    return float(correct / total) if total else 0.0


def build_data_driven_candidates(
    *,
    pair_rows: Sequence[Mapping[str, Any]],
    source_genres: Sequence[str],
    thresholds: Sequence[float] = (0.5, 0.35, 0.25),
) -> dict[str, Any]:
    """Create conservative merge candidates from high-scoring pair evidence."""
    candidates: dict[str, Any] = {}
    for threshold in thresholds:
        components = _components_for_threshold(pair_rows, source_genres, threshold)
        reduced_components = [component for component in components if len(component) > 1]
        if not reduced_components:
            continue
        mapping: dict[str, str] = {}
        target_genres: list[str] = []
        for component in components:
            target = _component_label(component)
            target_genres.append(target)
            for source in component:
                mapping[source] = target
        name = f"data_threshold_{str(threshold).replace('.', '_')}"
        candidates[name] = {
            "description": (
                "Connected components from unsupervised cluster-merge evidence "
                f"with cluster_merge_score >= {threshold}."
            ),
            "threshold": threshold,
            "score_column": "cluster_merge_score",
            "target_genres": sorted(target_genres),
            "mapping": mapping,
            "merged_components": reduced_components,
        }
    return candidates


def write_pair_table(path: Path, pair_rows: Sequence[Mapping[str, Any]]) -> None:
    """Write pairwise similarity/evidence rows as TSV."""
    columns = [
        "genre_a",
        "genre_b",
        "combined_score",
        "cluster_merge_score",
        "metadata_cluster_cooccurrence",
        "all_label_cluster_cooccurrence",
        "same_split_cooccurrence",
        "embedding_centroid_similarity",
        "evaluation_confusion",
        "genre_a_count",
        "genre_b_count",
    ]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in pair_rows:
            values = [
                "" if row.get(column) is None else str(row.get(column))
                for column in columns
            ]
            handle.write(
                "\t".join(values) + "\n"
            )


def write_report(
    path: Path,
    *,
    ud_version: Optional[str],
    release_dir: Path,
    candidate_config: Path,
    support: Mapping[str, Any],
    cluster_assignment_summary: Mapping[str, int],
    cluster_purity_basis: str,
    centroid_available: bool,
    pair_rows: Sequence[Mapping[str, Any]],
    projection_scores: Mapping[str, Any],
    data_driven: Mapping[str, Any],
    evaluation_available: bool,
) -> None:
    """Write the markdown analysis report."""
    lines = [
        "# Reduced Genre Schema Analysis",
        "",
        f"- UD version: `{ud_version or 'not specified'}`",
        f"- Release directory: `{release_dir}`",
        f"- Candidate config: `{candidate_config}`",
        f"- Total labeled sentences: {support['total_sentences']:,}",
        f"- Cluster assignment rows after exact deduplication: "
        f"{cluster_assignment_summary['deduplicated_rows']:,}",
        f"- Sentences with multiple cluster contexts: "
        f"{cluster_assignment_summary['sentences_with_multiple_cluster_contexts']:,}",
        f"- Candidate cluster-purity basis: `{cluster_purity_basis}` rows",
        f"- Embedding centroid evidence included: {'yes' if centroid_available else 'no'}",
        f"- Evaluation export included: {'yes' if evaluation_available else 'no'}",
        "",
        "## Current Genre Support",
        "",
        "| Genre | Sentences | Percent | Mean confidence | Main provenance |",
        "| --- | ---: | ---: | ---: | --- |",
    ]

    method_counts = support["method_counts_by_genre"]
    confidence = support["confidence_by_genre"]
    for genre, count in support["genre_counts"].items():
        methods = method_counts.get(genre, {})
        main_method = "n/a"
        if methods:
            main_method = sorted(methods.items(), key=lambda item: (-item[1], item[0]))[0][0]
        lines.append(
            "| {genre} | {count:,} | {pct:.2f}% | {conf:.3f} | {method} |".format(
                genre=genre,
                count=count,
                pct=support["genre_percentages"][genre],
                conf=confidence.get(genre, {}).get("mean", 0.0),
                method=main_method,
            )
        )

    lines.extend(
        [
            "",
            "## Candidate Projection Scores",
            "",
            "| Candidate | Target genres | Min target support | Cluster purity | "
            "Gain | Eval micro-F1 | Eval macro-F1 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, scores in projection_scores.items():
        projection_row = (
            "| {name} | {targets} | {min_count:,} | {purity:.3f} | {gain:.3f} | "
            "{micro} | {macro} |"
        )
        lines.append(
            projection_row.format(
                name=name,
                targets=scores["target_genre_count"],
                min_count=scores["min_target_count"],
                purity=scores["projected_cluster_purity"],
                gain=scores["cluster_purity_gain"],
                micro=_format_optional_float(scores.get("evaluation_micro_f1")),
                macro=_format_optional_float(scores.get("evaluation_macro_f1")),
            )
        )

    lines.extend(
        [
            "",
            "## Strongest Pairwise Merge Evidence",
            "",
            "| Genre A | Genre B | Cluster merge | Combined | Metadata cluster | "
            "All-label cluster | Same split | Centroid | Eval confusion |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in pair_rows:
        pair_row = (
            "| {a} | {b} | {cluster_merge:.3f} | {combined:.3f} | "
            "{metadata:.3f} | {all_cluster:.3f} | {same:.3f} | {centroid} | "
            "{eval_conf} |"
        )
        lines.append(
            pair_row.format(
                a=row["genre_a"],
                b=row["genre_b"],
                cluster_merge=row["cluster_merge_score"],
                combined=row["combined_score"],
                metadata=row["metadata_cluster_cooccurrence"],
                all_cluster=row["all_label_cluster_cooccurrence"],
                same=row["same_split_cooccurrence"],
                centroid=_format_optional_float(
                    row.get("embedding_centroid_similarity")
                ),
                eval_conf=_format_optional_float(row.get("evaluation_confusion")),
            )
        )

    lines.extend(["", "## Data-Driven Component Candidates", ""])
    if data_driven:
        for name, candidate in data_driven.items():
            lines.append(
                f"- `{name}`: {len(candidate['target_genres'])} target genres; "
                f"merged components: {candidate['merged_components']}"
            )
    else:
        lines.append("- No pairwise threshold produced a conservative merge component.")

    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat this report as exploratory evidence, not a release-schema decision.",
            "- Metadata-derived rows are the primary cluster evidence; all-label cluster "
            "evidence is secondary.",
            "- Automatic data-driven components use `cluster_merge_score`; evaluation "
            "confusion and centroid similarity are diagnostic context.",
            "- Sparse genres can look easy to merge because they have few anchors. "
            "Check counts before adopting a merge.",
            "- A future publication should introduce a new `label_schema` train rather "
            "than changing `ud` in place.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_figures(
    *,
    output_dir: Path,
    source_genres: Sequence[str],
    pair_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write optional heatmap/dendrogram figures when plotting dependencies exist."""
    try:
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
    except ImportError:
        return

    similarity = np.eye(len(source_genres), dtype=np.float64)
    index = {genre: idx for idx, genre in enumerate(source_genres)}
    for row in pair_rows:
        i = index[row["genre_a"]]
        j = index[row["genre_b"]]
        similarity[i, j] = similarity[j, i] = float(row["cluster_merge_score"])
    distance = np.clip(1.0 - similarity, 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(similarity, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(source_genres)))
    ax.set_yticks(range(len(source_genres)))
    ax.set_xticklabels(source_genres, rotation=90)
    ax.set_yticklabels(source_genres)
    ax.set_title("Genre Merge Evidence Similarity")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "genre_similarity_heatmap.png", dpi=150)
    plt.close(fig)

    condensed = squareform(distance, checks=False)
    linked = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(linked, labels=list(source_genres), leaf_rotation=90, ax=ax)
    ax.set_title("Genre Merge Evidence Dendrogram")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(output_dir / "genre_similarity_dendrogram.png", dpi=150)
    plt.close(fig)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON with stable formatting."""
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _cluster_pair_scores(df: pd.DataFrame, genre_counts: Counter) -> dict[tuple[str, str], float]:
    scores: defaultdict[tuple[str, str], float] = defaultdict(float)
    if df.empty:
        return {}
    grouped = df.groupby(["treebank", "split", "cluster_id"], observed=True)["genre"]
    for _cluster_key, labels in grouped:
        counts = Counter(labels.tolist())
        for genre_a, genre_b in _all_pairs(sorted(counts)):
            scores[(genre_a, genre_b)] += min(counts[genre_a], counts[genre_b])
    return _normalize_pair_scores(scores, genre_counts)


def _group_pair_scores(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    genre_counts: Counter,
) -> dict[tuple[str, str], float]:
    scores: defaultdict[tuple[str, str], float] = defaultdict(float)
    for _group_key, group in df.groupby(list(group_columns), observed=True):
        counts = Counter(group["genre"].tolist())
        for genre_a, genre_b in _all_pairs(sorted(counts)):
            scores[(genre_a, genre_b)] += min(counts[genre_a], counts[genre_b])
    return _normalize_pair_scores(scores, genre_counts)


def _normalize_pair_scores(
    pair_counts: Mapping[tuple[str, str], float],
    genre_counts: Counter,
) -> dict[tuple[str, str], float]:
    normalized = {}
    for pair, count in pair_counts.items():
        denom = min(genre_counts[pair[0]], genre_counts[pair[1]])
        normalized[pair] = min(float(count / denom), 1.0) if denom else 0.0
    return normalized


def _evaluation_confusion_scores(
    evaluation_payload: Optional[dict[str, Any]],
) -> dict[tuple[str, str], float]:
    if not evaluation_payload:
        return {}
    labels = evaluation_payload.get("genre_labels") or []
    matrix = np.asarray(evaluation_payload.get("confusion_matrix") or [], dtype=np.float64)
    if len(labels) == 0 or matrix.size == 0:
        return {}

    scores = {}
    row_totals = matrix.sum(axis=1)
    for i, genre_a in enumerate(labels):
        for j, genre_b in enumerate(labels):
            if i >= j:
                continue
            denom = min(row_totals[i], row_totals[j])
            score = (matrix[i, j] + matrix[j, i]) / denom if denom else 0.0
            scores[(genre_a, genre_b)] = float(min(score, 1.0))
    return scores


def _combine_pair_evidence(evidence: Mapping[str, Optional[float]]) -> float:
    has_eval = evidence.get("evaluation_confusion") is not None
    has_centroid = evidence.get("embedding_centroid_similarity") is not None
    if has_eval and has_centroid:
        weights = {
            "metadata_cluster_cooccurrence": 0.45,
            "embedding_centroid_similarity": 0.30,
            "evaluation_confusion": 0.15,
            "same_split_cooccurrence": 0.10,
        }
    elif has_centroid:
        weights = {
            "metadata_cluster_cooccurrence": 0.55,
            "embedding_centroid_similarity": 0.30,
            "same_split_cooccurrence": 0.15,
        }
    elif has_eval:
        weights = {
            "metadata_cluster_cooccurrence": 0.55,
            "evaluation_confusion": 0.35,
            "same_split_cooccurrence": 0.10,
        }
    else:
        weights = {
            "metadata_cluster_cooccurrence": 0.85,
            "same_split_cooccurrence": 0.15,
        }
    numer = 0.0
    denom = 0.0
    for key, weight in weights.items():
        value = evidence.get(key)
        if value is None:
            continue
        numer += float(value) * weight
        denom += weight
    return numer / denom if denom else 0.0


def _combine_cluster_merge_evidence(evidence: Mapping[str, Optional[float]]) -> float:
    """Combine unsupervised cluster/split evidence for automatic components."""
    metadata_cluster = float(evidence.get("metadata_cluster_cooccurrence") or 0.0)
    same_split = float(evidence.get("same_split_cooccurrence") or 0.0)
    return (0.85 * metadata_cluster) + (0.15 * same_split)


def _cosine_similarity_01(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(1.0, cosine))


def _components_for_threshold(
    pair_rows: Sequence[Mapping[str, Any]],
    source_genres: Sequence[str],
    threshold: float,
) -> list[list[str]]:
    parent = {genre: genre for genre in source_genres}

    def find(genre: str) -> str:
        while parent[genre] != genre:
            parent[genre] = parent[parent[genre]]
            genre = parent[genre]
        return genre

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for row in pair_rows:
        if float(row["cluster_merge_score"]) >= threshold:
            union(str(row["genre_a"]), str(row["genre_b"]))

    components: defaultdict[str, list[str]] = defaultdict(list)
    for genre in source_genres:
        components[find(genre)].append(genre)
    return sorted(
        [sorted(component) for component in components.values()],
        key=lambda c: (len(c), c),
    )


def _component_label(component: Sequence[str]) -> str:
    if len(component) == 1:
        return component[0]
    return "group_" + "_".join(component)


def _all_pairs(items: Sequence[str]) -> Iterable[tuple[str, str]]:
    for i, left in enumerate(items):
        for right in items[i + 1 :]:
            yield left, right


def _dataframe_to_nested_ints(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    return {
        str(index): {str(column): int(value) for column, value in row.items()}
        for index, row in df.iterrows()
    }


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    return value
