from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, f1_score

from ud_genre_bootstrap.bootstrapping.bootstrapper import GenreBootstrapper
from ud_genre_bootstrap.cli import (
    collect_treebank_descriptor_split_keys,
    filter_treebank_descriptors_by_available_splits,
    resolve_paper_evaluation_treebank_genres,
)
from ud_genre_bootstrap.evaluation.validator import ClusteringEvaluator
from ud_genre_bootstrap.utils.config import load_config
from ud_genre_bootstrap.utils.sentence_split_map import (
    filter_embeddings_by_sentence_split_map,
    load_sentence_split_map,
)


@dataclass
class ParityAuditPaths:
    config_path: Path
    split_map_path: Path
    original_repo: Path
    ud_root: Path
    cache_dir: Path
    json_out: Path
    markdown_out: Path


@dataclass
class PreparedCurrentParity:
    bootstrapper: GenreBootstrapper
    sentence_metadata: Dict[Tuple[str, str, str], str]
    clustering_treebanks: List[Dict[str, Any]]
    scoring_treebanks: List[Dict[str, Any]]
    single_anchor_treebanks: List[Dict[str, Any]]
    embeddings_by_tb: Dict[Tuple[str, str], Dict[str, Any]]
    paper_treebank_genre_map: Dict[str, List[str]]


def original_get_schedule(
    domain_combinations: List[Tuple[str, ...]],
) -> List[Dict[str, List[Tuple[str, ...]]]]:
    schedule = []
    known_domains = {dc[0] for dc in domain_combinations if len(dc) == 1}
    known_combinations = {dc for dc in domain_combinations if len(dc) == 1}
    prev_num_known_domains = -1
    while prev_num_known_domains != len(known_domains):
        environment = {"known": [], "predict": [], "disjunct": []}
        prev_num_known_domains = len(known_domains)
        environment["known"] = list(sorted(known_domains))
        predict_combinations = {
            dc
            for dc in domain_combinations
            if (len(set(dc) & known_domains) > 0) and (dc not in known_combinations)
        }
        environment["predict"] = list(sorted(predict_combinations))
        unknown_combinations = {
            dc for dc in domain_combinations if len(set(dc) & known_domains) == 0
        }
        environment["disjunct"] = list(sorted(unknown_combinations))
        new_known_domains = {
            (set(dc) - known_domains).pop()
            for dc in domain_combinations
            if len(set(dc) - known_domains) == 1
        }
        known_domains |= new_known_domains
        new_known_combinations = {
            dc for dc in domain_combinations if len(set(dc) - known_domains) == 0
        }
        known_combinations |= new_known_combinations
        schedule.append(environment)
    return schedule


def parse_cache_name(path: Path) -> Tuple[str, str]:
    match = re.match(r"(.+)-(train|dev|test)_ids\.txt$", path.name)
    if match is None:
        raise ValueError(f"Unexpected cache ids file: {path.name}")
    return match.group(1), match.group(2)


def parse_conllu_file_name(name: str) -> Tuple[str, str]:
    match = re.match(r"(.+)-ud-(train|dev|test)\.conllu$", name)
    if match is None:
        raise ValueError(f"Unexpected CoNLL-U filename: {name}")
    return match.group(1), match.group(2)


def extract_sent_id(sentence: Any) -> str:
    for comment in sentence.get_comments():
        if comment.startswith("sent_id ="):
            return comment.split("=", 1)[1].strip()
    raise ValueError(f"Missing sent_id for sentence idx={sentence.idx}")


def load_cache_embeddings(
    cache_dir: Path,
    treebank_filter: Optional[set[str]] = None,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    result: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ids_path in sorted(cache_dir.glob("*_ids.txt")):
        tb_code, split_name = parse_cache_name(ids_path)
        if treebank_filter is not None and tb_code not in treebank_filter:
            continue
        npy_path = cache_dir / f"{tb_code}-{split_name}.npy"
        if not npy_path.exists():
            continue
        sent_ids = [
            line.strip()
            for line in ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result[(tb_code, split_name)] = {
            "sent_id": sent_ids,
            "embedding": np.load(npy_path),
        }
    return result


def _import_original_modules(original_repo: Path):
    repo_str = str(original_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from cluster.gmm import run_gmm as original_run_gmm  # type: ignore
    from data.ud import (  # type: ignore
        UniversalDependencies,
        UniversalDependenciesIndexFilter,
    )

    return original_run_gmm, UniversalDependencies, UniversalDependenciesIndexFilter


def prepare_current_parity(paths: ParityAuditPaths) -> PreparedCurrentParity:
    cfg = load_config(paths.config_path)
    bootstrapper = GenreBootstrapper(cfg)
    paper_treebank_genre_map = resolve_paper_evaluation_treebank_genres(
        bootstrapper.data_loader
    )
    evaluation_treebank_ids = set(paper_treebank_genre_map)

    all_treebank_data = bootstrapper.data_loader.get_all_treebank_metadata()
    if cfg.exclude_treebanks:
        all_treebank_data = [
            tb for tb in all_treebank_data if tb["id"] not in cfg.exclude_treebanks
        ]

    test_split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])
    combined_split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])

    sentence_metadata: Dict[Tuple[str, str, str], str] = {}
    paper_test_split_keys_by_treebank: Dict[str, set[Tuple[str, str]]] = {}
    paper_anchor_split_keys_by_treebank: Dict[str, set[Tuple[str, str]]] = {}
    paper_test_metadata_counts_by_treebank: Dict[str, Dict[str, int]] = {}
    paper_test_sentence_counts_by_treebank: Dict[str, int] = {}
    paper_anchor_sentence_counts_by_treebank: Dict[str, int] = {}
    paper_treebank_languages: Dict[str, str] = {}

    genre_mapper = bootstrapper.genre_mapper

    for tb in all_treebank_data:
        tb_code = tb["id"]
        language = tb.get("language", tb_code.split("_", 1)[0])
        for split_name in bootstrapper.data_loader.get_available_splits(tb_code):
            if not test_split_map.includes_split(tb_code, split_name):
                continue

            split_key = (tb_code, split_name)
            paper_test_split_keys_by_treebank.setdefault(tb_code, set()).add(split_key)
            paper_anchor_split_keys_by_treebank.setdefault(tb_code, set()).add(split_key)
            paper_treebank_languages[tb_code] = language

            sentence_iter = bootstrapper.data_loader.iter_treebank_sentences(
                tb_code,
                split_name,
                metadata_only=True,
            )
            for idx, sentence in enumerate(sentence_iter):
                sent_id = sentence.get("sent_id", f"{tb_code}_{split_name}_{idx}")
                if not test_split_map.includes_sentence(tb_code, split_name, sent_id):
                    continue
                genres = genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                if not genres:
                    continue
                primary_genre = genres[0]
                sentence_metadata[(tb_code, split_name, sent_id)] = primary_genre
                tb_counts = paper_test_metadata_counts_by_treebank.setdefault(tb_code, {})
                tb_counts[primary_genre] = tb_counts.get(primary_genre, 0) + 1
                paper_test_sentence_counts_by_treebank[tb_code] = (
                    paper_test_sentence_counts_by_treebank.get(tb_code, 0) + 1
                )
                paper_anchor_sentence_counts_by_treebank[tb_code] = (
                    paper_anchor_sentence_counts_by_treebank.get(tb_code, 0) + 1
                )

    clustering_treebanks: List[Dict[str, Any]] = []
    scoring_treebanks: List[Dict[str, Any]] = []
    for tb_code, split_keys in sorted(paper_test_split_keys_by_treebank.items()):
        treebank_genres = sorted(bootstrapper.data_loader.get_treebank_genres(tb_code) or [])
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
            clustering_treebanks.append(descriptor)

        if tb_code in evaluation_treebank_ids:
            expected_genres = sorted(paper_treebank_genre_map.get(tb_code, []))
            if len(expected_genres) >= 2:
                scoring_descriptor = dict(descriptor)
                scoring_descriptor["genres"] = expected_genres
                scoring_treebanks.append(scoring_descriptor)

    single_anchor_treebanks: List[Dict[str, Any]] = []
    for tb_code, split_keys in sorted(paper_anchor_split_keys_by_treebank.items()):
        treebank_genres = list(bootstrapper.data_loader.get_treebank_genres(tb_code) or [])
        if len(treebank_genres) != 1:
            continue
        anchor_genre = treebank_genres[0]
        single_anchor_treebanks.append(
            {
                "treebank": tb_code,
                "split_keys": sorted(split_keys),
                "genres": [anchor_genre],
                "language": paper_treebank_languages.get(tb_code, tb_code.split("_", 1)[0]),
                "sentence_count": paper_anchor_sentence_counts_by_treebank.get(tb_code, 0),
                "genre_counts": {
                    anchor_genre: paper_anchor_sentence_counts_by_treebank.get(tb_code, 0)
                },
            }
        )

    treebank_ids_to_embed = sorted(
        {tb["treebank"] for tb in clustering_treebanks}.union(
            {tb["treebank"] for tb in single_anchor_treebanks}
        )
    )
    embeddings_by_tb = bootstrapper._generate_embeddings(
        treebank_filter=treebank_ids_to_embed
    )
    embeddings_by_tb, _ = filter_embeddings_by_sentence_split_map(
        embeddings_by_tb,
        combined_split_map,
    )
    available_embedding_splits = set(embeddings_by_tb.keys())
    clustering_treebanks = filter_treebank_descriptors_by_available_splits(
        clustering_treebanks,
        available_embedding_splits,
    )
    scoring_treebanks = filter_treebank_descriptors_by_available_splits(
        scoring_treebanks,
        available_embedding_splits,
    )
    single_anchor_treebanks = filter_treebank_descriptors_by_available_splits(
        single_anchor_treebanks,
        available_embedding_splits,
    )

    return PreparedCurrentParity(
        bootstrapper=bootstrapper,
        sentence_metadata=sentence_metadata,
        clustering_treebanks=clustering_treebanks,
        scoring_treebanks=scoring_treebanks,
        single_anchor_treebanks=single_anchor_treebanks,
        embeddings_by_tb=embeddings_by_tb,
        paper_treebank_genre_map=paper_treebank_genre_map,
    )


def run_current_with_tracking(
    prepared: PreparedCurrentParity,
    paths: ParityAuditPaths,
) -> Dict[str, Any]:
    cfg = load_config(paths.config_path)
    evaluator = ClusteringEvaluator(
        n_folds=cfg.evaluation.metadata_validation.k,
        group_by=cfg.evaluation.metadata_validation.group_by,
        random_state=cfg.clustering.seed,
        min_confidence=cfg.bootstrapping.min_confidence,
        min_margin=cfg.bootstrapping.min_margin,
        max_iterations=cfg.bootstrapping.max_iterations,
        anchor_mode=cfg.evaluation.metadata_validation.anchor_mode,
        anchor_pool_policy=cfg.evaluation.metadata_validation.anchor_pool_policy,
        reference_weighting=cfg.bootstrapping.reference_weighting,
        protocol=cfg.evaluation.metadata_validation.protocol,
    )

    sentence_metadata, embeddings_by_tb = evaluator._qualify_evaluation_inputs(
        prepared.sentence_metadata,
        prepared.embeddings_by_tb,
    )

    genre_combination_clusters: Dict[
        Tuple[str, ...], Dict[Tuple[str, str], List[Dict[str, Any]]]
    ] = defaultdict(dict)
    anchor_counts_by_genre: Dict[str, int] = defaultdict(int)
    single_anchor_counts = evaluator._add_treebank_single_genre_anchors(
        treebanks=prepared.single_anchor_treebanks,
        source_tag="__single_anchor__",
        embeddings_by_tb=embeddings_by_tb,
        genre_combination_clusters=genre_combination_clusters,
    )
    for genre, count in single_anchor_counts.items():
        anchor_counts_by_genre[genre] += count

    treebank_genres_map: Dict[str, set[str]] = defaultdict(set)
    for tb in prepared.clustering_treebanks:
        treebank_genres_map[tb["treebank"]].update(tb.get("genres", []))

    clustering_split_keys = list(
        dict.fromkeys(
            collect_treebank_descriptor_split_keys(prepared.clustering_treebanks)
        )
    )
    scoring_split_keys = set(
        collect_treebank_descriptor_split_keys(prepared.scoring_treebanks)
    )
    scoring_treebank_ids = {tb["treebank"] for tb in prepared.scoring_treebanks}
    test_treebank_groups = evaluator.clustering_ops.group_splits_by_treebank(
        clustering_split_keys,
        embeddings_by_tb,
    )

    current_cluster_map: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    current_cluster_sent_refs: Dict[str, Dict[int, List[str]]] = defaultdict(dict)
    test_sentence_batches = []
    for tb_code, tb_keys in sorted(test_treebank_groups.items()):
        combined_embeddings, all_sent_ids_list, sent_id_to_split = (
            evaluator.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
        )
        expected_genres = sorted(treebank_genres_map.get(tb_code, []))
        n_genres = max(1, len(expected_genres))
        cluster_result = prepared.bootstrapper.clusterer.cluster_treebank(
            embeddings=combined_embeddings,
            sent_ids=all_sent_ids_list,
            n_genres=n_genres,
            compute_metrics=False,
        )
        cluster_ids = cluster_result["cluster_ids"]
        clusters = cluster_result["clusters"]
        cluster_centroids = evaluator.clustering_ops.compute_cluster_centroids(
            cluster_ids,
            combined_embeddings,
            n_genres,
        )
        cluster_descriptors = []
        for cluster_id, centroid in sorted(cluster_centroids.items()):
            cluster_sent_ids = list(clusters.get(cluster_id, {}).get("sent_ids", []))
            cluster_descriptors.append(
                {
                    "cluster_id": cluster_id,
                    "embedding": centroid,
                    "sent_ids": cluster_sent_ids,
                }
            )
            current_cluster_sent_refs[tb_code][cluster_id] = cluster_sent_ids
            current_cluster_map[tb_code][cluster_id] = {
                "initial_sent_count": len(cluster_sent_ids),
                "expected_genres": expected_genres,
            }
        genre_combination_clusters[tuple(expected_genres)][
            (tb_code, "combined")
        ] = cluster_descriptors
        test_sentence_batches.append(
            {
                "tb_code": tb_code,
                "sent_ids": all_sent_ids_list,
                "sent_id_to_split": sent_id_to_split,
            }
        )

    schedule = evaluator.scheduler.create_schedule(set(genre_combination_clusters.keys()))
    final_labels, env_summaries = evaluator.clustering_ops.run_bootstrap_schedule(
        schedule=schedule,
        genre_combination_clusters=genre_combination_clusters,
        final_labels={},
        preserve_methods=None,
    )
    annotate_current_cluster_map(
        current_cluster_map=current_cluster_map,
        current_cluster_sent_refs=current_cluster_sent_refs,
        final_labels=final_labels,
    )

    sentence_records = []
    for batch in test_sentence_batches:
        tb_code = batch["tb_code"]
        for sent_ref in batch["sent_ids"]:
            split_name = batch["sent_id_to_split"][sent_ref]
            ref_tb_code, ref_split_name, raw_sent_id = evaluator._extract_sentence_ref_parts(
                sent_ref,
                tb_code=tb_code,
                split_name=split_name,
            )
            if (ref_tb_code, ref_split_name) not in scoring_split_keys:
                continue
            pred_label = final_labels.get(sent_ref)
            if pred_label is None:
                continue
            true_genre = sentence_metadata.get((ref_tb_code, ref_split_name, sent_ref))
            if true_genre is None:
                continue
            sentence_records.append(
                {
                    "treebank": ref_tb_code,
                    "split": ref_split_name,
                    "sent_id": raw_sent_id,
                    "true": true_genre,
                    "pred": pred_label[0],
                    "method": pred_label[2],
                }
            )

    filtered_cluster_map = {
        tb: {str(cid): data for cid, data in sorted(cluster_data.items())}
        for tb, cluster_data in sorted(current_cluster_map.items())
        if tb in scoring_treebank_ids
    }

    return {
        "cluster_map": filtered_cluster_map,
        "sentence_records": sentence_records,
        "anchors_by_genre": dict(sorted(anchor_counts_by_genre.items())),
        "missing_anchor_genres": sorted(
            {
                genre
                for tb in prepared.clustering_treebanks
                for genre in tb.get("genres", [])
            }
            - set(anchor_counts_by_genre.keys())
        ),
        "clustering_treebanks": [tb["treebank"] for tb in prepared.clustering_treebanks],
        "scoring_treebanks": [tb["treebank"] for tb in prepared.scoring_treebanks],
        "env_summaries": env_summaries,
    }


def annotate_current_cluster_map(
    current_cluster_map: Dict[str, Dict[int, Dict[str, Any]]],
    current_cluster_sent_refs: Dict[str, Dict[int, List[str]]],
    final_labels: Dict[str, Tuple[str, float, str]],
) -> None:
    """Annotate current-side cluster records with final schedule labels.

    Each clustered sentence should receive one final label tuple from the
    shared bootstrap schedule. For audit purposes we surface that label at the
    cluster level and retain any unexpected within-cluster variation.
    """

    for tb_code, clusters in current_cluster_map.items():
        for cluster_id, cluster_info in clusters.items():
            sent_refs = current_cluster_sent_refs.get(tb_code, {}).get(cluster_id, [])
            assigned = [
                final_labels[sent_ref]
                for sent_ref in sent_refs
                if sent_ref in final_labels
            ]
            if not assigned:
                continue

            label_counts = Counter(label for label, _confidence, _method in assigned)
            dominant_label, _ = label_counts.most_common(1)[0]
            dominant_assignments = [
                label_tuple for label_tuple in assigned if label_tuple[0] == dominant_label
            ]
            best_assignment = max(
                dominant_assignments,
                key=lambda label_tuple: label_tuple[1],
            )
            cluster_info.update(
                {
                    "label": dominant_label,
                    "confidence": float(best_assignment[1]),
                    "method": best_assignment[2],
                    "labeled_sent_count": len(assigned),
                }
            )
            if len(label_counts) > 1:
                cluster_info["label_counts"] = dict(sorted(label_counts.items()))


def build_original_idx_maps(
    ud: Any,
) -> Tuple[Dict[Tuple[str, str, str], int], Dict[int, Tuple[str, str, str]]]:
    key_to_idx: Dict[Tuple[str, str, str], int] = {}
    idx_to_key: Dict[int, Tuple[str, str, str]] = {}
    for _tb_label, sentences in ud.get_sentences_by_treebank():
        valid = [sentence for sentence in sentences if sentence is not None]
        if not valid:
            continue
        for sentence in valid:
            tb_code, split_name = parse_conllu_file_name(
                ud.get_treebank_file_of_index(sentence.idx)
            )
            try:
                sent_id = extract_sent_id(sentence)
            except ValueError:
                continue
            key = (tb_code, split_name, sent_id)
            key_to_idx[key] = sentence.idx
            idx_to_key[sentence.idx] = key
    return key_to_idx, idx_to_key


def build_original_embedding_lookup(
    idx_lookup: Dict[Tuple[str, str, str], int],
    paths: ParityAuditPaths,
) -> Dict[int, np.ndarray]:
    test_split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])
    cache_embeddings = load_cache_embeddings(paths.cache_dir)
    emb_by_idx: Dict[int, np.ndarray] = {}
    for split_key, allowed_sent_ids in test_split_map.split_to_sent_ids.items():
        emb_data = cache_embeddings.get(split_key)
        if emb_data is None:
            continue
        tb_code, split_name = split_key
        for row_idx, sent_id in enumerate(emb_data["sent_id"]):
            if sent_id not in allowed_sent_ids:
                continue
            global_idx = idx_lookup.get((tb_code, split_name, sent_id))
            if global_idx is None:
                continue
            emb_by_idx[global_idx] = emb_data["embedding"][row_idx]
    return emb_by_idx


def run_original_with_tracking(
    prepared: PreparedCurrentParity,
    paths: ParityAuditPaths,
) -> Dict[str, Any]:
    original_run_gmm, _UniversalDependencies, _UniversalDependenciesIndexFilter = _import_original_modules(
        paths.original_repo
    )
    split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])
    cache_embeddings = load_cache_embeddings(paths.cache_dir)
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for split_key in sorted(split_map.split_keys):
        grouped[split_key[0]].append(split_key)

    genre_combination_clusters: Dict[
        Tuple[str, ...], Dict[str, List[Dict[str, Any]]]
    ] = defaultdict(dict)
    original_cluster_map: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    anchor_counts: Dict[str, int] = defaultdict(int)

    for tb_code, split_keys in sorted(grouped.items()):
        genres = tuple(
            sorted(prepared.bootstrapper.data_loader.get_treebank_genres(tb_code) or [])
        )
        if not genres:
            continue

        sent_refs: List[Tuple[str, str, str]] = []
        emb_rows: List[np.ndarray] = []
        for split_key in split_keys:
            emb_data = cache_embeddings.get(split_key)
            if emb_data is None:
                continue
            allowed_sent_ids = split_map.split_to_sent_ids[split_key]
            for row_idx, sent_id in enumerate(emb_data["sent_id"]):
                if sent_id not in allowed_sent_ids:
                    continue
                sent_refs.append((split_key[0], split_key[1], sent_id))
                emb_rows.append(emb_data["embedding"][row_idx])
        if not emb_rows:
            continue

        emb = np.asarray(np.vstack(emb_rows), dtype=np.float64)
        if len(genres) < 2:
            assignments = np.zeros(len(sent_refs), dtype=int)
            cluster_count = 1
        else:
            probs = original_run_gmm(emb, len(genres), random_state=42)
            assignments = probs.argmax(axis=-1)
            cluster_count = probs.shape[1]

        cluster_rows: List[List[int]] = [[] for _ in range(cluster_count)]
        for row_idx, _sent_ref in enumerate(sent_refs):
            cluster_rows[int(assignments[row_idx])].append(row_idx)

        genre_combination_clusters[genres][tb_code] = []
        for cluster_id, row_indices in enumerate(cluster_rows):
            refs = [sent_refs[row_idx] for row_idx in row_indices]
            centroid = np.mean(np.vstack([emb_rows[row_idx] for row_idx in row_indices]), axis=0)
            genre_combination_clusters[genres][tb_code].append(
                {
                    "refs": refs,
                    "emb": centroid,
                    "orig_cluster_id": cluster_id,
                }
            )
            original_cluster_map[tb_code][cluster_id] = {
                "initial_sent_count": len(refs),
                "expected_genres": list(genres),
            }
        if len(genres) == 1:
            anchor_counts[genres[0]] += 1

    schedule = original_get_schedule(list(genre_combination_clusters.keys()))
    schedule_new_genres: List[str] = []
    known_so_far = {combo[0] for combo in genre_combination_clusters if len(combo) == 1}
    for environment in schedule:
        new_now = sorted(set(environment["known"]) - known_so_far)
        if new_now:
            schedule_new_genres.extend(new_now)
        known_so_far.update(environment["known"])

    final_by_ref: Dict[Tuple[str, str, str], str] = {}
    emb_dim = next(iter(next(iter(genre_combination_clusters.values())).values()))[0]["emb"].shape[0]
    for env_idx, environment in enumerate(schedule, start=1):
        known_embeddings = np.zeros((len(environment["known"]), emb_dim))
        for genre_idx, known_genre in enumerate(environment["known"]):
            known_embeddings[genre_idx] = np.mean(
                [
                    cluster["emb"]
                    for _tb_code, clusters in genre_combination_clusters[(known_genre,)].items()
                    for cluster in clusters
                ],
                axis=0,
            )

        for genre_combination in list(genre_combination_clusters.keys()):
            if len(genre_combination) < 2:
                continue
            predictable_genres = []
            predictable_genre_indices = []
            for genre_idx, known_genre in enumerate(environment["known"]):
                if known_genre in genre_combination:
                    predictable_genres.append(known_genre)
                    predictable_genre_indices.append(genre_idx)
            unresolved_genres = list(set(genre_combination) - set(predictable_genres))
            if not predictable_genres:
                continue
            rel_known_embeddings = known_embeddings[predictable_genre_indices, :]

            for tb_code, tb_clusters in list(genre_combination_clusters[genre_combination].items()):
                tb_cluster_embeddings = np.array([cluster["emb"] for cluster in tb_clusters])
                cos_distances = distance.cdist(tb_cluster_embeddings, rel_known_embeddings, "cosine")
                unlabeled_clusters = list(range(len(tb_clusters)))
                unassigned_genres = list(range(len(predictable_genre_indices)))

                while unassigned_genres:
                    rel_cluster_indices = np.array(unlabeled_clusters)
                    rel_genre_indices = np.array(unassigned_genres)
                    rel_distances = cos_distances[rel_cluster_indices[:, None], rel_genre_indices]
                    rel_cluster_idx, rel_genre_idx = np.unravel_index(
                        np.argmin(rel_distances), rel_distances.shape
                    )
                    closest_cluster_idx = unlabeled_clusters[rel_cluster_idx]
                    closest_genre_idx = unassigned_genres[rel_genre_idx]
                    cluster = tb_clusters[closest_cluster_idx]
                    genre = predictable_genres[closest_genre_idx]
                    original_cluster_map[tb_code][cluster["orig_cluster_id"]].update(
                        {
                            "label": genre,
                            "confidence": float(1.0 - cos_distances[closest_cluster_idx, closest_genre_idx]),
                            "method": "match",
                            "assigned_in_env": env_idx,
                            "via": "greedy_min_cosine",
                        }
                    )
                    for sent_ref in cluster["refs"]:
                        final_by_ref[sent_ref] = genre
                    genre_combination_clusters[(genre,)][tb_code] = (
                        genre_combination_clusters[(genre,)].get(tb_code, []) + [cluster]
                    )
                    unlabeled_clusters.remove(closest_cluster_idx)
                    unassigned_genres.remove(closest_genre_idx)

                if len(unresolved_genres) == 1 and len(unlabeled_clusters) == 1:
                    inferred_cluster_idx = unlabeled_clusters[0]
                    cluster = tb_clusters[inferred_cluster_idx]
                    inferred_genre = unresolved_genres[0]
                    original_cluster_map[tb_code][cluster["orig_cluster_id"]].update(
                        {
                            "label": inferred_genre,
                            "confidence": 0.0,
                            "method": "inferred",
                            "assigned_in_env": env_idx,
                            "via": "infer_last",
                        }
                    )
                    for sent_ref in cluster["refs"]:
                        final_by_ref[sent_ref] = inferred_genre
                    genre_combination_clusters[(inferred_genre,)][tb_code] = (
                        genre_combination_clusters[(inferred_genre,)].get(tb_code, []) + [cluster]
                    )
                    unlabeled_clusters.remove(inferred_cluster_idx)

                if unlabeled_clusters:
                    unpred_combination = tuple(sorted(unresolved_genres))
                    genre_combination_clusters[unpred_combination][tb_code] = [
                        cluster
                        for cluster_idx, cluster in enumerate(tb_clusters)
                        if cluster_idx in unlabeled_clusters
                    ]
                del genre_combination_clusters[genre_combination][tb_code]

            if not genre_combination_clusters.get(genre_combination):
                genre_combination_clusters.pop(genre_combination, None)

    scoring_split_keys = set(
        collect_treebank_descriptor_split_keys(prepared.scoring_treebanks)
    )
    scoring_treebank_ids = {tb["treebank"] for tb in prepared.scoring_treebanks}
    sentence_records = []
    for sent_ref, pred in final_by_ref.items():
        tb_code, split_name, sent_id = sent_ref
        if (tb_code, split_name) not in scoring_split_keys:
            continue
        sentence_records.append(
            {
                "treebank": tb_code,
                "split": split_name,
                "sent_id": sent_id,
                "pred": pred,
            }
        )

    filtered_cluster_map = {
        tb: {str(cid): data for cid, data in sorted(cluster_data.items())}
        for tb, cluster_data in sorted(original_cluster_map.items())
        if tb in scoring_treebank_ids
    }
    return {
        "cluster_map": filtered_cluster_map,
        "sentence_records": sentence_records,
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "test_treebanks_seen": sorted(grouped.keys()),
        "schedule_new_genres": schedule_new_genres,
        "source": "original_like_cached_embeddings",
    }


def summarize_per_treebank(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_treebank: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_treebank[record["treebank"]].append(record)

    summary: Dict[str, Dict[str, Any]] = {}
    for tb_code, tb_records in sorted(by_treebank.items()):
        true = [record["true"] for record in tb_records]
        pred = [record["pred"] for record in tb_records]
        summary[tb_code] = {
            "n": len(tb_records),
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(
                f1_score(true, pred, average="macro", zero_division=0)
            ),
            "gold_counts": dict(sorted(Counter(true).items())),
            "pred_counts": dict(sorted(Counter(pred).items())),
        }
    return summary


def compare_systems(
    current_records: List[Dict[str, Any]],
    original_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    current_by_key = {
        (record["treebank"], record["split"], record["sent_id"]): record
        for record in current_records
    }
    original_by_key = {
        (record["treebank"], record["split"], record["sent_id"]): record
        for record in original_records
    }
    shared_keys = sorted(set(current_by_key) & set(original_by_key))
    by_treebank: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for key in shared_keys:
        by_treebank[key[0]].append((current_by_key[key], original_by_key[key]))

    report = {}
    for tb_code, pairs in sorted(by_treebank.items()):
        same = sum(1 for current, original in pairs if current["pred"] == original["pred"])
        disagreements = []
        for current, original in pairs:
            if current["pred"] == original["pred"]:
                continue
            disagreements.append(
                {
                    "split": current["split"],
                    "sent_id": current["sent_id"],
                    "true": current["true"],
                    "current": current["pred"],
                    "original": original["pred"],
                }
            )
            if len(disagreements) >= 15:
                break
        report[tb_code] = {
            "shared_sentences": len(pairs),
            "prediction_agreement": same / len(pairs) if pairs else 0.0,
            "sample_disagreements": disagreements,
        }
    return report


def build_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Original vs Current Paper-Parity Comparison",
        "",
        "## Scope",
        f"- Current clustering treebanks: {', '.join(report['current']['clustering_treebanks'])}",
        f"- Current scored paper treebanks: {', '.join(report['current']['scoring_treebanks'])}",
        f"- Original full test-partition treebanks seen: {len(report['original']['test_treebanks_seen'])}",
        f"- Current missing anchor genres: {', '.join(report['current']['missing_anchor_genres']) or 'none'}",
        f"- Original single-genre anchor genres: {json.dumps(report['original']['anchor_counts'], ensure_ascii=False, sort_keys=True)}",
        f"- Original schedule learned later genres: {', '.join(report['original']['schedule_new_genres']) or 'none'}",
        "",
        "## Per-Treebank",
    ]
    for tb_code in sorted(report["comparison"]):
        current_tb = report["current"]["per_treebank"].get(tb_code, {})
        original_tb = report["original"]["per_treebank"].get(tb_code, {})
        comparison_tb = report["comparison"][tb_code]
        lines.append(f"### {tb_code}")
        lines.append(
            f"- Current: acc={current_tb.get('accuracy', 0):.4f}, macro_f1={current_tb.get('macro_f1', 0):.4f}, pred_counts={current_tb.get('pred_counts', {})}"
        )
        lines.append(
            f"- Original: acc={original_tb.get('accuracy', 0):.4f}, macro_f1={original_tb.get('macro_f1', 0):.4f}, pred_counts={original_tb.get('pred_counts', {})}"
        )
        lines.append(
            f"- Current vs original prediction agreement: {comparison_tb['prediction_agreement']:.4f} over {comparison_tb['shared_sentences']} shared sentences"
        )
        lines.append(
            f"- Current cluster labels: {report['current']['cluster_map'].get(tb_code, {})}"
        )
        lines.append(
            f"- Original cluster labels: {report['original']['cluster_map'].get(tb_code, {})}"
        )
        if comparison_tb["sample_disagreements"]:
            lines.append("- Sample disagreements:")
            for row in comparison_tb["sample_disagreements"]:
                lines.append(
                    f"  - {row['split']}:{row['sent_id']} true={row['true']} current={row['current']} original={row['original']}"
                )
        lines.append("")
    return "\n".join(lines)


def run_parity_audit(paths: ParityAuditPaths) -> Dict[str, Any]:
    prepared = prepare_current_parity(paths)
    current_result = run_current_with_tracking(prepared, paths)
    original_result = run_original_like_with_tracking(prepared, paths)

    current_records = current_result["sentence_records"]
    original_records = []
    truth_map = {
        (record["treebank"], record["split"], record["sent_id"]): record["true"]
        for record in current_records
    }
    for record in original_result["sentence_records"]:
        true = truth_map.get((record["treebank"], record["split"], record["sent_id"]))
        if true is None:
            continue
        updated = dict(record)
        updated["true"] = true
        original_records.append(updated)

    return {
        "paths": {
            "config": str(paths.config_path),
            "split_map": str(paths.split_map_path),
            "original_repo": str(paths.original_repo),
            "ud_root": str(paths.ud_root),
            "cache_dir": str(paths.cache_dir),
        },
        "current": {
            **current_result,
            "per_treebank": summarize_per_treebank(current_records),
        },
        "original": {
            **original_result,
            "per_treebank": summarize_per_treebank(original_records),
        },
        "comparison": compare_systems(current_records, original_records),
    }


def write_parity_audit(report: Dict[str, Any], paths: ParityAuditPaths) -> None:
    paths.json_out.parent.mkdir(parents=True, exist_ok=True)
    paths.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    paths.json_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths.markdown_out.write_text(build_markdown(report), encoding="utf-8")


def run_original_like_with_tracking(
    prepared: PreparedCurrentParity,
    paths: ParityAuditPaths,
) -> Dict[str, Any]:
    original_run_gmm, _UniversalDependencies, _UniversalDependenciesIndexFilter = _import_original_modules(
        paths.original_repo
    )
    split_map = load_sentence_split_map(paths.split_map_path, partitions=["test"])
    cache_embeddings = load_cache_embeddings(paths.cache_dir)
    grouped: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for split_key in sorted(split_map.split_keys):
        grouped[split_key[0]].append(split_key)

    genre_combination_clusters: Dict[
        Tuple[str, ...], Dict[str, List[Dict[str, Any]]]
    ] = defaultdict(dict)
    original_cluster_map: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    anchor_counts: Dict[str, int] = defaultdict(int)

    for tb_code, split_keys in sorted(grouped.items()):
        genres = tuple(sorted(prepared.bootstrapper.data_loader.get_treebank_genres(tb_code) or []))
        if not genres:
            continue
        sent_refs: List[Tuple[str, str, str]] = []
        emb_rows: List[np.ndarray] = []
        for split_key in split_keys:
            emb_data = cache_embeddings.get(split_key)
            if emb_data is None:
                continue
            allowed_sent_ids = split_map.split_to_sent_ids[split_key]
            for row_idx, sent_id in enumerate(emb_data["sent_id"]):
                if sent_id not in allowed_sent_ids:
                    continue
                sent_refs.append((split_key[0], split_key[1], sent_id))
                emb_rows.append(emb_data["embedding"][row_idx])
        if not emb_rows:
            continue

        emb = np.asarray(np.vstack(emb_rows), dtype=np.float64)
        if len(genres) < 2:
            assignments = np.zeros(len(sent_refs), dtype=int)
            cluster_count = 1
        else:
            probs = original_run_gmm(emb, len(genres), random_state=42)
            assignments = probs.argmax(axis=-1)
            cluster_count = probs.shape[1]

        cluster_refs: List[List[Tuple[str, str, str]]] = [[] for _ in range(cluster_count)]
        for row_idx, sent_ref in enumerate(sent_refs):
            cluster_refs[int(assignments[row_idx])].append(sent_ref)

        genre_combination_clusters[genres][tb_code] = []
        for cluster_id, refs in enumerate(cluster_refs):
            centroid = np.mean(
                np.vstack([emb_rows[sent_refs.index(ref)] for ref in refs]),
                axis=0,
            )
            genre_combination_clusters[genres][tb_code].append(
                {
                    "refs": refs,
                    "emb": centroid,
                    "orig_cluster_id": cluster_id,
                }
            )
            original_cluster_map[tb_code][cluster_id] = {
                "initial_sent_count": len(refs),
                "expected_genres": list(genres),
            }
        if len(genres) == 1:
            anchor_counts[genres[0]] += 1

    schedule = original_get_schedule(list(genre_combination_clusters.keys()))
    schedule_new_genres: List[str] = []
    known_so_far = {combo[0] for combo in genre_combination_clusters if len(combo) == 1}
    for environment in schedule:
        new_now = sorted(set(environment["known"]) - known_so_far)
        if new_now:
            schedule_new_genres.extend(new_now)
        known_so_far.update(environment["known"])

    final_by_ref: Dict[Tuple[str, str, str], str] = {}
    emb_dim = next(iter(next(iter(genre_combination_clusters.values())).values()))[0]["emb"].shape[0]
    for env_idx, environment in enumerate(schedule, start=1):
        known_embeddings = np.zeros((len(environment["known"]), emb_dim))
        for genre_idx, known_genre in enumerate(environment["known"]):
            known_embeddings[genre_idx] = np.mean(
                [
                    cluster["emb"]
                    for _tb_code, clusters in genre_combination_clusters[(known_genre,)].items()
                    for cluster in clusters
                ],
                axis=0,
            )

        for genre_combination in list(genre_combination_clusters.keys()):
            if len(genre_combination) < 2:
                continue
            predictable_genres = []
            predictable_genre_indices = []
            for genre_idx, known_genre in enumerate(environment["known"]):
                if known_genre in genre_combination:
                    predictable_genres.append(known_genre)
                    predictable_genre_indices.append(genre_idx)
            unresolved_genres = list(set(genre_combination) - set(predictable_genres))
            if not predictable_genres:
                continue
            rel_known_embeddings = known_embeddings[predictable_genre_indices, :]

            for tb_code, tb_clusters in list(genre_combination_clusters[genre_combination].items()):
                tb_cluster_embeddings = np.array([cluster["emb"] for cluster in tb_clusters])
                cos_distances = distance.cdist(tb_cluster_embeddings, rel_known_embeddings, "cosine")
                unlabeled_clusters = list(range(len(tb_clusters)))
                unassigned_genres = list(range(len(predictable_genre_indices)))
                while unassigned_genres:
                    rel_cluster_indices = np.array(unlabeled_clusters)
                    rel_genre_indices = np.array(unassigned_genres)
                    rel_distances = cos_distances[rel_cluster_indices[:, None], rel_genre_indices]
                    rel_cluster_idx, rel_genre_idx = np.unravel_index(
                        np.argmin(rel_distances), rel_distances.shape
                    )
                    closest_cluster_idx = unlabeled_clusters[rel_cluster_idx]
                    closest_genre_idx = unassigned_genres[rel_genre_idx]
                    cluster = tb_clusters[closest_cluster_idx]
                    genre = predictable_genres[closest_genre_idx]
                    original_cluster_map[tb_code][cluster["orig_cluster_id"]].update(
                        {
                            "label": genre,
                            "confidence": float(1.0 - cos_distances[closest_cluster_idx, closest_genre_idx]),
                            "method": "match",
                            "assigned_in_env": env_idx,
                            "via": "greedy_min_cosine",
                        }
                    )
                    for sent_ref in cluster["refs"]:
                        final_by_ref[sent_ref] = genre
                    genre_combination_clusters[(genre,)][tb_code] = (
                        genre_combination_clusters[(genre,)].get(tb_code, []) + [cluster]
                    )
                    unlabeled_clusters.remove(closest_cluster_idx)
                    unassigned_genres.remove(closest_genre_idx)

                if len(unresolved_genres) == 1 and len(unlabeled_clusters) == 1:
                    inferred_cluster_idx = unlabeled_clusters[0]
                    cluster = tb_clusters[inferred_cluster_idx]
                    inferred_genre = unresolved_genres[0]
                    original_cluster_map[tb_code][cluster["orig_cluster_id"]].update(
                        {
                            "label": inferred_genre,
                            "confidence": 0.0,
                            "method": "inferred",
                            "assigned_in_env": env_idx,
                            "via": "infer_last",
                        }
                    )
                    for sent_ref in cluster["refs"]:
                        final_by_ref[sent_ref] = inferred_genre
                    genre_combination_clusters[(inferred_genre,)][tb_code] = (
                        genre_combination_clusters[(inferred_genre,)].get(tb_code, []) + [cluster]
                    )
                    unlabeled_clusters.remove(inferred_cluster_idx)

                if unlabeled_clusters:
                    unpred_combination = tuple(sorted(unresolved_genres))
                    genre_combination_clusters[unpred_combination][tb_code] = [
                        cluster
                        for cluster_idx, cluster in enumerate(tb_clusters)
                        if cluster_idx in unlabeled_clusters
                    ]
                del genre_combination_clusters[genre_combination][tb_code]

            if not genre_combination_clusters.get(genre_combination):
                genre_combination_clusters.pop(genre_combination, None)

    scoring_split_keys = set(
        collect_treebank_descriptor_split_keys(prepared.scoring_treebanks)
    )
    scoring_treebank_ids = {tb["treebank"] for tb in prepared.scoring_treebanks}
    sentence_records = []
    for sent_ref, pred in final_by_ref.items():
        tb_code, split_name, sent_id = sent_ref
        if (tb_code, split_name) not in scoring_split_keys:
            continue
        sentence_records.append(
            {
                "treebank": tb_code,
                "split": split_name,
                "sent_id": sent_id,
                "pred": pred,
            }
        )

    filtered_cluster_map = {
        tb: {str(cid): data for cid, data in sorted(cluster_data.items())}
        for tb, cluster_data in sorted(original_cluster_map.items())
        if tb in scoring_treebank_ids
    }
    return {
        "cluster_map": filtered_cluster_map,
        "sentence_records": sentence_records,
        "anchor_counts": dict(sorted(anchor_counts.items())),
        "test_treebanks_seen": sorted(grouped.keys()),
        "schedule_new_genres": schedule_new_genres,
        "source": "original_like_cached_embeddings",
    }
