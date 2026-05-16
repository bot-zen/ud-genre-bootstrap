"""Main bootstrapping class for genre labeling."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ud_genre_bootstrap.bootstrapping.scheduler import BootstrapScheduler
from ud_genre_bootstrap.clustering.clustering_utils import ClusteringOperations
from ud_genre_bootstrap.clustering.gmm_clusterer import GMMClusterer
from ud_genre_bootstrap.clustering.kmeans_clusterer import KMeansClusterer
from ud_genre_bootstrap.embeddings.generator import EmbeddingGenerator
from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.data_loader import UDDataLoader
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper
from ud_genre_bootstrap.utils.release_artifacts import (
    build_release_row_metadata,
    prepare_release_directory,
    resolve_config_name,
    summarize_exported_labels_file,
    upload_release_directory_to_hub,
    write_release_artifacts,
)
from ud_genre_bootstrap.utils.sentence_refs import (
    extract_sentence_ref_parts,
    qualify_embeddings_for_split,
    qualify_sentence_ref,
)

logger = logging.getLogger(__name__)
COMBINED_SPLIT_KEY = "__combined__"


class GenreBootstrapper:
    """Main class coordinating the full bootstrapping pipeline."""

    def __init__(self, config: Config):
        """Initialize bootstrapper with configuration.

        Args:
            config: Configuration object
        """
        self.config = config

        # Initialize components
        self.data_loader = UDDataLoader(
            ud_source=config.ud_source,
            ud_version=config.ud_version,
            metadata_path=Path(config.metadata_path) if config.metadata_path else None,
        )

        # Initialize genre mapper with configuration
        from pathlib import Path as PathLib
        mapping_path = None
        patterns_path = None
        if config.genre_extraction.mapping_path:
            mapping_path = PathLib(config.genre_extraction.mapping_path)
        if config.genre_extraction.patterns_path:
            if isinstance(config.genre_extraction.patterns_path, list):
                patterns_path = [PathLib(p) for p in config.genre_extraction.patterns_path]
            else:
                patterns_path = PathLib(config.genre_extraction.patterns_path)

        self.genre_mapper = GenreMapper(
            genre_mapping_path=mapping_path,
            metadata_patterns_path=patterns_path,
            canonical_genres=config.genre_extraction.canonical_genres,
            data_loader=self.data_loader,
        )

        self.embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings.model,
            pooling=config.embeddings.pooling,
            layer=config.embeddings.layer,
            batch_size=config.embeddings.batch_size,
            device=config.embeddings.device,
        )

        # Select clusterer based on configuration
        clustering_method = config.clustering.method.lower()
        if clustering_method == "kmeans":
            logger.info("Using K-Means clustering")
            self.clusterer = KMeansClusterer(
                random_state=config.clustering.seed,
                device=config.clustering.device,
                max_iter=config.clustering.max_iter,
            )
        elif clustering_method == "gmm":
            logger.info("Using GMM clustering")
            self.clusterer = GMMClusterer(
                random_state=config.clustering.seed,
                device=config.clustering.device,
                max_iter=config.clustering.max_iter,
                fit_sample_size=config.clustering.fit_sample_size,
                reg_covar=config.clustering.reg_covar,
            )
        else:
            raise ValueError(
                f"Unknown clustering method: {clustering_method}. "
                f"Supported methods: 'gmm', 'kmeans'"
            )

        self.scheduler = BootstrapScheduler(
            max_iterations=config.bootstrapping.max_iterations,
        )

        # Initialize shared clustering operations
        self.clustering_ops = ClusteringOperations(
            min_confidence=config.bootstrapping.min_confidence,
            min_margin=config.bootstrapping.min_margin,
            reference_weighting=config.bootstrapping.reference_weighting,
        )

        # Storage for results
        self.treebank_clusters: Dict = {}  # {treebank_code: cluster_info}
        self.genre_combination_clusters: Dict = defaultdict(dict)
        self.final_labels: Dict = {}  # {(treebank, split, sent_id): (genre, confidence, method)}
        self._last_embeddings_by_tb: Dict = {}

    def fit(self, treebank_filter: Optional[List[str]] = None) -> Dict:
        """Run the full bootstrapping pipeline.

        Args:
            treebank_filter: Optional list of treebank codes to process

        Returns:
            Dictionary with results and statistics
        """
        logger.info("Starting bootstrap genre classification pipeline")

        # Step 1: Embed all treebanks
        logger.info("Step 1: Generating embeddings")
        embeddings_by_tb = self._generate_embeddings(treebank_filter=treebank_filter)

        # Step 2: Cluster each treebank
        logger.info("Step 2: Clustering treebanks")
        self._cluster_treebanks(embeddings_by_tb)

        # Step 3: Compute cluster embeddings and group by genre combinations
        logger.info("Step 3: Computing cluster embeddings")
        self._compute_cluster_embeddings(embeddings_by_tb)

        # Step 3.5: Compute genre separation metrics
        logger.info("Step 3.5: Computing genre separation metrics")
        self._compute_genre_separation_metrics()

        # Steps 4-6.5: Schedule + labeling + reporting
        self.execute_bootstrap_labeling()

        # Step 7: Export results
        logger.info("Step 7: Exporting results")
        results = self._export_results()

        logger.info("Bootstrap pipeline complete")
        return results

    def execute_bootstrap_labeling(
        self,
        schedule: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Run scheduling and bootstrap labeling stages.

        This centralizes the shared stage logic used by both the full `fit()`
        pipeline and the CLI `label` command.

        Args:
            schedule: Optional precomputed schedule. If omitted, it is created.

        Returns:
            Bootstrap schedule that was used for labeling.
        """
        if schedule is None:
            logger.info("Step 4: Creating bootstrap schedule")
            schedule = self._create_schedule()

        logger.info("Step 5: Labeling single-genre treebanks")
        self._label_single_genre_treebanks()

        logger.info("Step 6: Labeling multi-genre clusters via bootstrap")
        self._label_clusters(schedule)

        logger.info("Step 6.5: Generating cross-lingual assignment report")
        self._generate_cross_lingual_report()

        return schedule

    def load_cluster_state(self, cluster_state_path: Path) -> Dict:
        """Load pre-computed cluster state from disk.

        This allows the label command to skip the expensive clustering step
        and directly use previously computed clusters.

        Args:
            cluster_state_path: Path to cluster_state.pkl file

        Returns:
            Dictionary with embeddings_by_tb for computing cluster embeddings
        """
        import pickle

        logger.info(f"Loading cluster state from {cluster_state_path}")

        with open(cluster_state_path, 'rb') as f:
            cluster_state = pickle.load(f)

        embeddings_by_tb = self._normalize_embeddings_by_tb(cluster_state['embeddings_by_tb'])
        self.treebank_clusters = self._normalize_treebank_clusters(
            cluster_state['treebank_clusters'],
            embeddings_by_tb,
        )
        self._last_embeddings_by_tb = embeddings_by_tb

        logger.info(
            f"Loaded cluster state for {len(self.treebank_clusters)} cluster entries"
        )

        return embeddings_by_tb

    @property
    def last_embeddings_by_tb(self) -> Dict:
        """Expose most recently generated or loaded embeddings."""
        return self._last_embeddings_by_tb

    def _normalize_embeddings_by_tb(self, embeddings_by_tb: Dict) -> Dict:
        """Ensure all embedding batches use split-qualified sentence refs."""
        normalized = {}
        for (tb_code, split_name), emb_data in embeddings_by_tb.items():
            normalized[(tb_code, split_name)] = qualify_embeddings_for_split(
                tb_code,
                split_name,
                emb_data,
            )
        return normalized

    def _normalize_treebank_clusters(self, treebank_clusters: Dict, embeddings_by_tb: Dict) -> Dict:
        """Rewrite cluster sent_ids to match normalized embedding sentence refs."""
        sent_ref_lookup = {}
        for (tb_code, _split_name), emb_data in embeddings_by_tb.items():
            for sent_ref in emb_data.get('sent_id', []):
                ref_tb_code, _ref_split, raw_sent_id = extract_sentence_ref_parts(sent_ref)
                sent_ref_lookup[(ref_tb_code, raw_sent_id)] = sent_ref

        normalized_clusters = {}
        for tb_key, tb_info in treebank_clusters.items():
            normalized_info = dict(tb_info)
            cluster_result = dict(tb_info.get('cluster_result', {}))
            normalized_cluster_map = {}
            for cluster_id, cluster_info in cluster_result.get('clusters', {}).items():
                normalized_cluster_info = dict(cluster_info)
                normalized_sent_ids = []
                for sent_id in cluster_info.get('sent_ids', []):
                    if isinstance(sent_id, tuple) and len(sent_id) == 3:
                        normalized_sent_ids.append(sent_id)
                        continue

                    lookup_key = (tb_key[0], str(sent_id))
                    normalized_sent_ids.append(
                        sent_ref_lookup.get(
                            lookup_key,
                            qualify_sentence_ref(tb_key[0], str(tb_key[1]), sent_id),
                        )
                    )
                normalized_cluster_info['sent_ids'] = normalized_sent_ids
                normalized_cluster_map[cluster_id] = normalized_cluster_info
            cluster_result['clusters'] = normalized_cluster_map
            normalized_info['cluster_result'] = cluster_result
            normalized_clusters[tb_key] = normalized_info

        return normalized_clusters

    def _generate_embeddings(self, treebank_filter: Optional[List[str]] = None, overwrite: bool = False) -> Dict:
        """Generate embeddings for all treebanks.

        Checks cache first if configured, generates and caches otherwise.

        Args:
            treebank_filter: Optional list of treebank codes to process
            overwrite: If True, regenerate embeddings even if cached versions exist

        Returns:
            Dictionary: {(treebank_code, split): {'sent_ids': [...], 'embeddings': array}}
        """
        embeddings_by_tb = {}
        cache_dir = self.config.embeddings.cache_dir

        # Count total treebanks for progress reporting
        all_treebanks = list(self.data_loader.iter_all_treebanks(treebank_filter=treebank_filter))
        total_count = len(all_treebanks)
        logger.info(f"Processing embeddings for {total_count} treebank split(s)")

        # Iterate over filtered treebanks and splits
        for idx, (tb_code, split, dataset) in enumerate(all_treebanks, 1):
            # Try loading from cache first (unless overwrite is enabled)
            if cache_dir and not overwrite:
                cached = self.embedding_generator.load_embeddings(
                    tb_code, split, Path(cache_dir)
                )
                if cached is not None:
                    logger.info(f"[{idx}/{total_count}] Loaded cached embeddings for {tb_code}:{split} ({len(cached['sent_id'])} sentences)")
                    embeddings_by_tb[(tb_code, split)] = qualify_embeddings_for_split(
                        tb_code,
                        split,
                        cached,
                    )
                    continue

            # Generate embeddings
            logger.info(f"[{idx}/{total_count}] Generating embeddings for {tb_code}:{split} ({len(dataset)} sentences)")
            result = self.embedding_generator.embed_treebank(
                treebank_code=tb_code,
                split=split,
                dataset=dataset,
                output_path=Path(cache_dir) if cache_dir else None,
            )
            embeddings_by_tb[(tb_code, split)] = qualify_embeddings_for_split(
                tb_code,
                split,
                result,
            )

        self._last_embeddings_by_tb = embeddings_by_tb
        return embeddings_by_tb

    def _cluster_treebanks(self, embeddings_by_tb: Dict):
        """Cluster each treebank based on expected genres.

        Groups all splits (train/dev/test) of the same treebank together before clustering,
        then keeps a single combined-treebank clustering unit for labeling.

        Args:
            embeddings_by_tb: Pre-computed embeddings keyed by (treebank_code, split)
        """
        embeddings_by_tb = self._normalize_embeddings_by_tb(embeddings_by_tb)

        # Group by clustering level
        if self.config.clustering.level == "treebank":
            # Use shared operation: Group embeddings by treebank (combining all splits)
            treebank_keys = list(embeddings_by_tb.keys())
            treebank_groups = self.clustering_ops.group_splits_by_treebank(
                treebank_keys, embeddings_by_tb
            )

            total_treebanks = len(treebank_groups)
            logger.info(f"Clustering {total_treebanks} treebank(s) (combining splits)")

            for idx, (tb_code, tb_keys) in enumerate(treebank_groups.items(), 1):
                # Remove stale entries for treebanks being re-clustered.
                existing_keys = [key for key in self.treebank_clusters.keys() if key[0] == tb_code]
                for existing_key in existing_keys:
                    del self.treebank_clusters[existing_key]

                # Use shared operation: Combine all embeddings from all splits
                combined_embeddings, all_sent_ids, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )

                # Try to extract genres from actual sentences with per-sentence metadata
                # This enables creating virtual splits for multi-genre treebanks
                sentence_metadata = {}  # Maps (tb_code, split, sent_id) -> genre
                genres_from_sentences = set()
                ambiguous_genre_sentences = 0
                metadata_split_errors = []
                metadata_sentence_errors = 0

                # Extract genres from ALL splits
                for tb_key in tb_keys:
                    split = tb_key[1]
                    try:
                        dataset = self.data_loader.load_treebank(tb_code, split)
                    except Exception as e:
                        metadata_split_errors.append((split, e))
                        continue

                    for sentence in dataset:
                        sent_id = sentence.get('sent_id', None)
                        try:
                            extracted = self.genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                        except Exception:
                            metadata_sentence_errors += 1
                            continue

                        if sent_id and extracted:
                            sent_ref = qualify_sentence_ref(tb_code, split, sent_id)
                            if len(extracted) == 1:
                                genre = extracted[0]
                                sentence_metadata[(tb_code, split, sent_ref)] = genre
                                genres_from_sentences.add(genre)
                            else:
                                # Ambiguous sentence-level metadata: avoid arbitrary label choice.
                                ambiguous_genre_sentences += 1

                if metadata_split_errors:
                    preview = ", ".join(
                        f"{split}: {type(err).__name__}({err})"
                        for split, err in metadata_split_errors[:3]
                    )
                    if len(metadata_split_errors) > 3:
                        preview += f", ... (+{len(metadata_split_errors) - 3} more)"
                    logger.warning(
                        f"[{idx}/{total_treebanks}] Metadata extraction failed for "
                        f"{len(metadata_split_errors)} split(s) in {tb_code}: {preview}"
                    )

                if metadata_sentence_errors > 0:
                    logger.warning(
                        f"[{idx}/{total_treebanks}] Metadata extraction failed for "
                        f"{metadata_sentence_errors} sentence(s) in {tb_code}; skipped them"
                    )

                if ambiguous_genre_sentences > 0:
                    logger.info(
                        f"[{idx}/{total_treebanks}] Skipped {ambiguous_genre_sentences} sentence(s) "
                        f"with ambiguous multi-genre metadata in {tb_code}"
                    )

                if genres_from_sentences:
                    # Use genres extracted from actual sentences
                    genres = sorted(list(genres_from_sentences))
                else:
                    # Fallback: Get genres from treebank-level metadata and normalize
                    raw_genres = self.data_loader.get_treebank_genres(tb_code)
                    genres = [
                        self.genre_mapper.normalize_genre(g, tb_code) for g in raw_genres
                    ]
                    # Remove duplicates after normalization
                    genres = sorted(list(set(genres)))

                n_genres = len(genres)

                if n_genres == 0:
                    logger.warning(f"[{idx}/{total_treebanks}] {tb_code} has no genre metadata, skipping")
                    continue

                # Quality gates for virtual-split anchors are config-driven.
                virtual_split_coverage_threshold = (
                    self.config.evaluation.metadata_validation.coverage_threshold
                )
                virtual_split_min_genre_sentences = (
                    self.config.evaluation.metadata_validation.min_genre_sentences
                )

                # Use shared operation: Check if we can create virtual splits
                can_create_virtual_splits, eligible_virtual_split_genres = (
                    self.clustering_ops.check_virtual_split_coverage(
                        combined_embeddings,
                        all_sent_ids,
                        sent_id_to_split,
                        sentence_metadata,
                        tb_code,
                        coverage_threshold=virtual_split_coverage_threshold,
                        min_genre_sentences=virtual_split_min_genre_sentences,
                    )
                )

                virtual_splits = {}
                if can_create_virtual_splits:
                    virtual_splits = self.clustering_ops.create_virtual_splits(
                        tb_code,
                        combined_embeddings,
                        all_sent_ids,
                        sent_id_to_split,
                        sentence_metadata,
                    )

                    # Keep only genres that pass min_genre_sentences.
                    dropped_genres = sorted(
                        set(virtual_splits.keys()) - set(eligible_virtual_split_genres)
                    )
                    if dropped_genres:
                        logger.info(
                            f"[{idx}/{total_treebanks}] Ignoring low-support virtual split genre(s) "
                            f"for {tb_code}: {', '.join(dropped_genres)} "
                            f"(min_genre_sentences={virtual_split_min_genre_sentences})"
                        )
                        virtual_splits = {
                            genre: split_data
                            for genre, split_data in virtual_splits.items()
                            if genre in eligible_virtual_split_genres
                        }

                    # Safety check: require at least 2 eligible genres.
                    if len(virtual_splits) < 2:
                        can_create_virtual_splits = False
                        logger.info(
                            f"[{idx}/{total_treebanks}] Skipping virtual splits for {tb_code} "
                            f"(eligible genres < 2 after quality gates: "
                            f"coverage_threshold={virtual_split_coverage_threshold}, "
                            f"min_genre_sentences={virtual_split_min_genre_sentences})"
                        )

                if can_create_virtual_splits:
                    splits_list = [tk[1] for tk in tb_keys]
                    logger.info(
                        f"[{idx}/{total_treebanks}] Creating {len(virtual_splits)} virtual splits from {tb_code} "
                        f"({len(combined_embeddings)} sentences across {len(splits_list)} splits, "
                        f"genres: {', '.join(sorted(virtual_splits.keys()))}, "
                        f"coverage_threshold={virtual_split_coverage_threshold}, "
                        f"min_genre_sentences={virtual_split_min_genre_sentences})"
                    )

                    # Store virtual splits in production format
                    for genre, split_data in virtual_splits.items():
                        logger.info(
                            f"  Virtual split {tb_code}:{genre} ({len(split_data['sent_ids'])} sentences)"
                        )

                        # Create one combined virtual split cluster per treebank+genre.
                        cluster_result = {
                            "clusters": {
                                0: {
                                    "sent_ids": split_data["sent_ids"],
                                    "size": len(split_data["sent_ids"]),
                                    "confidence": 1.0,
                                }
                            },
                            "metrics": {},
                        }

                        virtual_key = (tb_code, COMBINED_SPLIT_KEY, genre)
                        self.treebank_clusters[virtual_key] = {
                            "genres": [genre],
                            "cluster_result": cluster_result,
                            "is_virtual_split": True,
                            "source_splits": sorted(split_data.get("split_distribution", {}).keys()),
                        }

                    # Cluster the combined treebank once for multi-genre labeling.
                    cluster_result = self.clusterer.cluster_treebank(
                        embeddings=combined_embeddings,
                        sent_ids=all_sent_ids,
                        n_genres=n_genres,
                    )

                    combined_key = (tb_code, COMBINED_SPLIT_KEY)
                    self.treebank_clusters[combined_key] = {
                        "genres": genres,
                        "cluster_result": cluster_result,
                        "has_virtual_splits": True,
                        "source_splits": [key[1] for key in tb_keys],
                    }

                elif n_genres == 1:
                    # Single-genre treebank
                    splits_list = [tk[1] for tk in tb_keys]
                    logger.info(
                        f"[{idx}/{total_treebanks}] Skipping clustering for single-genre treebank {tb_code} "
                        f"({len(combined_embeddings)} sentences across {len(splits_list)} splits, genre: {genres[0]})"
                    )

                    cluster_result = {
                        "clusters": {
                            0: {
                                "sent_ids": all_sent_ids,
                                "size": len(all_sent_ids),
                                "confidence": 1.0,
                            }
                        },
                        "metrics": {},
                    }

                    combined_key = (tb_code, COMBINED_SPLIT_KEY)
                    self.treebank_clusters[combined_key] = {
                        "genres": genres,
                        "cluster_result": cluster_result,
                        "source_splits": [key[1] for key in tb_keys],
                    }

                else:
                    # Multi-genre treebank without sentence-level metadata
                    splits_list = [tk[1] for tk in tb_keys]
                    logger.info(
                        f"[{idx}/{total_treebanks}] Clustering {tb_code} "
                        f"({len(combined_embeddings)} sentences across {len(splits_list)} splits, {n_genres} genres: {', '.join(genres)})"
                    )

                    # Cluster combined data from all splits
                    cluster_result = self.clusterer.cluster_treebank(
                        embeddings=combined_embeddings,
                        sent_ids=all_sent_ids,
                        n_genres=n_genres,
                    )

                    combined_key = (tb_code, COMBINED_SPLIT_KEY)
                    self.treebank_clusters[combined_key] = {
                        "genres": genres,
                        "cluster_result": cluster_result,
                        "source_splits": [key[1] for key in tb_keys],
                    }

        elif self.config.clustering.level == "language":
            # TODO: Implement language-level clustering
            # Group all treebanks by language and cluster together
            raise NotImplementedError("Language-level clustering not yet implemented")

        else:
            raise ValueError(f"Unknown clustering level: {self.config.clustering.level}")

    def _compute_cluster_embeddings(self, embeddings_by_tb: Dict):
        """Compute mean embeddings for each cluster.

        Args:
            embeddings_by_tb: Pre-computed embeddings
        """
        embeddings_by_tb = self._normalize_embeddings_by_tb(embeddings_by_tb)
        total_treebanks = len(self.treebank_clusters)
        logger.info(f"Computing cluster embeddings for {total_treebanks} cluster entries")

        for idx, (tb_key, tb_info) in enumerate(self.treebank_clusters.items(), 1):
            # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
            if len(tb_key) == 3:
                # Virtual split: (tb_code, split, genre)
                tb_code, split, genre = tb_key
                is_virtual_split = True
                display_split = "combined" if split == COMBINED_SPLIT_KEY else split
                display_name = f"{tb_code}:{display_split}:{genre}"
            else:
                # Regular treebank: (tb_code, split)
                tb_code, split = tb_key
                is_virtual_split = False
                display_split = "combined" if split == COMBINED_SPLIT_KEY else split
                display_name = f"{tb_code}:{display_split}"

            genres = tb_info["genres"]
            genre_combination = tuple(sorted(genres))

            cluster_result = tb_info["cluster_result"]

            source_splits = tb_info.get("source_splits")
            if source_splits:
                source_keys = [
                    (tb_code, source_split)
                    for source_split in source_splits
                    if (tb_code, source_split) in embeddings_by_tb
                ]
            elif (tb_code, split) in embeddings_by_tb:
                source_keys = [(tb_code, split)]
            else:
                source_keys = [
                    key for key in embeddings_by_tb.keys() if key[0] == tb_code
                ]

            if len(source_keys) == 0:
                logger.warning(
                    f"[{idx}/{total_treebanks}] No embeddings found for {display_name}, skipping"
                )
                continue

            if len(source_keys) == 1:
                emb_data = embeddings_by_tb[source_keys[0]]
            else:
                combined_embeddings, all_sent_ids, _ = self.clustering_ops.combine_treebank_splits(
                    source_keys,
                    embeddings_by_tb,
                )
                emb_data = {
                    "sent_id": all_sent_ids,
                    "embedding": combined_embeddings,
                }

            n_clusters = len(cluster_result["clusters"])

            virtual_tag = " (virtual split)" if is_virtual_split else ""
            logger.info(
                f"[{idx}/{total_treebanks}] Computing embeddings for {display_name} "
                f"({n_clusters} clusters, genres: {', '.join(genres)}){virtual_tag}"
            )

            # Compute mean embedding for each cluster
            for cluster_id, cluster_info in cluster_result["clusters"].items():
                # Get indices of sentences in this cluster
                sent_ids = cluster_info["sent_ids"]
                indices = [
                    i
                    for i, sid in enumerate(emb_data["sent_id"])
                    if sid in sent_ids
                ]
                if len(indices) == 0:
                    logger.warning(
                        f"[{idx}/{total_treebanks}] No embeddings found for {display_name} cluster {cluster_id}, skipping"
                    )
                    continue

                # Compute mean embedding
                cluster_emb = np.mean(emb_data["embedding"][indices], axis=0)

                # Store with appropriate key
                # For virtual splits, use the original key (tb_code, split, genre)
                # For regular treebanks, use (tb_code, split)
                storage_key = tb_key

                if storage_key not in self.genre_combination_clusters[genre_combination]:
                    self.genre_combination_clusters[genre_combination][storage_key] = []

                self.genre_combination_clusters[genre_combination][storage_key].append(
                    {
                        "cluster_id": cluster_id,
                        "sent_ids": sent_ids,
                        "embedding": cluster_emb,
                        "confidence": cluster_info["confidence"],
                    }
                )

    def _create_schedule(self) -> List[Dict]:
        """Create bootstrapping schedule.

        Returns:
            List of environments
        """
        genre_combinations = set(self.genre_combination_clusters.keys())
        schedule = self.scheduler.create_schedule(genre_combinations)

        if not self.scheduler.validate_schedule(schedule):
            if self.config.bootstrapping.fail_on_incomplete:
                raise ValueError("Cannot resolve all genre combinations")
            else:
                logger.warning("Some genre combinations cannot be resolved")

        return schedule

    def _label_single_genre_treebanks(self):
        """Label all sentences from single-genre treebanks and virtual splits.

        Single-genre treebanks (e.g., PoSTWITA with only 'social') and virtual splits
        should have all their sentences trivially labeled with that genre at 100% confidence.
        This step runs before bootstrap labeling.
        """
        labeled_count = 0
        treebanks_labeled = 0
        virtual_splits_labeled = 0

        for tb_key, tb_info in self.treebank_clusters.items():
            genres = tb_info['genres']

            # Only process single-genre treebanks/splits
            if len(genres) == 1:
                genre = genres[0]
                cluster_result = tb_info['cluster_result']

                # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
                if len(tb_key) == 3:
                    tb_code, split, genre_tag = tb_key
                    is_virtual_split = True
                    display_split = "combined" if split == COMBINED_SPLIT_KEY else split
                    display_name = f"{tb_code}:{display_split}:{genre_tag}"
                    method_tag = 'virtual-split'
                else:
                    tb_code, split = tb_key
                    is_virtual_split = False
                    display_split = "combined" if split == COMBINED_SPLIT_KEY else split
                    display_name = f"{tb_code}:{display_split}"
                    method_tag = 'single-genre-treebank'

                # Get all sentence IDs from all clusters
                for cluster_id, cluster_info in cluster_result['clusters'].items():
                    for sent_id in cluster_info['sent_ids']:
                        # Label with 100% confidence (known single-genre treebank/split)
                        self.final_labels[sent_id] = (genre, 1.0, method_tag)
                        labeled_count += 1

                if is_virtual_split:
                    virtual_splits_labeled += 1
                else:
                    treebanks_labeled += 1

                n_sentences = sum(len(c['sent_ids']) for c in cluster_result['clusters'].values())
                logger.info(
                    f"  Labeled {display_name} with '{genre}' ({n_sentences} sentences)"
                )

        if virtual_splits_labeled > 0:
            logger.info(
                f"Labeled {labeled_count} sentences from {treebanks_labeled} single-genre treebank(s) "
                f"and {virtual_splits_labeled} virtual split(s)"
            )
        else:
            logger.info(
                f"Labeled {labeled_count} sentences from {treebanks_labeled} single-genre treebank(s)"
            )

    def _label_clusters(self, schedule: List[Dict]):
        """Label clusters according to bootstrap schedule.

        Args:
            schedule: Bootstrap schedule from scheduler
        """
        preserve_methods = {"virtual-split", "single-genre-treebank"}

        # Iterate through environments and delegate label propagation to shared ops.
        for env_idx, environment in enumerate(schedule):
            logger.info(
                f"Environment {env_idx + 1}/{len(schedule)}: "
                f"{len(environment['known'])} known genres"
            )

            if len(environment['predict']) == 0:
                logger.info("No predictions to make, schedule complete")
                break

            _, env_summaries = self.clustering_ops.run_bootstrap_schedule(
                schedule=[environment],
                genre_combination_clusters=self.genre_combination_clusters,
                final_labels=self.final_labels,
                preserve_methods=preserve_methods,
            )
            env_summary = env_summaries[0]
            logger.info(
                f"  Summary: {env_summary['labels_assigned']} clusters labeled "
                f"({env_summary['labels_high_confidence']} high conf, "
                f"{env_summary['labels_low_confidence']} low conf)"
            )

    def _get_known_genre_embeddings(self, known_genres: List[str]) -> Dict[str, np.ndarray]:
        """Get reference embeddings for known single-genre sources.

        Reference embeddings are computed as weighted means of cluster centroids
        for each known genre. Weighting behavior is controlled by
        ``bootstrapping.reference_weighting``.

        Args:
            known_genres: List of known genre labels

        Returns:
            Dictionary: {genre: reference_embedding}
        """
        return self.clustering_ops.build_reference_embeddings_from_cluster_pool(
            self.genre_combination_clusters,
            known_genres,
        )

    def _label_environment(self, environment: Dict, known_embeddings: Dict):
        """Label clusters in current environment.

        Args:
            environment: Current environment from schedule
            known_embeddings: Mean embeddings for known genres
        """
        env_summary = self.clustering_ops.label_predictable_combinations(
            predict_combinations=environment.get("predict", []),
            genre_combination_clusters=self.genre_combination_clusters,
            known_embeddings=known_embeddings,
            final_labels=self.final_labels,
            preserve_methods={"virtual-split", "single-genre-treebank"},
        )

        logger.info(
            f"  Summary: {env_summary['labels_assigned']} clusters labeled "
            f"({env_summary['labels_high_confidence']} high conf, "
            f"{env_summary['labels_low_confidence']} low conf)"
        )

    def _generate_cross_lingual_report(self):
        """Generate a report showing cross-lingual genre assignments.

        This helps verify if GMM+L is correctly identifying the same genres
        across different languages.
        """
        from collections import defaultdict

        # Group assignments by assigned genre
        genre_assignments = defaultdict(lambda: defaultdict(list))

        # Collect cluster assignments grouped by genre
        for genre_combo, treebank_clusters in self.genre_combination_clusters.items():
            for tb_key, clusters in treebank_clusters.items():
                # tb_key can be (tb_code, split) or (tb_code, split, genre) for virtual splits
                if len(tb_key) == 3:
                    tb_code, split, genre_tag = tb_key
                else:
                    tb_code, split = tb_key
                display_split = "combined" if split == COMBINED_SPLIT_KEY else split

                # Extract language code from treebank
                lang_code = tb_code.split('_')[0]

                for cluster in clusters:
                    # Find the assigned genre for sentences in this cluster
                    if cluster['sent_ids']:
                        first_sent = cluster['sent_ids'][0]
                        if first_sent in self.final_labels:
                            assigned_genre, confidence, method = self.final_labels[first_sent]
                            genre_assignments[assigned_genre][lang_code].append({
                                'treebank': tb_code,
                                'split': display_split,
                                'cluster_id': cluster['cluster_id'],
                                'n_sentences': len(cluster['sent_ids']),
                                'confidence': confidence,
                                'original_genres': genre_combo,
                            })

        # Log the cross-lingual report
        logger.info("\n" + "=" * 80)
        logger.info("CROSS-LINGUAL GENRE ASSIGNMENT REPORT")
        logger.info("=" * 80)

        for genre in sorted(genre_assignments.keys()):
            langs = genre_assignments[genre]
            total_clusters = sum(len(clusters) for clusters in langs.values())
            total_sentences = sum(
                sum(c['n_sentences'] for c in clusters)
                for clusters in langs.values()
            )

            logger.info(f"\nGenre: {genre.upper()}")
            logger.info(f"  Found in {len(langs)} language(s), {total_clusters} cluster(s), {total_sentences} sentence(s)")

            # Show per-language breakdown
            for lang in sorted(langs.keys()):
                clusters = langs[lang]
                n_clusters = len(clusters)
                n_sentences = sum(c['n_sentences'] for c in clusters)
                avg_conf = sum(c['confidence'] for c in clusters) / n_clusters if n_clusters > 0 else 0

                logger.info(f"    {lang}: {n_clusters} cluster(s), {n_sentences} sent(s), avg_conf={avg_conf:.3f}")

                # Show details for each cluster
                for cluster_info in clusters[:3]:  # Show first 3 clusters per language
                    logger.debug(
                        f"      - {cluster_info['treebank']}:{cluster_info['split']} "
                        f"c{cluster_info['cluster_id']} ({cluster_info['n_sentences']} sents, "
                        f"conf={cluster_info['confidence']:.3f}, "
                        f"orig={cluster_info['original_genres']})"
                    )
                if len(clusters) > 3:
                    logger.debug(f"      ... and {len(clusters)-3} more cluster(s)")

        logger.info("\n" + "=" * 80)

        # Cross-lingual consistency check
        logger.info("\nCROSS-LINGUAL CONSISTENCY CHECK:")
        multi_lingual_genres = {g: langs for g, langs in genre_assignments.items() if len(langs) > 1}

        if multi_lingual_genres:
            logger.info(f"✓ Found {len(multi_lingual_genres)} genre(s) spanning multiple languages:")
            for genre in sorted(multi_lingual_genres.keys()):
                langs = list(multi_lingual_genres[genre].keys())
                logger.info(f"  - {genre}: {', '.join(sorted(langs))}")
        else:
            logger.warning("✗ No genres found spanning multiple languages!")
            logger.warning("  This suggests clustering may be separating by language rather than genre.")

        logger.info("=" * 80 + "\n")

    def _compute_genre_separation_metrics(self):
        """Compute and report how separable different genres are in embedding space.

        Uses single-genre cluster embeddings to measure genre separability.
        This helps answer: "How far apart are 'news' and 'social' in the embedding space?"
        """
        from ud_genre_bootstrap.evaluation.metrics import GenreSeparationMetrics

        logger.info("\n" + "=" * 80)
        logger.info("Genre Separation Analysis")
        logger.info("=" * 80)

        # Collect mean embeddings for each known single genre
        genre_embeddings = {}

        for genre_combo, treebank_clusters in self.genre_combination_clusters.items():
            # Only process single genres
            if len(genre_combo) == 1:
                genre = genre_combo[0]

                # Collect all cluster embeddings for this genre
                all_embeddings = []
                for tb_key, clusters in treebank_clusters.items():
                    # tb_key can be (tb_code, split) or (tb_code, split, genre) for virtual splits
                    for cluster in clusters:
                        all_embeddings.append(cluster["embedding"])

                if all_embeddings:
                    # Compute mean embedding across all treebanks for this genre
                    genre_embeddings[genre] = np.mean(all_embeddings, axis=0)

        if len(genre_embeddings) < 2:
            logger.info("Need at least 2 single-genre references to compute separation metrics")
            return

        # Compute pairwise distances
        separation_metrics = GenreSeparationMetrics.compute_genre_centroid_distances(
            genre_embeddings, metric="cosine"
        )

        logger.info(f"Analyzing {len(separation_metrics['genres'])} genres")
        logger.info(f"Average pairwise distance (cosine): {separation_metrics['mean_distance']:.4f}")

        if separation_metrics['min_pair']:
            min_genre1, min_genre2, min_dist = separation_metrics['min_pair']
            logger.info(f"Closest pair: {min_genre1} ↔ {min_genre2} (distance: {min_dist:.4f})")

        if separation_metrics['max_pair']:
            max_genre1, max_genre2, max_dist = separation_metrics['max_pair']
            logger.info(f"Furthest pair: {max_genre1} ↔ {max_genre2} (distance: {max_dist:.4f})")

        # Display pairwise distance matrix
        logger.info("\nPairwise Genre Distances (cosine):")
        genres = separation_metrics['genres']
        matrix = separation_metrics['pairwise_matrix']

        # Format as table
        header = "         " + "  ".join(f"{g:>8}" for g in genres)
        logger.info(header)
        for i, genre_i in enumerate(genres):
            row = f"{genre_i:>8} " + "  ".join(f"{matrix[i][j]:>8.4f}" for j in range(len(genres)))
            logger.info(row)

        logger.info("\nInterpretation:")
        logger.info("  - Higher distance = better separated (easier to distinguish)")
        logger.info("  - Lower distance = more similar (harder to distinguish)")
        logger.info("  - Cosine distance range: [0, 2], typical range: [0, 1]")

        # Save to output
        output_path = Path(self.config.output.genres_path)
        output_path.mkdir(parents=True, exist_ok=True)
        import json
        metrics_file = output_path / "genre_separation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(separation_metrics, f, indent=2)
        logger.info(f"\nSaved genre separation metrics to: {metrics_file}")

        logger.info("=" * 80 + "\n")

    def _summarize_exported_labels_file(self, output_file: Path) -> Dict:
        """Summarize an exported ``all_genres.parquet`` file."""
        return summarize_exported_labels_file(output_file)

    def _export_results(self) -> Dict:
        """Export final genre labels to parquet files.

        Returns:
            Dictionary with statistics
        """
        import pandas as pd

        output_path = Path(self.config.output.genres_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting results to: {output_path}")

        stats = {
            'total_sentences': len(self.final_labels),
            'labeled_sentences': sum(1 for _, (genre, _confidence, _method) in self.final_labels.items() if genre is not None),
            'method_counts': {},
            'genre_counts': {},
        }

        for genre, _confidence, method in self.final_labels.values():
            stats['method_counts'][method] = stats['method_counts'].get(method, 0) + 1
            if genre:
                stats['genre_counts'][genre] = stats['genre_counts'].get(genre, 0) + 1

        row_metadata = build_release_row_metadata(self.config)
        df_data = []
        for sent_ref, (genre, confidence, method) in self.final_labels.items():
            tb_code, split_name, raw_sent_id = extract_sentence_ref_parts(sent_ref)
            row = {
                'treebank': tb_code,
                'split': split_name,
                'sent_id': raw_sent_id,
                'genre': genre,
                'confidence': float(confidence) if confidence is not None else None,
                'method': method,
            }
            row.update(row_metadata)
            df_data.append(row)

        output_file = output_path / 'all_genres.parquet'
        if df_data:
            df = pd.DataFrame(df_data).sort_values(
                by=['treebank', 'split', 'sent_id'],
                kind='stable',
            )
            df.to_parquet(output_file, index=False)
            logger.info(f"Exported {len(df)} labels to {output_file}")
        elif output_file.exists():
            output_file.unlink()

        stats['release_artifacts'] = write_release_artifacts(
            self.config,
            output_path,
            stats,
            all_genres_path=output_file if output_file.exists() else None,
        )
        logger.info(
            "Wrote release artifacts for config %s",
            resolve_config_name(self.config),
        )

        return stats

    def prepare_release_artifacts(self) -> Dict[str, str]:
        """Regenerate release metadata for an existing exported labels directory."""
        output_path = Path(self.config.output.genres_path)
        return prepare_release_directory(self.config, output_path)

    def push_to_hub(self, repo_id: str, revision):
        """Push results to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            revision: Revision/branch name or list of revision/branch names
        """
        revisions = [revision] if isinstance(revision, str) else list(revision)
        revisions = [str(item) for item in revisions if str(item).strip()]
        if not revisions:
            raise ValueError("At least one Hugging Face revision is required")

        token = self.config.output.hf_token
        if not token:
            logger.warning("No HF token provided, skipping push to hub")
            return None

        output_path = Path(self.config.output.genres_path)
        return upload_release_directory_to_hub(
            self.config,
            output_path,
            repo_id,
            revisions,
        )
