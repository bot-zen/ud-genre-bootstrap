"""Main bootstrapping class for genre labeling."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance

from ud_genre_bootstrap.bootstrapping.scheduler import BootstrapScheduler
from ud_genre_bootstrap.clustering.clustering_utils import ClusteringOperations
from ud_genre_bootstrap.clustering.gmm_clusterer import GMMClusterer
from ud_genre_bootstrap.clustering.kmeans_clusterer import KMeansClusterer
from ud_genre_bootstrap.embeddings.generator import EmbeddingGenerator
from ud_genre_bootstrap.utils.config import Config
from ud_genre_bootstrap.utils.data_loader import UDDataLoader
from ud_genre_bootstrap.utils.genre_mapping import GenreMapper

logger = logging.getLogger(__name__)


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
        )

        # Storage for results
        self.treebank_clusters: Dict = {}  # {treebank_code: cluster_info}
        self.genre_combination_clusters: Dict = defaultdict(dict)
        self.final_labels: Dict = {}  # {sent_id: (genre, confidence, method)}

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

        # Restore treebank clusters
        self.treebank_clusters = cluster_state['treebank_clusters']
        embeddings_by_tb = cluster_state['embeddings_by_tb']

        logger.info(
            f"Loaded cluster state for {len(self.treebank_clusters)} treebank splits"
        )

        return embeddings_by_tb

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
                    embeddings_by_tb[(tb_code, split)] = cached
                    continue

            # Generate embeddings
            logger.info(f"[{idx}/{total_count}] Generating embeddings for {tb_code}:{split} ({len(dataset)} sentences)")
            result = self.embedding_generator.embed_treebank(
                treebank_code=tb_code,
                split=split,
                dataset=dataset,
                output_path=Path(cache_dir) if cache_dir else None,
            )
            embeddings_by_tb[(tb_code, split)] = result

        return embeddings_by_tb

    def _cluster_treebanks(self, embeddings_by_tb: Dict):
        """Cluster each treebank based on expected genres.

        Groups all splits (train/dev/test) of the same treebank together before clustering,
        then distributes results back to individual splits.

        Args:
            embeddings_by_tb: Pre-computed embeddings keyed by (treebank_code, split)
        """
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
                # Use shared operation: Combine all embeddings from all splits
                combined_embeddings, all_sent_ids, sent_id_to_split = (
                    self.clustering_ops.combine_treebank_splits(tb_keys, embeddings_by_tb)
                )

                # Try to extract genres from actual sentences with per-sentence metadata
                # This enables creating virtual splits for multi-genre treebanks
                sentence_metadata = {}  # Maps (tb_code, split, sent_id) -> genre
                genres_from_sentences = set()

                # Extract genres from ALL splits
                for tb_key in tb_keys:
                    split = tb_key[1]
                    try:
                        dataset = self.data_loader.load_treebank(tb_code, split)
                        for sentence in dataset:
                            sent_id = sentence.get('sent_id', None)
                            extracted = self.genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                            if extracted and sent_id:
                                # Use first genre if multiple (rare)
                                genre = extracted[0]
                                sentence_metadata[(tb_code, split, sent_id)] = genre
                                genres_from_sentences.add(genre)
                    except Exception:
                        pass  # Fall back to treebank-level metadata

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

                # Use shared operation: Check if we can create virtual splits
                can_create_virtual_splits, _ = self.clustering_ops.check_virtual_split_coverage(
                    combined_embeddings,
                    all_sent_ids,
                    sent_id_to_split,
                    sentence_metadata,
                    tb_code,
                    coverage_threshold=0.8,
                )

                if can_create_virtual_splits:
                    # Use shared operation: Create virtual splits from combined data
                    splits_list = [tk[1] for tk in tb_keys]
                    logger.info(
                        f"[{idx}/{total_treebanks}] Creating {n_genres} virtual splits from {tb_code} "
                        f"({len(combined_embeddings)} sentences across {len(splits_list)} splits, genres: {', '.join(genres)})"
                    )

                    virtual_splits = self.clustering_ops.create_virtual_splits(
                        tb_code,
                        combined_embeddings,
                        all_sent_ids,
                        sent_id_to_split,
                        sentence_metadata,
                    )

                    # Store virtual splits in production format
                    for genre, split_data in virtual_splits.items():
                        logger.info(
                            f"  Virtual split {tb_code}:{genre} ({len(split_data['sent_ids'])} sentences)"
                        )

                        # Group by original split for storage
                        split_distribution = split_data['split_distribution']
                        for split in split_distribution.keys():
                            split_genre_sent_ids = [
                                sid for sid in split_data['sent_ids']
                                if sent_id_to_split[sid] == split
                            ]

                            if len(split_genre_sent_ids) == 0:
                                continue

                            # Create trivial cluster (virtual splits are single-genre by definition)
                            cluster_result = {
                                "clusters": {
                                    0: {
                                        "sent_ids": split_genre_sent_ids,
                                        "size": len(split_genre_sent_ids),
                                        "confidence": 1.0,
                                    }
                                },
                                "metrics": {},
                            }

                            # Store virtual split with special key including genre
                            virtual_key = (tb_code, split, genre)
                            self.treebank_clusters[virtual_key] = {
                                "genres": [genre],  # Single genre for virtual split
                                "cluster_result": cluster_result,
                                "is_virtual_split": True,
                            }

                    # Also cluster the combined data and store results for each split
                    # This is used for clustering evaluation
                    cluster_result = self.clusterer.cluster_treebank(
                        embeddings=combined_embeddings,
                        sent_ids=all_sent_ids,
                        n_genres=n_genres,
                    )

                    # Distribute cluster assignments back to individual splits
                    for tb_key in tb_keys:
                        split = tb_key[1]
                        emb_data = embeddings_by_tb[tb_key]
                        split_sent_ids = emb_data['sent_id']

                        # Extract cluster assignments for this split's sentences
                        split_clusters = {}
                        for cluster_id, cluster_info in cluster_result["clusters"].items():
                            split_cluster_sent_ids = [
                                sid for sid in cluster_info["sent_ids"]
                                if sid in split_sent_ids
                            ]
                            if len(split_cluster_sent_ids) > 0:
                                split_clusters[cluster_id] = {
                                    "sent_ids": split_cluster_sent_ids,
                                    "size": len(split_cluster_sent_ids),
                                    "confidence": cluster_info.get("confidence", 1.0),
                                }

                        split_cluster_result = {
                            "clusters": split_clusters,
                            "metrics": cluster_result.get("metrics", {}),
                        }

                        self.treebank_clusters[(tb_code, split)] = {
                            "genres": genres,
                            "cluster_result": split_cluster_result,
                            "has_virtual_splits": True,
                        }

                elif n_genres == 1:
                    # Single-genre treebank
                    splits_list = [tk[1] for tk in tb_keys]
                    logger.info(
                        f"[{idx}/{total_treebanks}] Skipping clustering for single-genre treebank {tb_code} "
                        f"({len(combined_embeddings)} sentences across {len(splits_list)} splits, genre: {genres[0]})"
                    )

                    # Create trivial cluster result for each split
                    for tb_key in tb_keys:
                        split = tb_key[1]
                        emb_data = embeddings_by_tb[tb_key]
                        sent_ids = emb_data["sent_id"]
                        cluster_result = {
                            "clusters": {
                                0: {
                                    "sent_ids": sent_ids,
                                    "size": len(sent_ids),
                                    "confidence": 1.0,
                                }
                            },
                            "metrics": {},
                        }

                        self.treebank_clusters[(tb_code, split)] = {
                            "genres": genres,
                            "cluster_result": cluster_result,
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

                    # Distribute cluster assignments back to individual splits
                    for tb_key in tb_keys:
                        split = tb_key[1]
                        emb_data = embeddings_by_tb[tb_key]
                        split_sent_ids = emb_data['sent_id']

                        # Extract cluster assignments for this split's sentences
                        split_clusters = {}
                        for cluster_id, cluster_info in cluster_result["clusters"].items():
                            split_cluster_sent_ids = [
                                sid for sid in cluster_info["sent_ids"]
                                if sid in split_sent_ids
                            ]
                            if len(split_cluster_sent_ids) > 0:
                                split_clusters[cluster_id] = {
                                    "sent_ids": split_cluster_sent_ids,
                                    "size": len(split_cluster_sent_ids),
                                    "confidence": cluster_info.get("confidence", 1.0),
                                }

                        split_cluster_result = {
                            "clusters": split_clusters,
                            "metrics": cluster_result.get("metrics", {}),
                        }

                        self.treebank_clusters[(tb_code, split)] = {
                            "genres": genres,
                            "cluster_result": split_cluster_result,
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
        total_treebanks = len(self.treebank_clusters)
        logger.info(f"Computing cluster embeddings for {total_treebanks} treebank split(s)")

        for idx, (tb_key, tb_info) in enumerate(self.treebank_clusters.items(), 1):
            # Handle both regular keys (tb_code, split) and virtual split keys (tb_code, split, genre)
            if len(tb_key) == 3:
                # Virtual split: (tb_code, split, genre)
                tb_code, split, genre = tb_key
                is_virtual_split = True
                display_name = f"{tb_code}:{split}:{genre}"
            else:
                # Regular treebank: (tb_code, split)
                tb_code, split = tb_key
                is_virtual_split = False
                display_name = f"{tb_code}:{split}"

            genres = tb_info["genres"]
            genre_combination = tuple(sorted(genres))

            cluster_result = tb_info["cluster_result"]
            emb_data = embeddings_by_tb[(tb_code, split)]

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
                    display_name = f"{tb_code}:{split}:{genre_tag}"
                    method_tag = 'virtual-split'
                else:
                    tb_code, split = tb_key
                    is_virtual_split = False
                    display_name = f"{tb_code}:{split}"
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
        # Iterate through environments
        for env_idx, environment in enumerate(schedule):
            logger.info(
                f"Environment {env_idx + 1}/{len(schedule)}: "
                f"{len(environment['known'])} known genres"
            )

            if len(environment['predict']) == 0:
                logger.info("No predictions to make, schedule complete")
                break

            # Compute mean embeddings for known genres
            known_embeddings = self._get_known_genre_embeddings(environment["known"])

            # Label predictable combinations
            self._label_environment(environment, known_embeddings)

    def _get_known_genre_embeddings(self, known_genres: List[str]) -> Dict[str, np.ndarray]:
        """Get mean embeddings for known single-genre treebanks.

        Args:
            known_genres: List of known genre labels

        Returns:
            Dictionary: {genre: mean_embedding}
        """
        known_embeddings = {}

        for genre in known_genres:
            # Get all clusters for this single genre
            if (genre,) in self.genre_combination_clusters:
                all_embeddings = [
                    cluster["embedding"]
                    for tb_clusters in self.genre_combination_clusters[(genre,)].values()
                    for cluster in tb_clusters
                ]

                if all_embeddings:
                    known_embeddings[genre] = np.mean(all_embeddings, axis=0)

        return known_embeddings

    def _label_environment(self, environment: Dict, known_embeddings: Dict):
        """Label clusters in current environment.

        Args:
            environment: Current environment from schedule
            known_embeddings: Mean embeddings for known genres
        """
        min_confidence = self.config.bootstrapping.min_confidence

        # Track statistics for this environment
        labels_assigned = 0
        labels_high_confidence = 0
        labels_low_confidence = 0

        # Iterate through genre combinations that can be predicted
        for genre_combination in environment['predict']:
            if genre_combination not in self.genre_combination_clusters:
                continue

            # Get all clusters for this genre combination
            tb_cluster_count = len(self.genre_combination_clusters[genre_combination])
            logger.info(f"  Labeling {tb_cluster_count} treebank(s) with genre combination {genre_combination}")

            for tb_key, clusters in self.genre_combination_clusters[genre_combination].items():
                # tb_key can be (tb_code, split) or (tb_code, split, genre) for virtual splits
                if len(tb_key) == 3:
                    tb_code, split, genre_tag = tb_key
                    display_name = f"{tb_code}:{split}:{genre_tag}"
                else:
                    tb_code, split = tb_key
                    display_name = f"{tb_code}:{split}"
                logger.debug(f"    Processing {display_name} ({len(clusters)} clusters)")

                for cluster in clusters:
                    cluster_emb = cluster['embedding']
                    cluster_id = cluster['cluster_id']
                    n_sentences = len(cluster['sent_ids'])

                    # Compute cosine similarity to each known genre
                    similarities = {}
                    for genre, genre_emb in known_embeddings.items():
                        # Cosine similarity
                        cosine_sim = 1 - distance.cosine(cluster_emb, genre_emb)
                        similarities[genre] = cosine_sim

                    # Find best matching genre
                    if similarities:
                        best_genre = max(similarities, key=similarities.get)
                        confidence = similarities[best_genre]

                        # Sort similarities for logging
                        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
                        top_3 = ", ".join([f"{g}:{s:.3f}" for g, s in sorted_sims[:3]])

                        # Only label if confidence exceeds threshold
                        if confidence >= min_confidence:
                            method = "bootstrap-labeled"
                            labels_high_confidence += 1
                            logger.debug(
                                f"      Cluster c{cluster_id} ({n_sentences} sents) → {best_genre} "
                                f"(conf={confidence:.3f}, top3: {top_3})"
                            )
                        else:
                            # Low confidence - mark as inferred
                            method = "bootstrap-inferred"
                            labels_low_confidence += 1
                            logger.info(
                                f"      ⚠ Cluster c{cluster_id} in {tb_code}:{split} → {best_genre} "
                                f"(LOW conf={confidence:.3f}, top3: {top_3})"
                            )

                        labels_assigned += 1

                        # Store labels for all sentences in this cluster
                        for sent_id in cluster['sent_ids']:
                            # Preserve metadata-derived/trivial labels assigned earlier.
                            existing = self.final_labels.get(sent_id)
                            if existing is not None and existing[2] in {"virtual-split", "single-genre-treebank"}:
                                continue
                            self.final_labels[sent_id] = (best_genre, confidence, method)

        # Log summary for this environment
        logger.info(
            f"  Summary: {labels_assigned} clusters labeled "
            f"({labels_high_confidence} high conf, {labels_low_confidence} low conf)"
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
                                'split': split,
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

    def _export_results(self) -> Dict:
        """Export final genre labels to parquet files.

        Returns:
            Dictionary with statistics
        """
        import pandas as pd

        output_path = Path(self.config.output.genres_path)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting results to: {output_path}")

        # Group labels by treebank and split
        labels_by_tb = defaultdict(lambda: defaultdict(list))

        for sent_id, (genre, confidence, method) in self.final_labels.items():
            # Parse sent_id to extract treebank info
            # Format might be: "treebank-split-sentid" or similar
            # For now, we'll need to track this during clustering
            # Let's create a mapping
            pass

        # Export to parquet for each treebank/split
        stats = {
            'total_sentences': len(self.final_labels),
            'labeled_sentences': sum(1 for _, (g, c, m) in self.final_labels.items() if g is not None),
            'method_counts': {},
            'genre_counts': {},
        }

        # Count methods and genres
        for genre, confidence, method in self.final_labels.values():
            stats['method_counts'][method] = stats['method_counts'].get(method, 0) + 1
            if genre:
                stats['genre_counts'][genre] = stats['genre_counts'].get(genre, 0) + 1

        # TODO: Group by treebank and export individual parquet files
        # For now, export a single file with all labels
        df_data = []
        for sent_id, (genre, confidence, method) in self.final_labels.items():
            df_data.append({
                'sent_id': sent_id,
                'genre': genre,
                'confidence': float(confidence) if confidence is not None else None,
                'method': method,
            })

        if df_data:
            df = pd.DataFrame(df_data)
            output_file = output_path / "all_genres.parquet"
            df.to_parquet(output_file, index=False)
            logger.info(f"Exported {len(df)} labels to {output_file}")

        return stats

    def push_to_hub(self, repo_id: str, revision: str):
        """Push results to HuggingFace Hub.

        Args:
            repo_id: HuggingFace repo ID
            revision: Revision/branch name
        """
        from huggingface_hub import HfApi, create_repo

        api = HfApi()

        # Get token from config
        token = self.config.output.hf_token
        if not token:
            logger.warning("No HF token provided, skipping push to hub")
            return

        # Create repo if it doesn't exist
        try:
            create_repo(
                repo_id,
                token=token,
                repo_type="dataset",
                exist_ok=True,
                private=False,
            )
            logger.info(f"Created/verified repo: {repo_id}")
        except Exception as e:
            logger.error(f"Failed to create repo: {e}")
            raise

        # Upload parquet files
        output_path = Path(self.config.output.genres_path)

        for parquet_file in output_path.glob("**/*.parquet"):
            # Upload to HF Hub
            relative_path = parquet_file.relative_to(output_path)
            try:
                api.upload_file(
                    path_or_fileobj=str(parquet_file),
                    path_in_repo=str(relative_path),
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=token,
                    revision=revision,
                )
                logger.info(f"Uploaded {relative_path} to {repo_id}")
            except Exception as e:
                logger.error(f"Failed to upload {relative_path}: {e}")

        logger.info(f"Push to hub complete: {repo_id} (revision: {revision})")
