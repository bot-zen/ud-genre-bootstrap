"""Main bootstrapping class for genre labeling."""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import distance

from ud_genre_bootstrap.bootstrapping.scheduler import BootstrapScheduler
from ud_genre_bootstrap.clustering.gmm_clusterer import GMMClusterer
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
        )

        self.embedding_generator = EmbeddingGenerator(
            model_name=config.embeddings.model,
            pooling=config.embeddings.pooling,
            layer=config.embeddings.layer,
            batch_size=config.embeddings.batch_size,
            device=config.embeddings.device,
        )

        self.clusterer = GMMClusterer(
            random_state=config.clustering.seed,
        )

        self.scheduler = BootstrapScheduler(
            max_iterations=config.bootstrapping.max_iterations,
        )

        # Storage for results
        self.treebank_clusters: Dict = {}  # {treebank_code: cluster_info}
        self.genre_combination_clusters: Dict = defaultdict(dict)
        self.final_labels: Dict = {}  # {sent_id: (genre, confidence, method)}

    def fit(self) -> Dict:
        """Run the full bootstrapping pipeline.

        Returns:
            Dictionary with results and statistics
        """
        logger.info("Starting bootstrap genre classification pipeline")

        # Step 1: Embed all treebanks
        logger.info("Step 1: Generating embeddings")
        embeddings_by_tb = self._generate_embeddings()

        # Step 2: Cluster each treebank
        logger.info("Step 2: Clustering treebanks")
        self._cluster_treebanks(embeddings_by_tb)

        # Step 3: Compute cluster embeddings and group by genre combinations
        logger.info("Step 3: Computing cluster embeddings")
        self._compute_cluster_embeddings(embeddings_by_tb)

        # Step 4: Create bootstrap schedule
        logger.info("Step 4: Creating bootstrap schedule")
        schedule = self._create_schedule()

        # Step 5: Label clusters according to schedule
        logger.info("Step 5: Labeling clusters")
        self._label_clusters(schedule)

        # Step 5.5: Generate cross-lingual assignment report
        logger.info("Step 5.5: Generating cross-lingual assignment report")
        self._generate_cross_lingual_report()

        # Step 6: Export results
        logger.info("Step 6: Exporting results")
        results = self._export_results()

        logger.info("Bootstrap pipeline complete")
        return results

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

        # Iterate over filtered treebanks and splits
        for tb_code, split, dataset in self.data_loader.iter_all_treebanks(treebank_filter=treebank_filter):
            # Try loading from cache first (unless overwrite is enabled)
            if cache_dir and not overwrite:
                cached = self.embedding_generator.load_embeddings(
                    tb_code, split, Path(cache_dir)
                )
                if cached is not None:
                    embeddings_by_tb[(tb_code, split)] = cached
                    continue

            # Generate embeddings
            logger.info(f"Embedding {tb_code} {split}: {len(dataset)} sentences")
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

        Args:
            embeddings_by_tb: Pre-computed embeddings
        """
        # Group by clustering level
        if self.config.clustering.level == "treebank":
            # Cluster each treebank independently
            for (tb_code, split), emb_data in embeddings_by_tb.items():
                # Try to extract genres from actual sentences first (more accurate)
                # This respects genre extraction patterns and mappings
                genres_from_sentences = set()
                try:
                    dataset = self.data_loader.load_treebank(tb_code, split)
                    for sentence in dataset:
                        extracted = self.genre_mapper.extract_genres_from_metadata(sentence, tb_code)
                        genres_from_sentences.update(extracted)
                except Exception:
                    pass  # Fall back to treebank-level metadata

                if genres_from_sentences:
                    # Use genres extracted from actual sentences
                    genres = list(genres_from_sentences)
                else:
                    # Fallback: Get genres from treebank-level metadata and normalize
                    raw_genres = self.data_loader.get_treebank_genres(tb_code)
                    genres = [
                        self.genre_mapper.normalize_genre(g, tb_code) for g in raw_genres
                    ]
                    # Remove duplicates after normalization
                    genres = list(set(genres))

                n_genres = len(genres)

                if n_genres == 0:
                    logger.warning(f"{tb_code} has no genre metadata, skipping")
                    continue

                cluster_result = self.clusterer.cluster_treebank(
                    embeddings=emb_data["embedding"],
                    sent_ids=emb_data["sent_id"],
                    n_genres=n_genres,
                )

                self.treebank_clusters[(tb_code, split)] = {
                    "genres": genres,
                    "cluster_result": cluster_result,
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
        for (tb_code, split), tb_info in self.treebank_clusters.items():
            genres = tb_info["genres"]
            genre_combination = tuple(sorted(genres))

            cluster_result = tb_info["cluster_result"]
            emb_data = embeddings_by_tb[(tb_code, split)]

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

                # Store
                if (tb_code, split) not in self.genre_combination_clusters[genre_combination]:
                    self.genre_combination_clusters[genre_combination][(tb_code, split)] = []

                self.genre_combination_clusters[genre_combination][(tb_code, split)].append(
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

            for (tb_code, split), clusters in self.genre_combination_clusters[genre_combination].items():
                logger.debug(f"    Processing {tb_code}:{split} ({len(clusters)} clusters)")

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
            for (tb_code, split), clusters in treebank_clusters.items():
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
