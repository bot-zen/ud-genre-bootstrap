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

        self.genre_mapper = GenreMapper()

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

        # Step 6: Export results
        logger.info("Step 6: Exporting results")
        results = self._export_results()

        logger.info("Bootstrap pipeline complete")
        return results

    def _generate_embeddings(self, treebank_filter: Optional[str] = None) -> Dict:
        """Generate embeddings for all treebanks.

        Checks cache first if configured, generates and caches otherwise.

        Args:
            treebank_filter: Optional treebank code to process only that treebank

        Returns:
            Dictionary: {(treebank_code, split): {'sent_ids': [...], 'embeddings': array}}
        """
        embeddings_by_tb = {}
        cache_dir = self.config.embeddings.cache_dir

        # Iterate over all treebanks and splits
        for tb_code, split, dataset in self.data_loader.iter_all_treebanks():
            # Skip if filtering by treebank and this isn't the one
            if treebank_filter and tb_code != treebank_filter:
                continue

            # Try loading from cache first
            if cache_dir:
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
                genres = self.data_loader.get_treebank_genres(tb_code)
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

        # Iterate through genre combinations that can be predicted
        for genre_combination in environment['predict']:
            if genre_combination not in self.genre_combination_clusters:
                continue

            # Get all clusters for this genre combination
            for (tb_code, split), clusters in self.genre_combination_clusters[genre_combination].items():
                logger.debug(f"Labeling {tb_code} {split} with genres {genre_combination}")

                for cluster in clusters:
                    cluster_emb = cluster['embedding']

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

                        # Only label if confidence exceeds threshold
                        if confidence >= min_confidence:
                            method = "bootstrap-labeled"
                        else:
                            # Low confidence - mark as inferred
                            method = "bootstrap-inferred"
                            logger.debug(
                                f"Low confidence ({confidence:.3f}) for cluster "
                                f"{cluster['cluster_id']} in {tb_code}"
                            )

                        # Store labels for all sentences in this cluster
                        for sent_id in cluster['sent_ids']:
                            self.final_labels[sent_id] = (best_genre, confidence, method)

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
