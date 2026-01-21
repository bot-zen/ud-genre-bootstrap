"""Gaussian Mixture Model clustering for genre discovery."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GMMClusterer:
    """Cluster sentences using Gaussian Mixture Models."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: int = 42,
        pca_components: int = 0,
        device: str = "auto",
    ):
        """Initialize GMM clusterer.

        Args:
            n_components: Number of clusters (if None, inferred from data)
            random_state: Random seed
            pca_components: If > 0, apply PCA before clustering
            device: Device to use ("auto", "cuda", or "cpu")
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca_components = pca_components
        self.device = self._determine_device(device)
        self.use_gpu = self.device == "cuda"

        # Import appropriate libraries based on device
        self._import_libraries()

        self.pca = None
        self.gmm = None

    def _determine_device(self, device: str) -> str:
        """Determine which device to use.

        Args:
            device: User-specified device ("auto", "cuda", or "cpu")

        Returns:
            Device string ("cuda" or "cpu")
        """
        if device == "cpu":
            return "cpu"

        if device in ("auto", "cuda"):
            try:
                import cupy as cp
                # Try to initialize CUDA
                cp.cuda.Device(0).compute_capability
                if device == "auto":
                    logger.info("GPU detected, using CUDA for GMM clustering")
                return "cuda"
            except (ImportError, Exception):
                if device == "cuda":
                    logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"

        return "cpu"

    def _import_libraries(self):
        """Import appropriate GMM and PCA libraries based on device."""
        if self.use_gpu:
            # NOTE: cuML does not currently support GaussianMixture
            # See: https://github.com/rapidsai/cuml/issues
            # Available GPU algorithms: KMeans, DBSCAN, HDBSCAN, AgglomerativeClustering
            logger.warning(
                "GPU acceleration requested but cuML does not support Gaussian Mixture Models. "
                "Falling back to CPU (scikit-learn). "
                "For GPU clustering, consider using KMeans instead of GMM."
            )
            self.use_gpu = False
            self.device = "cpu"
            # Fall through to CPU imports

        if not self.use_gpu:
            from sklearn.mixture import GaussianMixture
            from sklearn.decomposition import PCA

            self.GaussianMixture = GaussianMixture
            self.PCA = PCA
            self.cp = None
            logger.info("Using scikit-learn (CPU) for GMM clustering")

    def fit(self, embeddings: np.ndarray, n_genres: int) -> np.ndarray:
        """Fit GMM to embeddings.

        Args:
            embeddings: Sentence embeddings [n_sentences, embedding_dim]
            n_genres: Number of genres (used as n_components)

        Returns:
            Cluster probability distributions [n_sentences, n_genres]
        """
        n_components = n_genres if self.n_components is None else self.n_components

        logger.info(
            f"Fitting GMM with {n_components} components on {len(embeddings)} samples "
            f"(device: {self.device})"
        )

        # Convert to GPU array if using CUDA
        if self.use_gpu:
            embeddings_input = self.cp.asarray(embeddings)
        else:
            embeddings_input = embeddings

        # Apply PCA if requested
        if self.pca_components > 0:
            if self.pca_components < n_components:
                logger.warning(
                    f"PCA components ({self.pca_components}) < n_components ({n_components}). "
                    f"Falling back to {n_components} components."
                )
                self.pca_components = n_components

            self.pca = self.PCA(n_components=self.pca_components, random_state=self.random_state)
            embeddings_input = self.pca.fit_transform(embeddings_input)
            logger.info(f"Applied PCA: {embeddings_input.shape[1]} components")

        # Fit GMM
        self.gmm = self.GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm.fit(embeddings_input)

        # Get cluster probabilities
        cluster_probs = self.gmm.predict_proba(embeddings_input)

        # Convert back to CPU if using GPU
        if self.use_gpu:
            cluster_probs = self.cp.asnumpy(cluster_probs)
            bic_value = float(self.cp.asnumpy(self.gmm.bic(embeddings_input)))
        else:
            bic_value = self.gmm.bic(embeddings_input)

        logger.info(f"GMM fit complete. BIC: {bic_value:.2f}")

        return cluster_probs

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster assignments for new embeddings.

        Args:
            embeddings: Sentence embeddings

        Returns:
            Tuple of (cluster_ids, probabilities)
        """
        if self.gmm is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to GPU array if using CUDA
        if self.use_gpu:
            embeddings_input = self.cp.asarray(embeddings)
        else:
            embeddings_input = embeddings

        # Apply PCA if it was used during fitting
        if self.pca is not None:
            embeddings_input = self.pca.transform(embeddings_input)

        cluster_probs = self.gmm.predict_proba(embeddings_input)
        cluster_ids = cluster_probs.argmax(axis=1)

        # Convert back to CPU if using GPU
        if self.use_gpu:
            cluster_ids = self.cp.asnumpy(cluster_ids)
            cluster_probs = self.cp.asnumpy(cluster_probs)

        return cluster_ids, cluster_probs

    def cluster_treebank(
        self,
        embeddings: np.ndarray,
        sent_ids: List[str],
        n_genres: int,
    ) -> Dict:
        """Cluster a treebank's sentences.

        Args:
            embeddings: Sentence embeddings
            sent_ids: Sentence IDs
            n_genres: Number of expected genres

        Returns:
            Dictionary with clustering results
        """
        if len(embeddings) < n_genres:
            logger.warning(
                f"Only {len(embeddings)} sentences but {n_genres} genres expected. "
                f"Using {len(embeddings)} clusters."
            )
            n_genres = len(embeddings)

        # Fit and predict
        cluster_probs = self.fit(embeddings, n_genres)
        cluster_ids = cluster_probs.argmax(axis=1)

        # Organize by clusters
        clusters = {}
        for cluster_id in range(n_genres):
            mask = cluster_ids == cluster_id
            clusters[cluster_id] = {
                "sent_ids": [sid for sid, m in zip(sent_ids, mask) if m],
                "size": mask.sum(),
                "confidence": cluster_probs[mask, cluster_id].mean(),
            }

        return {
            "n_clusters": n_genres,
            "cluster_ids": cluster_ids,
            "cluster_probs": cluster_probs,
            "clusters": clusters,
        }

    def get_cluster_centroids(self) -> np.ndarray:
        """Get cluster centroid embeddings.

        Returns:
            Centroid embeddings [n_clusters, embedding_dim]
        """
        if self.gmm is None:
            raise ValueError("Model not fitted.")

        centroids = self.gmm.means_

        # Convert back to CPU if using GPU
        if self.use_gpu:
            centroids = self.cp.asnumpy(centroids)

        return centroids
