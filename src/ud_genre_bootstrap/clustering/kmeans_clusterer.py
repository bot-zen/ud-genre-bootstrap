"""K-Means clustering for genre discovery with GPU support."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KMeansClusterer:
    """Cluster sentences using K-Means.

    Unlike GMM, K-Means has GPU support via cuML.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: int = 42,
        pca_components: int = 0,
        device: str = "auto",
        max_iter: int = 300,
    ):
        """Initialize K-Means clusterer.

        Args:
            n_components: Number of clusters (if None, inferred from data)
            random_state: Random seed
            pca_components: If > 0, apply PCA before clustering
            device: Device to use ("auto", "cuda", or "cpu")
            max_iter: Maximum number of iterations
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca_components = pca_components
        self.max_iter = max_iter
        self.device = self._determine_device(device)
        self.use_gpu = self.device == "cuda"

        # Import appropriate libraries based on device
        self._import_libraries()

        self.pca = None
        self.kmeans = None

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
                    logger.info("GPU detected, using CUDA for K-Means clustering")
                return "cuda"
            except (ImportError, Exception) as e:
                if device == "cuda":
                    logger.warning(f"CUDA requested but not available ({e}), falling back to CPU")
                return "cpu"

        return "cpu"

    def _import_libraries(self):
        """Import appropriate K-Means and PCA libraries based on device."""
        if self.use_gpu:
            try:
                from cuml.cluster import KMeans as cuKMeans
                from cuml.decomposition import PCA as cuPCA
                import cupy as cp

                self.KMeans = cuKMeans
                self.PCA = cuPCA
                self.cp = cp
                logger.info("Using cuML (GPU) for K-Means clustering")
            except ImportError as e:
                logger.warning(
                    f"cuML import failed ({e}), falling back to CPU. "
                    "Install cuML for GPU acceleration: pip install cuml-cu12"
                )
                self.use_gpu = False
                self.device = "cpu"
                # Fall through to CPU imports

        if not self.use_gpu:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            self.KMeans = KMeans
            self.PCA = PCA
            self.cp = None
            logger.info("Using scikit-learn (CPU) for K-Means clustering")

    def fit(self, embeddings: np.ndarray, n_genres: int) -> np.ndarray:
        """Fit K-Means to embeddings.

        Args:
            embeddings: Sentence embeddings [n_sentences, embedding_dim]
            n_genres: Number of genres (used as n_clusters)

        Returns:
            Distance matrix [n_sentences, n_genres] (negative distances for soft assignments)
        """
        n_components = n_genres if self.n_components is None else self.n_components

        logger.info(
            f"Fitting K-Means with {n_components} clusters on {len(embeddings)} samples "
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

        # Fit K-Means
        kmeans_kwargs = {
            "n_clusters": n_components,
            "random_state": self.random_state,
            "max_iter": self.max_iter,
        }

        # cuML uses different parameter names
        if self.use_gpu:
            kmeans_kwargs["init"] = "k-means++"
            kmeans_kwargs["n_init"] = 10
        else:
            kmeans_kwargs["n_init"] = "auto"

        self.kmeans = self.KMeans(**kmeans_kwargs)
        self.kmeans.fit(embeddings_input)

        # Get distances to cluster centers
        # K-Means doesn't provide probabilities like GMM, so we convert distances to pseudo-probabilities
        distances = self._compute_distances(embeddings_input)

        # Convert distances to soft assignments (closer = higher probability)
        # Use negative exponential of squared distances
        cluster_probs = self._distances_to_probs(distances)

        # Convert back to CPU if using GPU
        if self.use_gpu:
            cluster_probs = self.cp.asnumpy(cluster_probs)
            inertia = float(self.cp.asnumpy(self.kmeans.inertia_))
        else:
            inertia = self.kmeans.inertia_

        logger.info(f"K-Means fit complete. Inertia: {inertia:.2f}")

        return cluster_probs

    def _compute_distances(self, embeddings):
        """Compute distances from points to cluster centers.

        Args:
            embeddings: Input embeddings

        Returns:
            Distance matrix [n_samples, n_clusters]
        """
        if self.use_gpu:
            # cuML KMeans has transform() method that returns distances
            if hasattr(self.kmeans, 'transform'):
                return self.kmeans.transform(embeddings)
            else:
                # Manually compute distances
                centers = self.kmeans.cluster_centers_
                distances = self.cp.zeros((embeddings.shape[0], centers.shape[0]))
                for i in range(centers.shape[0]):
                    diff = embeddings - centers[i]
                    distances[:, i] = self.cp.sqrt(self.cp.sum(diff ** 2, axis=1))
                return distances
        else:
            # scikit-learn KMeans has transform() method
            return self.kmeans.transform(embeddings)

    def _distances_to_probs(self, distances, temperature: float = 1.0):
        """Convert distances to probability-like soft assignments.

        Uses softmax with negative distances: closer points get higher probabilities.

        Args:
            distances: Distance matrix [n_samples, n_clusters]
            temperature: Temperature for softmax (lower = harder assignments)

        Returns:
            Probability matrix [n_samples, n_clusters]
        """
        if self.use_gpu:
            # Negative distances (closer = higher value)
            scores = -distances / temperature
            # Softmax
            exp_scores = self.cp.exp(scores - self.cp.max(scores, axis=1, keepdims=True))
            probs = exp_scores / self.cp.sum(exp_scores, axis=1, keepdims=True)
            return probs
        else:
            # Negative distances
            scores = -distances / temperature
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            return probs

    def predict(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict cluster assignments for new embeddings.

        Args:
            embeddings: Sentence embeddings

        Returns:
            Tuple of (cluster_ids, probabilities)
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert to GPU array if using CUDA
        if self.use_gpu:
            embeddings_input = self.cp.asarray(embeddings)
        else:
            embeddings_input = embeddings

        # Apply PCA if it was used during fitting
        if self.pca is not None:
            embeddings_input = self.pca.transform(embeddings_input)

        # Get distances and convert to probabilities
        distances = self._compute_distances(embeddings_input)
        cluster_probs = self._distances_to_probs(distances)
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
        compute_metrics: bool = True,
    ) -> Dict:
        """Cluster a treebank's sentences.

        Args:
            embeddings: Sentence embeddings
            sent_ids: Sentence IDs
            n_genres: Number of expected genres
            compute_metrics: Whether to compute cluster quality metrics

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

        # Compute cluster quality metrics
        metrics = {}
        if compute_metrics and n_genres > 1:  # Need at least 2 clusters for metrics
            try:
                from ud_genre_bootstrap.evaluation.metrics import ClusterQualityMetrics
                metrics = ClusterQualityMetrics.compute_all(embeddings, cluster_ids)
                logger.debug(f"Cluster metrics: {metrics}")
            except Exception as e:
                logger.warning(f"Failed to compute cluster metrics: {e}")

        return {
            "n_clusters": n_genres,
            "cluster_ids": cluster_ids,
            "cluster_probs": cluster_probs,
            "clusters": clusters,
            "metrics": metrics,
        }

    def get_cluster_centroids(self) -> np.ndarray:
        """Get cluster centroid embeddings.

        Returns:
            Centroid embeddings [n_clusters, embedding_dim]
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted.")

        centroids = self.kmeans.cluster_centers_

        # Convert back to CPU if using GPU
        if self.use_gpu:
            centroids = self.cp.asnumpy(centroids)

        return centroids
