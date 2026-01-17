"""Gaussian Mixture Model clustering for genre discovery."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class GMMClusterer:
    """Cluster sentences using Gaussian Mixture Models."""

    def __init__(
        self,
        n_components: Optional[int] = None,
        random_state: int = 42,
        pca_components: int = 0,
    ):
        """Initialize GMM clusterer.

        Args:
            n_components: Number of clusters (if None, inferred from data)
            random_state: Random seed
            pca_components: If > 0, apply PCA before clustering
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca_components = pca_components
        self.pca = None
        self.gmm = None

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
            f"Fitting GMM with {n_components} components on {len(embeddings)} samples"
        )

        # Apply PCA if requested
        if self.pca_components > 0:
            if self.pca_components < n_components:
                logger.warning(
                    f"PCA components ({self.pca_components}) < n_components ({n_components}). "
                    f"Falling back to {n_components} components."
                )
                self.pca_components = n_components

            self.pca = PCA(n_components=self.pca_components, random_state=self.random_state)
            embeddings = self.pca.fit_transform(embeddings)
            logger.info(f"Applied PCA: {embeddings.shape[1]} components")

        # Fit GMM
        self.gmm = GaussianMixture(
            n_components=n_components,
            random_state=self.random_state,
            verbose=1,
        )
        self.gmm.fit(embeddings)

        # Get cluster probabilities
        cluster_probs = self.gmm.predict_proba(embeddings)

        logger.info(f"GMM fit complete. BIC: {self.gmm.bic(embeddings):.2f}")

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

        # Apply PCA if it was used during fitting
        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)

        cluster_probs = self.gmm.predict_proba(embeddings)
        cluster_ids = cluster_probs.argmax(axis=1)

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

        return self.gmm.means_
