"""Generate sentence embeddings using transformer models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate sentence embeddings using multilingual transformer models."""

    def __init__(
        self,
        model_name: str = "xlm-roberta-base",
        pooling: str = "mean",
        layer: int = -1,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """Initialize embedding generator.

        Args:
            model_name: HuggingFace model identifier
            pooling: Pooling strategy ('mean' or 'cls')
            layer: Which layer to use for embeddings (-1 = last)
            batch_size: Batch size for encoding
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.model_name = model_name
        self.pooling = pooling
        self.layer = layer
        self.batch_size = batch_size
        self._device_str = device
        self.device = self._get_device(device)

        # Lazy-load model and tokenizer on first use
        self.tokenizer = None
        self.model = None
        self._model_loaded = False

    def _get_device(self, device: Optional[str]) -> torch.device:
        """Determine which device to use.

        Args:
            device: User-specified device or 'auto'

        Returns:
            torch.device
        """
        if device == "auto" or device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_model(self):
        """Load model and tokenizer (lazy loading).

        Called automatically before first use to avoid loading when not needed.
        """
        if self._model_loaded:
            return

        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self._model_loaded = True

        logger.info(f"Model loaded on device: {self.device}")

    def _pool_embeddings(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Pool token embeddings to sentence embeddings.

        Args:
            token_embeddings: Token-level embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sentence embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            # Mean pooling over non-masked tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask

        elif self.pooling == "cls":
            # Use [CLS] token (first token)
            return token_embeddings[:, 0, :]

        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

    @torch.no_grad()
    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for a list of sentences.

        Args:
            sentences: List of sentence strings

        Returns:
            Numpy array of embeddings [num_sentences, hidden_dim]
        """
        # Lazy-load model on first use
        self._load_model()

        all_embeddings = []

        # Process in batches
        for i in tqdm(range(0, len(sentences), self.batch_size), desc="Embedding"):
            batch = sentences[i : i + self.batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Get embeddings
            outputs = self.model(**encoded, output_hidden_states=True)
            token_embeddings = outputs.hidden_states[self.layer]

            # Pool to sentence level
            sentence_embeddings = self._pool_embeddings(
                token_embeddings, encoded["attention_mask"]
            )

            # Move to CPU and convert to numpy
            all_embeddings.append(sentence_embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def embed_dataset(
        self, dataset: Dataset, text_column: str = "text"
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for all sentences in a dataset.

        Args:
            dataset: HuggingFace Dataset
            text_column: Name of column containing text

        Returns:
            Dictionary with 'sent_id' and 'embedding' arrays
        """
        # Extract sentences
        sentences = [item[text_column] for item in dataset]
        sent_ids = [item["sent_id"] for item in dataset]

        # Generate embeddings
        embeddings = self.embed_sentences(sentences)

        return {"sent_id": sent_ids, "embedding": embeddings}

    def embed_treebank(
        self,
        treebank_code: str,
        split: str,
        dataset: Dataset,
        output_path: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """Embed a full treebank split and optionally save to disk.

        Args:
            treebank_code: Treebank code (e.g., 'en_ewt')
            split: Split name ('train', 'dev', 'test')
            dataset: Dataset to embed
            output_path: If provided, save embeddings to this path

        Returns:
            Dictionary with embeddings
        """
        logger.info(f"Embedding {treebank_code} {split} ({len(dataset)} sentences)")

        result = self.embed_dataset(dataset)

        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save as numpy array
            save_path = output_path / f"{treebank_code}-{split}.npy"
            np.save(save_path, result["embedding"])
            logger.info(f"Saved embeddings to {save_path}")

            # Also save sent_id mapping
            mapping_path = output_path / f"{treebank_code}-{split}_ids.txt"
            with open(mapping_path, "w") as f:
                f.write("\n".join(result["sent_id"]))

        return result

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of the embeddings.

        Returns:
            Embedding dimension
        """
        # Lazy-load model if needed
        self._load_model()
        return self.model.config.hidden_size

    @staticmethod
    def load_embeddings(
        treebank_code: str, split: str, cache_dir: Path
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load cached embeddings from disk if they exist.

        Args:
            treebank_code: Treebank code (e.g., 'en_ewt')
            split: Split name ('train', 'dev', 'test')
            cache_dir: Directory containing cached embeddings

        Returns:
            Dictionary with embeddings or None if not found
        """
        cache_dir = Path(cache_dir)
        embedding_path = cache_dir / f"{treebank_code}-{split}.npy"
        ids_path = cache_dir / f"{treebank_code}-{split}_ids.txt"

        if not embedding_path.exists() or not ids_path.exists():
            return None

        try:
            embeddings = np.load(embedding_path)
            with open(ids_path, "r") as f:
                sent_ids = [line.strip() for line in f.readlines()]

            logger.info(f"Loaded cached embeddings from {embedding_path}")
            return {"sent_id": sent_ids, "embedding": embeddings}
        except Exception as e:
            logger.warning(f"Failed to load cached embeddings: {e}")
            return None
