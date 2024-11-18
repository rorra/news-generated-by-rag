"""
MiniLM Embedder Module

This module implements text embedding using the MiniLM model through Sentence-BERT.
MiniLM is a compact yet powerful language model that provides high-quality
multilingual embeddings while being computationally efficient.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class MiniLMEmbedder(BaseEmbedder):
    """
    MiniLM-based text embedder.

    This embedder uses a pre-trained MiniLM model through Sentence-BERT to generate
    dense vector representations of text. It provides a good balance between
    performance and computational efficiency.

    Parameters
    ----------
    model_name : str, optional (default="sentence-transformers/all-MiniLM-L6-v2")
        The name or path of the pre-trained MiniLM model to use

    Attributes
    ----------
    model : SentenceTransformer
        The underlying MiniLM model
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the MiniLM embedder.

        Parameters
        ----------
        model_name : str, optional
            Name or path of the pre-trained model to load
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to MiniLM embedding vector.

        Parameters
        ----------
        text : str
            Input text to be embedded

        Returns
        -------
        np.ndarray
            Normalized MiniLM embedding vector
        """
        return self.model.encode(text, normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the MiniLM embeddings.

        Returns
        -------
        int
            Size of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()

    @property
    def collection_name(self) -> str:
        """
        Get the name of the Qdrant collection for MiniLM embeddings.

        Returns
        -------
        str
            Collection name for storing MiniLM embeddings
        """
        return "news_minilm"
