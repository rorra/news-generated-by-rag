"""
SBERT Embedder Module

This module implements sentence embeddings using Sentence-BERT (SBERT) models.
It specifically uses a Spanish-tuned model by default for better performance
with Spanish text.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from .base import BaseEmbedder


class SBERTEmbedder(BaseEmbedder):
    """
    Sentence-BERT (SBERT) based text embedder.

    This embedder uses a pre-trained SBERT model to generate dense vector
    representations of text. It's particularly well-suited for semantic
    similarity tasks and doesn't require fitting.

    Parameters
    ----------
    model_name : str, optional (default="hiiamsid/sentence_similarity_spanish_es")
        The name or path of the pre-trained SBERT model to use.
        Default is a Spanish-tuned model.

    Attributes
    ----------
    model : SentenceTransformer
        The underlying SBERT model
    """

    def __init__(self, model_name: str = "hiiamsid/sentence_similarity_spanish_es"):
        """
        Initialize the SBERT embedder.

        Parameters
        ----------
        model_name : str, optional
            Name or path of the pre-trained model to load
        """
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to SBERT embedding vector.

        Parameters
        ----------
        text : str
            Input text to be embedded

        Returns
        -------
        np.ndarray
            Normalized SBERT embedding vector
        """
        return self.model.encode(text, normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the SBERT embeddings.

        Returns
        -------
        int
            Size of the embedding vectors
        """
        return self.model.get_sentence_embedding_dimension()

    @property
    def collection_name(self) -> str:
        """
        Get the name of the Qdrant collection for SBERT embeddings.

        Returns
        -------
        str
            Collection name for storing SBERT embeddings
        """
        return "news_sbert"
