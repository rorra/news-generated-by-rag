"""
BM25 Embedder Module

This module implements text embedding using the BM25 (Best Matching 25) algorithm.
BM25 is a probabilistic ranking function used for information retrieval, adapted
here to produce fixed-size vector representations compatible with vector stores.
"""

from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from .base import BaseEmbedder


class BM25Embedder(BaseEmbedder):
    """
    BM25-based text embedder.

    This embedder uses the BM25 algorithm to convert text into numerical vectors.
    It requires fitting on a corpus before it can be used for embedding. The BM25
    scores are normalized and padded/truncated to produce fixed-size vectors.

    Parameters
    ----------
    dimension : int, optional (default=384)
        Size of the output vector, matches TF-IDF dimension for consistency

    Attributes
    ----------
    bm25 : BM25Okapi or None
        The underlying BM25 model, None until fitted
    corpus : List[List[str]] or None
        The tokenized corpus used for fitting
    _dimension : int
        The fixed size of output vectors
    """

    def __init__(self):
        """Initialize the BM25 embedder."""
        self.bm25 = None
        self.corpus = None
        self._dimension = 384  # Same as TF-IDF for consistency

    def fit(self, texts: List[str]) -> None:
        """
        Fit the BM25 model on a corpus of texts.

        Parameters
        ----------
        texts : List[str]
            List of documents to prepare the BM25 model with

        Returns
        -------
        None
        """
        self.corpus = [text.split() for text in texts]
        self.bm25 = BM25Okapi(self.corpus)

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to BM25-based vector representation.

        Parameters
        ----------
        text : str
            Input text to be embedded

        Returns
        -------
        np.ndarray
            Normalized vector of BM25 scores

        Raises
        ------
        ValueError
            If the BM25 model hasn't been fitted yet
        """
        if self.bm25 is None:
            raise ValueError("BM25 needs to be fitted first")

        query_tokens = text.split()
        scores = self.bm25.get_scores(query_tokens)

        # Pad or truncate to match dimension
        if len(scores) < self.dimension:
            scores = np.pad(scores, (0, self.dimension - len(scores)))
        else:
            scores = scores[:self.dimension]

        # Normalize the scores
        norm = np.linalg.norm(scores)
        if norm > 0:
            scores = scores / norm

        return scores

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the BM25 vectors.

        Returns
        -------
        int
            Fixed size of the output vectors
        """
        return self._dimension

    @property
    def collection_name(self) -> str:
        """
        Get the name of the Qdrant collection for BM25 embeddings.

        Returns
        -------
        str
            Collection name for storing BM25 embeddings
        """
        return "news_bm25"
