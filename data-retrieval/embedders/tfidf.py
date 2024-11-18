"""
TF-IDF Embedder Module

This module implements the TF-IDF (Term Frequency-Inverse Document Frequency)
embedding approach. It uses scikit-learn's TfidfVectorizer to convert text
into fixed-size vector representations.
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .base import BaseEmbedder


class TfidfEmbedder(BaseEmbedder):
    """
    TF-IDF based text embedder.

    This embedder uses TF-IDF vectorization to convert text into numerical vectors.
    It requires fitting on a corpus before it can be used for embedding.

    Parameters
    ----------
    max_features : int, optional (default=384)
        Maximum number of features (terms) to include in the vocabulary

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The underlying scikit-learn TF-IDF vectorizer
    fitted : bool
        Indicates whether the vectorizer has been fitted to a corpus

    """

    def __init__(self, max_features: int = 384):
        """
        Initialize the TF-IDF embedder.

        Parameters
        ----------
        max_features : int, optional (default=384)
            Maximum number of features in the TF-IDF vocabulary
        """
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.fitted = False

    def fit(self, texts: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on a corpus of texts.

        Parameters
        ----------
        texts : List[str]
            List of documents to learn the vocabulary from

        Returns
        -------
        None
        """
        self.vectorizer.fit(texts)
        self.fitted = True

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to TF-IDF vector representation.

        Parameters
        ----------
        text : str
            Input text to be embedded

        Returns
        -------
        np.ndarray
            TF-IDF vector representation of the input text

        Raises
        ------
        ValueError
            If the vectorizer hasn't been fitted yet
        """
        if not self.fitted:
            raise ValueError("TfidfVectorizer needs to be fitted first")
        vector = self.vectorizer.transform([text]).toarray()
        return vector[0]

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the TF-IDF vectors.

        Returns
        -------
        int
            Number of features in the TF-IDF vocabulary
        """
        return self.vectorizer.max_features

    @property
    def collection_name(self) -> str:
        """
        Get the name of the Qdrant collection for TF-IDF embeddings.

        Returns
        -------
        str
            Collection name for storing TF-IDF embeddings
        """
        return "news_tfidf"
