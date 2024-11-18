"""
Base Embedder Module

This module provides the abstract base class for all embedder implementations.
Each concrete embedder class must inherit from BaseEmbedder and implement
all abstract methods.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEmbedder(ABC):
    """
    Abstract base class for all embedding models.

    This class defines the interface that all embedder implementations must follow.
    Each embedder should be able to convert text into a fixed-size vector representation
    and provide information about its vector dimensions and collection name in Qdrant.

    Methods
    -------
    embed(text: str) -> np.ndarray:
        Convert input text into a vector representation

    Properties
    ----------
    dimension: int
        The size of the output embedding vector
    collection_name: str
        The name of the collection where embeddings will be stored in Qdrant
    """

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to embedding vector.

        Parameters
        ----------
        text : str
            The input text to be embedded

        Returns
        -------
        np.ndarray
            The embedding vector representation of the input text
        """
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """
        Return the dimension of the embedding vector.

        Returns
        -------
        int
            The size of the embedding vector
        """
        pass

    @property
    @abstractmethod
    def collection_name(self) -> str:
        """
        Return the name of the collection for this embedder.

        Returns
        -------
        str
            The name of the Qdrant collection where embeddings will be stored
        """
        pass
