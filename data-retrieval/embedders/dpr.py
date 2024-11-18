"""
DPR Embedder Module

This module implements text embedding using Dense Passage Retrieval (DPR)
question encoder. While primarily designed for question-answering tasks,
it can be effectively used for general text embedding.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, DPRQuestionEncoder
from .base import BaseEmbedder


class DPREmbedder(BaseEmbedder):
    """
    Dense Passage Retrieval (DPR) based text embedder.

    This embedder uses a pre-trained DPR question encoder to generate
    dense vector representations of text. It's particularly effective
    for query/question embedding in retrieval tasks.

    Parameters
    ----------
    model_name : str, optional (default="facebook/dpr-question_encoder-single-nq-base")
        The name or path of the pre-trained DPR model to use

    Attributes
    ----------
    tokenizer : AutoTokenizer
        Tokenizer for the DPR model
    model : DPRQuestionEncoder
        The underlying DPR model

    """

    def __init__(self, model_name: str = "facebook/dpr-question_encoder-single-nq-base"):
        """
        Initialize the DPR embedder.

        Parameters
        ----------
        model_name : str, optional
            Name or path of the pre-trained model to load
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DPRQuestionEncoder.from_pretrained(model_name)
        self.model.eval()  # Set to evaluation mode

    def embed(self, text: str) -> np.ndarray:
        """
        Convert text to DPR embedding vector.

        Parameters
        ----------
        text : str
            Input text to be embedded

        Returns
        -------
        np.ndarray
            Normalized DPR embedding vector
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Normalize the embeddings
        embeddings = outputs.pooler_output[0].numpy()
        return embeddings / np.linalg.norm(embeddings)

    @property
    def dimension(self) -> int:
        """
        Get the dimension of the DPR embeddings.

        Returns
        -------
        int
            Size of the embedding vectors (768 for base model)
        """
        return 768

    @property
    def collection_name(self) -> str:
        """
        Get the name of the Qdrant collection for DPR embeddings.

        Returns
        -------
        str
            Collection name for storing DPR embeddings
        """
        return "news_dpr"
