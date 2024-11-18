"""
Duplicate sentence removal module.

This module handles the removal of duplicate sentences from text while
maintaining the original order of first appearance.
"""

from .base import TextPreprocessor
import nltk
from typing import List


class DuplicateRemover(TextPreprocessor):
    """
    A preprocessor for removing duplicate sentences from text.

    Uses NLTK's sentence tokenizer to split text into sentences and removes
    any duplicates while preserving the original order.
    """

    def __init__(self, language: str = 'spanish'):
        """
        Initialize the DuplicateRemover.

        Args:
            language (str): The language for sentence tokenization (default: 'spanish')
        """
        self.language = language

    def _tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using NLTK.

        Args:
            text (str): The text to be split into sentences.

        Returns:
            List[str]: A list of sentence strings.
        """
        return nltk.sent_tokenize(text, language=self.language)

    def process(self, text: str) -> str:
        """
        Remove duplicate sentences from the text.

        Args:
            text (str): The input text.

        Returns:
            str: The text with duplicate sentences removed.
        """
        sentences = self._tokenize_sentences(text)
        unique_sentences = list(dict.fromkeys(sentences))
        return ' '.join(unique_sentences)
