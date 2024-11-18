"""
Case normalization module.

This module handles converting text to lowercase for consistent text processing.
"""

from .base import TextPreprocessor


class CaseNormalizer(TextPreprocessor):
    """
    A preprocessor for normalizing text case.

    Converts all text to lowercase to ensure consistent text processing
    in subsequent steps and during retrieval operations.
    """

    def process(self, text: str) -> str:
        """
        Convert input text to lowercase.

        Args:
            text (str): The input text to convert.

        Returns:
            str: The text converted to lowercase.
        """
        return text.lower()
