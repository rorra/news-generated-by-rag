"""
Content cleaning module.

This module handles the removal of irrelevant content and links from text.
"""

from .base import TextPreprocessor
import re


class ContentCleaner(TextPreprocessor):
    """
    A preprocessor for cleaning irrelevant content from text.

    Removes specific patterns like "Lee también" phrases and URLs that
    don't contribute to the main content.
    """

    def __init__(self):
        """Initialize the ContentCleaner with predefined patterns to remove."""
        self.irrelevant_patterns = [
            r'Lee también.*?\. ',  # Remove "Lee también" phrases
            r'http[s]?://\S+|www\.\S+'  # Remove URLs
        ]

    def process(self, text: str) -> str:
        """
        Clean the input text by removing irrelevant content.

        Args:
            text (str): The input text to clean.

        Returns:
            str: The cleaned text with irrelevant content removed.
        """
        for pattern in self.irrelevant_patterns:
            text = re.sub(pattern, '', text)
        return text
