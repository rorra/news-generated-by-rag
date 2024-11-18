"""
Paragraph segmentation module.

This module handles the segmentation of text into paragraphs based on
sentence count for improved readability.
"""

from .base import TextPreprocessor
import nltk
from typing import List


class ParagraphSegmenter(TextPreprocessor):
    """
    A preprocessor for segmenting text into paragraphs.

    Splits text into paragraphs with a specified number of sentences per
    paragraph to improve readability.
    """

    def __init__(self, language: str = 'spanish', sentences_per_paragraph: int = 4):
        """
        Initialize the ParagraphSegmenter.

        Args:
            language (str): The language for sentence tokenization (default: 'spanish')
            sentences_per_paragraph (int): Number of sentences per paragraph (default: 4)
        """
        self.language = language
        self.sentences_per_paragraph = sentences_per_paragraph

    def _segment_into_paragraphs(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into paragraphs.

        Args:
            sentences (List[str]): List of sentence strings.

        Returns:
            List[str]: List of paragraph strings.
        """
        paragraphs = []
        current_paragraph = []

        for i, sentence in enumerate(sentences, 1):
            current_paragraph.append(sentence)
            if i % self.sentences_per_paragraph == 0:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []

        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        return paragraphs

    def process(self, text: str) -> str:
        """
        Segment the input text into paragraphs.

        Args:
            text (str): The input text to segment.

        Returns:
            str: The text segmented into paragraphs.
        """
        sentences = nltk.sent_tokenize(text, language=self.language)
        paragraphs = self._segment_into_paragraphs(sentences)
        return '\n\n'.join(paragraphs)
