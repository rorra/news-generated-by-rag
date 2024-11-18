"""
Text normalization module.

This module handles text normalization tasks including:
- Removing extra whitespace
- Removing stop words
- Lemmatization
"""

from .base import TextPreprocessor
import re
import spacy
from functools import lru_cache


class TextNormalizer(TextPreprocessor):
    """
    A preprocessor for normalizing text.

    Uses spaCy for advanced text processing including lemmatization and
    stop word removal. The spaCy model is cached for efficiency.
    """

    def __init__(self, model: str = 'es_core_news_sm'):
        """
        Initialize the TextNormalizer.

        Args:
            model (str): The name of the spaCy model to use (default: 'es_core_news_sm')
        """
        self.model_name = model
        self.nlp = self._load_spacy_model()

    @lru_cache(maxsize=1)
    def _load_spacy_model(self):
        """
        Load and cache the spaCy model.

        Returns:
            spacy.Language: A loaded spaCy model.
        """
        return spacy.load(self.model_name)

    def process(self, text: str) -> str:
        """
        Normalize the input text.

        Performs the following operations:
        1. Removes extra whitespace
        2. Removes stop words
        3. Applies lemmatization

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.
        """
        text = re.sub(r'\s+', ' ', text).strip()
        doc = self.nlp(text)
        normalized_words = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        return ' '.join(normalized_words)
