"""
Spelling correction module.

This module provides functionality for correcting spelling and grammar errors
in Spanish text using the LanguageTool library.
"""

from .base import TextPreprocessor
import language_tool_python
from functools import lru_cache


class SpellingCorrector(TextPreprocessor):
    """
    A preprocessor for correcting spelling and grammar errors.

    Uses LanguageTool to perform spelling and grammar corrections. The LanguageTool
    instance is cached to avoid creating multiple instances.
    """

    def __init__(self, language: str = 'es'):
        """
        Initialize the SpellingCorrector.

        Args:
            language (str): The language code for spell checking (default: 'es' for Spanish)
        """
        self.language = language
        self.tool = self._get_language_tool()

    @lru_cache(maxsize=1)
    def _get_language_tool(self):
        """
        Create and cache a LanguageTool instance.

        Returns:
            LanguageTool: A cached instance of LanguageTool.
        """
        return language_tool_python.LanguageTool(self.language)

    def process(self, text: str) -> str:
        """
        Correct spelling and grammar in the input text.

        Args:
            text (str): The text to be corrected.

        Returns:
            str: The text with spelling and grammar corrections applied.
        """
        return self.tool.correct(text)