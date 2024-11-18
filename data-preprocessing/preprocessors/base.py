"""
Base module for text preprocessing.

This module defines the base abstract class that all preprocessors must implement.
Each preprocessor is responsible for a specific text transformation task.
"""

from abc import ABC, abstractmethod


class TextPreprocessor(ABC):
    """
    Abstract base class for text preprocessors.

    All preprocessors in the pipeline must inherit from this class and implement
    the process method. This ensures a consistent interface across all preprocessing
    steps.
    """

    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process the input text and return the processed result.

        Args:
            text (str): The input text to be processed.

        Returns:
            str: The processed text.

        Raises:
            NotImplementedError: If the child class doesn't implement this method.
        """
        pass
