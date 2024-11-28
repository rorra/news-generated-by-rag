"""
Main module for article preprocessing.

This module provides the entry point for the article preprocessing system.
It sets up logging, initializes the preprocessing pipeline, and orchestrates
the processing of articles.

Usage:
    python main.py
"""

import nltk
from typing import List
import logging
from config import SessionLocal
from preprocessors.base import TextPreprocessor
from preprocessors.spelling import SpellingCorrector
from preprocessors.duplicate_remover import DuplicateRemover
from preprocessors.case_normalizer import CaseNormalizer
from preprocessors.text_normalizer import TextNormalizer
from preprocessors.content_cleaner import ContentCleaner
from preprocessors.paragraph_segmenter import ParagraphSegmenter
from services.article_processor import ArticleProcessor


def setup_logging():
    """
    Configure logging for the application.

    Sets up basic logging configuration with appropriate format and level.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def download_nltk_data():
    """
    Download required NLTK data.

    Downloads the necessary NLTK resources for text processing.
    """
    nltk_resources = ['punkt', 'punkt_tab']
    for resource in nltk_resources:
        nltk.download(resource)


def create_preprocessor_pipeline() -> List[TextPreprocessor]:
    """
    Create and return the preprocessing pipeline.

    Returns:
        List[TextPreprocessor]: A list of preprocessor instances in the order
        they should be applied.
    """
    return [
        SpellingCorrector(language='es'),
        DuplicateRemover(language='spanish'),
        CaseNormalizer(),  # Convert to lowercase before normalization
        TextNormalizer(model='es_core_news_sm'),
        ContentCleaner(),
        ParagraphSegmenter(language='spanish') # No really needed
    ]


def main():
    """
    Main function to run the article preprocessing system.

    This function:
    1. Sets up logging
    2. Downloads required NLTK data
    3. Creates the preprocessing pipeline
    4. Processes all unprocessed articles
    5. Handles any errors that occur during processing
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    download_nltk_data()

    preprocessors = create_preprocessor_pipeline()
    processor = ArticleProcessor(preprocessors, batch_size=500)

    db = SessionLocal()
    try:
        logger.info("Starting article processing...")
        total_processed = processor.process_and_store(db)
        logger.info(f"Successfully processed {total_processed} articles")

    except Exception as e:
        logger.error(f"Error processing articles: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    main()
