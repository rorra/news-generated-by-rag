"""
Article processing service module.

This module provides the main service for processing articles through the
preprocessing pipeline and storing them in the database.
"""

from typing import List
from sqlalchemy.orm import Session
from preprocessors.base import TextPreprocessor
from models.db_models import Article, ProcessedArticle


class ArticleProcessor:
    """
    Service class for processing articles through the preprocessing pipeline.

    This class orchestrates the preprocessing of articles, applying each
    preprocessor in sequence and handling the storage of processed articles
    in the database.
    """

    def __init__(self, preprocessors: List[TextPreprocessor], batch_size: int = 100):
        """
        Initialize the ArticleProcessor.

        Args:
            preprocessors (List[TextPreprocessor]): List of preprocessor instances to apply
            batch_size (int): Number of articles to process in each batch (default: 100)
        """
        self.preprocessors = preprocessors
        self.batch_size = batch_size

    def process_text(self, text: str) -> str:
        """
        Apply all preprocessors to the text in sequence.

        Args:
            text (str): The input text to process.

        Returns:
            str: The fully processed text.
        """
        processed_text = text
        for preprocessor in self.preprocessors:
            processed_text = preprocessor.process(processed_text)
        return processed_text

    def process_article(self, article: Article) -> ProcessedArticle:
        """
        Process a single article and create a ProcessedArticle instance.

        Args:
            article (Article): The article to process.

        Returns:
            ProcessedArticle: A new ProcessedArticle instance with processed content.
        """
        processed_title = self.process_text(article.title)
        processed_content = self.process_text(article.content)

        return ProcessedArticle.from_article(
            article=article,
            processed_title=processed_title,
            processed_content=processed_content
        )

    def process_and_store(self, db: Session) -> int:
        """
        Process all unprocessed articles in batches and store them in the database.

        Args:
            db (Session): The database session to use.

        Returns:
            int: The total number of articles processed.

        Raises:
            Exception: If there's an error processing the articles.
        """
        total_processed = 0

        while True:
            # Query next batch of unprocessed articles
            articles = (
                db.query(Article)
                .filter(Article.processed_article == None)
                .limit(self.batch_size)
                .all()
            )

            if not articles:
                break

            try:
                for article in articles:
                    processed_article = self.process_article(article)
                    db.add(processed_article)
                    total_processed += 1

                    if total_processed % 10 == 0:
                        print(f"Processed {total_processed} articles...")

                db.commit()

            except Exception as e:
                print(f"Error processing batch: {e}")
                db.rollback()
                raise

        return total_processed


