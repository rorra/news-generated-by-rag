"""
News Generator Module

Handles the generation of news articles using writer agents and stores them in the database.
"""

from typing import Dict, List
from datetime import datetime
import logging
from sqlalchemy.orm import Session
from agents.writer_agent import WriterFactory
from models.db_models import GeneratedNews, Section
from storage.qdrant_manager import QdrantManager
from ngconfig.openai_config import WRITER_SETTINGS
import json

logger = logging.getLogger(__name__)


class NewsGenerator:
    """Handles news generation and storage."""

    def __init__(self, db_session: Session, qdrant: QdrantManager):
        """Initialize generator with database session and Qdrant manager."""
        self.session = db_session
        self.writer_factory = WriterFactory(qdrant, db_session)

    def _get_section_id(self, section_name: str) -> int:
        """Get section ID from name."""
        section = self.session.query(Section).filter(Section.name == section_name).first()
        if not section:
            raise ValueError(f"Section not found: {section_name}")
        return section.id

    def _store_article(self, title: str, body: str, section_name: str) -> GeneratedNews:
        """Store generated article in database."""
        try:
            section_id = self._get_section_id(section_name)

            generated_news = GeneratedNews(
                section_id=section_id,
                title=title,
                body=body,
                generated_at=datetime.utcnow()
            )

            self.session.add(generated_news)
            self.session.commit()

            return generated_news

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error storing generated article: {str(e)}")
            raise

    def generate_articles(self, news_to_generate: Dict[str, List[str]], date: str) -> List[GeneratedNews]:
        """
        Generate articles for selected news.

        Args:
            news_to_generate: Dictionary mapping sections to lists of titles
            date: Date for RAG context in YYYY-MM-DD format

        Returns:
            List of generated news articles
        """
        generated_articles = []

        for section, titles in news_to_generate.items():
            logger.info(f"Generating articles for section: {section}")

            for title in titles:
                try:
                    # Get random writer
                    writer = self.writer_factory.get_random_writer()
                    logger.info(f"Selected writer type: {writer.__class__.__name__}")

                    # Generate article
                    llm_generated_article = writer.generate_article(
                        title=title,
                        section=section,
                        date=date
                    )

                    # Store in database
                    article = json.loads(llm_generated_article.content)
                    generated_article = self._store_article(
                        title=article['titulo'],
                        body=article['cuerpo'],
                        section_name=section
                    )

                    generated_articles.append(generated_article)
                    logger.info(f"Generated and stored article: {title}")

                except Exception as e:
                    logger.error(f"Error generating article '{title}': {str(e)}")
                    continue

        return generated_articles
