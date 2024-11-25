"""
Main entry point for news generation system.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from config import SessionLocal
from storage.qdrant_manager import QdrantManager
from agents.news_agent import NewsSelectionAgent
from news.generator import NewsGenerator
from ngconfig.openai_config import (
    OPENAI_CONFIG,
    NEWS_SETTINGS,
    STORAGE_SETTINGS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_environment():
    """Initialize environment and configurations."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found in environment")


def init_services() -> tuple[Session, QdrantManager]:
    """Initialize database and Qdrant connections."""
    db = SessionLocal()
    qdrant = QdrantManager(local=STORAGE_SETTINGS["local_qdrant"])
    return db, qdrant


def main():
    """Main execution function."""
    try:
        # Initialize environment and services
        init_environment()
        db_session, qdrant = init_services()
        today = datetime.now().strftime("%Y-%m-%d")

        try:
            # Initialize news selection agent
            selection_agent = NewsSelectionAgent(
                openai_model=OPENAI_CONFIG["model"],
                local_qdrant=STORAGE_SETTINGS["local_qdrant"],
                news_per_section=NEWS_SETTINGS["news_per_section"]
            )

            # Get news to generate for each section
            logger.info("Selecting important news for each section...")
            news_to_generate = selection_agent.select_all_sections()

            # Initialize news generator
            generator = NewsGenerator(db_session, qdrant)

            # Generate articles
            logger.info("Generating articles...")
            generated_articles = generator.generate_articles(
                news_to_generate=news_to_generate,
                date=today
            )

            # Log results
            logger.info(f"\nGenerated {len(generated_articles)} articles:")
            for article in generated_articles:
                logger.info(f"- {article.title} (Section: {article.section.name})")

            return generated_articles

        finally:
            db_session.close()

    except Exception as e:
        logger.error(f"Error in news generation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
