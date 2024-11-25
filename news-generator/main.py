"""
Main entry point for news generation system.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from agents.news_agent import NewsSelectionAgent
from ngconfig.openai_config import OPENAI_CONFIG, NEWS_SETTINGS

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


def main():
    """Main execution function."""
    try:
        init_environment()

        # Initialize news selection agent
        agent = NewsSelectionAgent(
            openai_model=OPENAI_CONFIG["model"],
            local_qdrant=True,
            news_per_section=NEWS_SETTINGS["news_per_section"]
        )

        # Get news to generate for each section
        news_to_generate = agent.select_all_sections()

        # Log selected news
        logger.info(f"Selected news for {datetime.now().strftime('%Y-%m-%d')}:")
        for section, titles in news_to_generate.items():
            logger.info(f"\n{section}:")
            for title in titles:
                logger.info(f"- {title}")

        return news_to_generate

    except Exception as e:
        logger.error(f"Error in news selection: {str(e)}")
        raise


if __name__ == "__main__":
    main()
