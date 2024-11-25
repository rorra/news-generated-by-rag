"""
News Fetcher Module

Handles RAG retrieval using existing QdrantManager.
"""

from typing import List, Dict, Optional
from storage.qdrant_manager import QdrantManager


class NewsFetcher:
    def __init__(self, local: bool = True, embedder: str = "minilm"):
        """Initialize the news fetcher."""
        self.collection_name = f"news_{embedder}"
        self.qdrant = QdrantManager(local=local)

    def fetch_news(
        self,
        date: str,
        section: str,
        limit: int = 100
    ) -> List[Dict]:
        """
        Fetch news for a specific date and section.

        Args:
            date: Date in YYYY-MM-DD format
            section: News section name
            limit: Maximum number of articles to fetch

        Returns:
            List of news articles with their metadata
        """
        filter_conditions = {
            'date': date,
            'section': section
        }

        return self.qdrant.search_by_keywords(
            collection_name=self.collection_name,
            keywords=[],  # Empty list to fetch all news
            filter_conditions=filter_conditions,
            limit=limit
        )
