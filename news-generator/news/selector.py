"""
News Selector Module with Qdrant-based similarity.
"""
from typing import List, Dict
from storage.qdrant_manager import QdrantManager


class NewsSelector:
    def __init__(
        self,
        qdrant_manager: QdrantManager,
        similarity_threshold: float = 0.75,
        collection_name: str = "news_minilm"
    ):
        self.qdrant = qdrant_manager
        self.similarity_threshold = similarity_threshold
        self.collection_name = collection_name

    def select_unique_news(self, articles: List[Dict]) -> List[Dict]:
        """Select unique news articles using Qdrant embeddings."""
        if not articles:
            return []

        # Get article IDs
        article_ids = [str(article['id']) for article in articles]

        # Use Qdrant's built-in search to find similar pairs
        unique_articles = []
        seen_ids = set()

        for article in articles:
            article_id = str(article['id'])
            if article_id in seen_ids:
                continue

            # Search for similar articles
            similar = self.qdrant.search_similar(
                collection_name=self.collection_name,
                vector_id=article_id,
                threshold=self.similarity_threshold,
                limit=len(articles)
            )

            # Mark similar articles as seen
            for sim_article in similar:
                seen_ids.add(str(sim_article.id))

            unique_articles.append(article)

        return unique_articles