"""
Qdrant Manager Module

This module provides a high-level interface for interacting with the Qdrant
vector database, supporting both local (Docker) and cloud deployments.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest
)


class QdrantManager:
    """Manager class for Qdrant vector database operations."""

    def __init__(
        self,
        local: bool = True,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize the Qdrant manager."""
        if local:
            self.client = QdrantClient(host=host, port=port)
        else:
            if not url or not api_key:
                raise ValueError(
                    "Cloud configuration requires both 'url' and 'api_key'"
                )
            self.client = QdrantClient(url=url, api_key=api_key)

    @staticmethod
    def _format_date(dt: datetime) -> str:
        """Format datetime object to YYYY-MM-DD string."""
        return dt.strftime('%Y-%m-%d')

    @classmethod
    def from_config(cls, config: Dict[str, Any], local: bool = True) -> 'QdrantManager':
        """Create a QdrantManager instance from configuration dictionary."""
        if local:
            return cls(
                local=True,
                host=config['qdrant']['local']['host'],
                port=config['qdrant']['local']['port']
            )
        else:
            return cls(
                local=False,
                url=config['qdrant']['cloud']['url'],
                api_key=config['qdrant']['cloud']['api_key']
            )

    def create_collection(self, collection_name: str, vector_size: int):
        """Create a new collection in Qdrant."""
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    def insert_articles(self,
                        collection_name: str,
                        articles: List[Dict[str, Any]],
                        embeddings: List[np.ndarray]):
        """Insert articles with their embeddings into Qdrant."""
        points = []
        for idx, (article, embedding) in enumerate(zip(articles, embeddings)):
            published_date = self._format_date(article['published_at'])

            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload={
                    'original_id': article['id'],
                    'title': article['title'],
                    'keywords': article['keywords'],
                    'section': article['section'],
                    'published_at': published_date,
                    'newspaper': article['newspaper']
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    def search(self,
               collection_name: str,
               query_vector: np.ndarray,
               filter_conditions: Optional[Dict] = None,
               limit: int = 5) -> List[Dict]:
        """
        Search for similar articles in Qdrant with optional filtering.

        Parameters
        ----------
        collection_name : str
            Name of the collection to search in
        query_vector : np.ndarray
            Query vector to search with
        filter_conditions : Optional[Dict], optional
            Dictionary containing filter conditions:
            - date: str (YYYY-MM-DD)
            - section: str
        limit : int, optional (default=5)
            Maximum number of results to return
        """
        must_conditions = []
        if filter_conditions:
            if 'date' in filter_conditions:
                must_conditions.append(
                    FieldCondition(
                        key='published_at',
                        match=MatchValue(value=filter_conditions['date'])
                    )
                )

            if 'section' in filter_conditions:
                must_conditions.append(
                    FieldCondition(
                        key='section',
                        match=MatchValue(value=filter_conditions['section'])
                    )
                )

        search_request = SearchRequest(
            vector=query_vector.tolist(),
            limit=limit,
            filter=Filter(must=must_conditions) if must_conditions else None
        )

        results = self.client.search(
            collection_name=collection_name,
            **search_request.dict(exclude_none=True)
        )

        return [
            {
                'id': hit.id,
                'score': hit.score,
                'original_id': hit.payload['original_id'],
                'title': hit.payload['title'],
                'section': hit.payload['section'],
                'keywords': hit.payload['keywords'],
                'published_at': hit.payload['published_at'],
                'newspaper': hit.payload['newspaper']
            }
            for hit in results
        ]
