"""
Qdrant Manager Module

This module provides a high-level interface for interacting with the Qdrant
vector database, supporting both local (Docker) and cloud deployments.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    SearchParams,
    MatchAny
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

    def _build_filter_conditions(
        self,
        filter_conditions: Optional[Dict] = None,
        keywords: Optional[List[str]] = None,
        min_keyword_score: float = 0.0,
        match_any_keyword: bool = True
    ) -> Optional[Filter]:
        """Build Qdrant filter conditions."""
        must_conditions = []

        # Add date and section filters
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

        # Add keyword filters
        if keywords:
            if match_any_keyword:
                # Match any of the keywords using MatchAny
                must_conditions.append(
                    FieldCondition(
                        key='keywords',
                        match=MatchAny(any=keywords)
                    )
                )
                if min_keyword_score > 0:
                    must_conditions.append(
                        FieldCondition(
                            key='keyword_scores',
                            range=Range(gte=min_keyword_score)
                        )
                    )
            else:
                # Match all keywords
                for keyword in keywords:
                    must_conditions.append(
                        FieldCondition(
                            key='keywords',
                            match=MatchValue(value=keyword)
                        )
                    )
                    if min_keyword_score > 0:
                        must_conditions.append(
                            FieldCondition(
                                key='keyword_scores',
                                range=Range(gte=min_keyword_score)
                            )
                        )

        return Filter(must=must_conditions) if must_conditions else None

    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        filter_conditions: Optional[Dict] = None,
        keywords: Optional[List[str]] = None,
        min_keyword_score: float = 0.0,
        match_any_keyword: bool = True,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for similar articles in Qdrant with optional filtering.

        Parameters
        ----------
        collection_name : str
            Name of the collection to search in
        query_vector : np.ndarray
            Query vector to search with
        filter_conditions : Optional[Dict]
            Dictionary containing filter conditions (date, section)
        keywords : Optional[List[str]]
            List of keywords to filter by
        min_keyword_score : float
            Minimum score for keyword matches
        match_any_keyword : bool
            If True, matches articles with any of the keywords
        limit : int
            Maximum number of results to return
        """
        search_filter = self._build_filter_conditions(
            filter_conditions,
            keywords,
            min_keyword_score,
            match_any_keyword
        )

        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            query_filter=search_filter,
            search_params=SearchParams(hnsw_ef=128),
            limit=limit
        )

        return [
            {
                'id': hit.id,
                'score': hit.score,
                'original_id': hit.payload['original_id'],
                'title': hit.payload['title'],
                'section': hit.payload['section'],
                'keywords': list(zip(
                    hit.payload['keywords'],
                    hit.payload['keyword_scores']
                )),
                'published_at': hit.payload['published_at'],
                'newspaper': hit.payload['newspaper']
            }
            for hit in results
        ]

    def search_by_keywords(
        self,
        collection_name: str,
        keywords: List[str],
        min_keyword_score: float = 0.0,
        filter_conditions: Optional[Dict] = None,
        match_any_keyword: bool = True,
        limit: int = 5
    ) -> List[Dict]:
        """
        Search for articles by keywords without requiring a query vector.

        Parameters
        ----------
        collection_name : str
            Name of the collection to search in
        keywords : List[str]
            List of keywords to search for
        min_keyword_score : float
            Minimum score for keyword matches
        filter_conditions : Optional[Dict]
            Dictionary containing filter conditions
        match_any_keyword : bool
            If True, matches articles with any of the keywords
        limit : int
            Maximum number of results to return
        """
        search_filter = self._build_filter_conditions(
            filter_conditions,
            keywords,
            min_keyword_score,
            match_any_keyword
        )

        results = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,  # Changed from filter to scroll_filter
            limit=limit
        )[0]  # scroll returns (points, next_page_offset)

        return [
            {
                'id': point.id,
                'original_id': point.payload['original_id'],
                'title': point.payload['title'],
                'section': point.payload['section'],
                'keywords': list(zip(
                    point.payload['keywords'],
                    point.payload['keyword_scores']
                )),
                'published_at': point.payload['published_at'],
                'newspaper': point.payload['newspaper']
            }
            for point in results
        ]
