"""
Evaluation Metrics Module

This module provides functionality for evaluating the performance of different
embedding and retrieval strategies for the news article RAG system.
"""

from typing import List, Dict, Any, Optional
import time
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.metrics import precision_score, recall_score, ndcg_score
from qdrant_client.models import Filter, FieldCondition, MatchValue


@dataclass
class SearchQuery:
    """Data class representing a search query with optional filters."""
    prompt: str
    date: Optional[str] = None
    section: Optional[str] = None


@dataclass
class SearchResult:
    """Data class representing search results and metrics."""
    query: SearchQuery
    results: List[Dict]
    execution_time: float


@dataclass
class EvaluationMetrics:
    """Data class containing evaluation metrics for a search strategy."""
    precision: float
    recall: float
    ndcg: float
    mean_execution_time: float
    queries_per_second: float


class RAGEvaluator:
    """
    Class for evaluating RAG system performance across different scenarios.

    This evaluator measures retrieval quality and performance metrics for
    different search strategies and embedding methods.
    """

    def __init__(self, embedder, qdrant):
        """Initialize the RAG evaluator."""
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_name = embedder.collection_name

    def execute_search(self, query: SearchQuery, limit: int = 5) -> SearchResult:
        """
        Execute a search query and measure execution time.

        Parameters
        ----------
        query : SearchQuery
            The search query to execute
        limit : int, optional
            Maximum number of results to retrieve

        Returns
        -------
        SearchResult
            Search results and execution metrics
        """
        start_time = time.time()

        # Embed query
        query_vector = self.embedder.embed(query.prompt)

        # Prepare filter conditions
        must_conditions = []
        if query.date:
            must_conditions.append(
                FieldCondition(
                    key='published_at',
                    match=MatchValue(value=query.date)
                )
            )
        if query.section:
            must_conditions.append(
                FieldCondition(
                    key='section',
                    match=MatchValue(value=query.section)
                )
            )

        # Execute search
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit
        )

        execution_time = time.time() - start_time

        # Format results
        formatted_results = [
            {
                'id': hit.id,
                'score': hit.score,
                'original_id': hit.payload.get('original_id'),
                'title': hit.payload.get('title'),
                'section': hit.payload.get('section'),
                'published_at': hit.payload.get('published_at'),
                'newspaper': hit.payload.get('newspaper')
            }
            for hit in results
        ]

        return SearchResult(
            query=query,
            results=formatted_results,
            execution_time=execution_time
        )

    def evaluate_queries(
        self,
        queries: List[SearchQuery],
        relevant_docs: Dict[str, List[str]],
        k: int = 5
    ) -> EvaluationMetrics:
        """
        Evaluate a list of queries and compute metrics.

        Parameters
        ----------
        queries : List[SearchQuery]
            List of queries to evaluate
        relevant_docs : Dict[str, List[str]]
            Dictionary mapping query prompts to lists of relevant document IDs
        k : int, optional
            Number of results to consider for metrics

        Returns
        -------
        EvaluationMetrics
            Computed evaluation metrics
        """
        execution_times = []
        precision_scores = []
        recall_scores = []
        ndcg_scores = []

        for query in queries:
            # Execute search
            result = self.execute_search(query, limit=k)
            execution_times.append(result.execution_time)

            # Get retrieved doc IDs
            retrieved_ids = [str(r.get('original_id')) for r in result.results]

            # Get relevant docs for this query
            relevant = relevant_docs.get(query.prompt, [])

            # Calculate precision and recall
            relevant_set = set(relevant)
            retrieved_set = set(retrieved_ids)

            intersection = len(relevant_set & retrieved_set)

            precision = intersection / len(retrieved_ids) if retrieved_ids else 0
            recall = intersection / len(relevant) if relevant else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Calculate NDCG
            relevance_scores = [1 if doc_id in relevant_set else 0 for doc_id in retrieved_ids]
            ideal_scores = [1] * len(relevant_set) + [0] * (k - len(relevant_set))

            try:
                ndcg = ndcg_score([ideal_scores], [relevance_scores])
            except Exception:
                ndcg = 0.0  # Handle cases where ndcg_score fails (e.g., all zeros)

            ndcg_scores.append(ndcg)

        # Calculate mean metrics
        mean_execution_time = np.mean(execution_times)

        return EvaluationMetrics(
            precision=np.mean(precision_scores),
            recall=np.mean(recall_scores),
            ndcg=np.mean(ndcg_scores),
            mean_execution_time=mean_execution_time,
            queries_per_second=1.0 / mean_execution_time if mean_execution_time > 0 else 0.0
        )

    def generate_evaluation_report(
        self,
        test_queries: List[SearchQuery],
        relevant_docs: Dict[str, List[str]],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        test_queries : List[SearchQuery]
            Test queries to evaluate
        relevant_docs : Dict[str, List[str]]
            Dictionary of relevant documents for each query
        k : int, optional
            Number of results to consider

        Returns
        -------
        Dict[str, Any]
            Evaluation report with metrics and additional information
        """
        metrics = self.evaluate_queries(test_queries, relevant_docs, k)

        report = {
            'embedder_type': self.embedder.__class__.__name__.replace('Embedder', '').lower(),
            'collection_name': self.collection_name,
            'number_of_queries': len(test_queries),
            'k': k,
            'metrics': {
                'precision_at_k': metrics.precision,
                'recall_at_k': metrics.recall,
                'ndcg': metrics.ndcg,
                'mean_execution_time': metrics.mean_execution_time,
                'queries_per_second': metrics.queries_per_second
            },
            'query_categories': {
                'date_and_section': len([q for q in test_queries if q.date and q.section]),
                'date_only': len([q for q in test_queries if q.date and not q.section]),
                'section_only': len([q for q in test_queries if q.section and not q.date]),
                'no_filters': len([q for q in test_queries if not q.date and not q.section])
            },
            'timestamp': datetime.now().isoformat()
        }

        return report