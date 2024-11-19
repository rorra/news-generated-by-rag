"""
Evaluation Metrics Module

This module provides functionality for evaluating the performance of different
embedding and retrieval strategies for the news article RAG system.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
import time
import numpy as np
from sklearn.metrics import ndcg_score
from storage.qdrant_manager import QdrantManager


@dataclass
class SearchQuery:
    """Data class representing a search query with optional filters."""
    prompt: str
    date: Optional[str] = None
    section: Optional[str] = None
    keywords: Optional[List[str]] = None
    min_keyword_score: float = 0.0


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
    keyword_precision: float
    keyword_recall: float
    keyword_f1: float
    mean_execution_time: float
    queries_per_second: float


class RAGEvaluator:
    """Class for evaluating RAG system performance across different scenarios."""

    def __init__(self, embedder, local: bool = True):
        """
        Initialize the RAG evaluator.

        Parameters
        ----------
        embedder : BaseEmbedder
            The embedder to use for evaluation
        local : bool
            Whether to use local Qdrant instance
        """
        self.embedder = embedder
        self.qdrant = QdrantManager(local=local)
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

        # Prepare filter conditions
        filter_conditions = {}
        if query.date:
            filter_conditions['date'] = query.date
        if query.section:
            filter_conditions['section'] = query.section

        # Execute appropriate search based on query type
        if query.prompt and query.keywords:
            # Combined semantic and keyword search
            query_vector = self.embedder.embed(query.prompt)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                keywords=query.keywords,
                min_keyword_score=query.min_keyword_score,
                limit=limit
            )
        elif query.prompt:
            # Semantic search only
            query_vector = self.embedder.embed(query.prompt)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                limit=limit
            )
        else:
            # Keyword search only
            results = self.qdrant.search_by_keywords(
                collection_name=self.collection_name,
                keywords=query.keywords,
                min_keyword_score=query.min_keyword_score,
                filter_conditions=filter_conditions,
                limit=limit
            )

        execution_time = time.time() - start_time

        return SearchResult(query=query, results=results, execution_time=execution_time)

    def calculate_keyword_metrics(
        self,
        retrieved_keywords: Set[str],
        relevant_keywords: Set[str]
    ) -> Tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score for keyword retrieval.

        Parameters
        ----------
        retrieved_keywords : Set[str]
            Set of keywords retrieved by the search
        relevant_keywords : Set[str]
            Set of relevant keywords for the query

        Returns
        -------
        Tuple[float, float, float]
            Precision, recall, and F1 scores
        """
        if not relevant_keywords:
            return 0.0, 0.0, 0.0

        intersection = len(retrieved_keywords & relevant_keywords)
        precision = intersection / len(retrieved_keywords) if retrieved_keywords else 0
        recall = intersection / len(relevant_keywords)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1

    def evaluate_queries(
        self,
        queries: List[SearchQuery],
        relevant_docs: Dict[str, Dict[str, Any]],
        k: int = 5
    ) -> EvaluationMetrics:
        """
        Evaluate a list of queries and compute metrics.

        Parameters
        ----------
        queries : List[SearchQuery]
            List of queries to evaluate
        relevant_docs : Dict[str, Dict[str, Any]]
            Dictionary mapping query prompts to relevant document info
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
        keyword_precision_scores = []
        keyword_recall_scores = []
        keyword_f1_scores = []

        for query in queries:
            # Execute search
            result = self.execute_search(query, limit=k)
            execution_times.append(result.execution_time)

            # Get relevant info for this query
            relevant_info = relevant_docs.get(query.prompt or '', {})
            relevant_ids = set(relevant_info.get('doc_ids', []))
            relevant_keywords = set(relevant_info.get('keywords', []))

            # Get retrieved info
            retrieved_ids = {str(r['original_id']) for r in result.results}
            retrieved_keywords = {kw for r in result.results for kw, _ in r['keywords']}

            # Calculate document-level metrics
            intersection = len(retrieved_ids & relevant_ids)
            precision = intersection / len(retrieved_ids) if retrieved_ids else 0
            recall = intersection / len(relevant_ids) if relevant_ids else 0

            precision_scores.append(precision)
            recall_scores.append(recall)

            # Calculate NDCG
            relevance_scores = [1 if str(r['original_id']) in relevant_ids else 0 for r in result.results]
            ideal_scores = [1] * len(relevant_ids) + [0] * (k - len(relevant_ids))

            try:
                ndcg = ndcg_score([ideal_scores], [relevance_scores])
            except ValueError:
                ndcg = 0.0

            ndcg_scores.append(ndcg)

            # Calculate keyword metrics
            kw_precision, kw_recall, kw_f1 = self.calculate_keyword_metrics(
                retrieved_keywords,
                relevant_keywords
            )
            keyword_precision_scores.append(kw_precision)
            keyword_recall_scores.append(kw_recall)
            keyword_f1_scores.append(kw_f1)

        # Calculate mean metrics
        mean_execution_time = np.mean(execution_times)

        return EvaluationMetrics(
            precision=np.mean(precision_scores),
            recall=np.mean(recall_scores),
            ndcg=np.mean(ndcg_scores),
            keyword_precision=np.mean(keyword_precision_scores),
            keyword_recall=np.mean(keyword_recall_scores),
            keyword_f1=np.mean(keyword_f1_scores),
            mean_execution_time=mean_execution_time,
            queries_per_second=1.0 / mean_execution_time if mean_execution_time > 0 else 0.0
        )

    def generate_evaluation_report(
        self,
        test_queries: List[SearchQuery],
        relevant_docs: Dict[str, Dict[str, Any]],
        k: int = 5
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Parameters
        ----------
        test_queries : List[SearchQuery]
            List of queries to evaluate
        relevant_docs : Dict[str, Dict[str, Any]]
            Dictionary of relevant documents for each query
        k : int, optional
            Number of results to consider

        Returns
        -------
        Dict[str, Any]
            Evaluation report with metrics and statistics
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
                'keyword_precision': metrics.keyword_precision,
                'keyword_recall': metrics.keyword_recall,
                'keyword_f1': metrics.keyword_f1,
                'mean_execution_time': metrics.mean_execution_time,
                'queries_per_second': metrics.queries_per_second
            },
            'query_categories': {
                'semantic_only': len([q for q in test_queries if q.prompt and not q.keywords]),
                'keyword_only': len([q for q in test_queries if q.keywords and not q.prompt]),
                'combined': len([q for q in test_queries if q.prompt and q.keywords]),
                'with_date': len([q for q in test_queries if q.date]),
                'with_section': len([q for q in test_queries if q.section])
            },
            'timestamp': datetime.now().isoformat()
        }

        return report
