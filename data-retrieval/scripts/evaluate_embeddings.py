"""
Embedding Evaluation Script

This script evaluates different embedding strategies using various types
of queries and generates comparative metrics.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from dotenv import load_dotenv
from config import SessionLocal
from storage.data_loader import load_articles_from_db
from evaluation.metrics import RAGEvaluator, SearchQuery
from utils.logging_config import setup_logging
from utils.common import (
    load_config,
    get_qdrant_client,
    get_embedder
)

# Set up logger
logger = setup_logging("evaluator")


def load_test_queries() -> List[SearchQuery]:
    """
    Load test queries from the test set file.

    Returns
    -------
    List[SearchQuery]
        List of search queries for evaluation

    Raises
    ------
    FileNotFoundError
        If test queries file is not found
    """
    test_queries_path = Path(__file__).parent.parent / "evaluation" / "test_sets" / "test_queries.json"

    if not test_queries_path.exists():
        raise FileNotFoundError(f"Test queries file not found at {test_queries_path}")

    try:
        with open(test_queries_path) as f:
            queries_data = json.load(f)

        return [
            SearchQuery(
                prompt=q['prompt'],
                date=q['date'],
                section=q['section']
            )
            for q in queries_data
        ]
    except Exception as e:
        logger.error(f"Error loading test queries: {str(e)}")
        raise


def get_relevant_documents(queries: List[SearchQuery], articles: List[Dict]) -> Dict[str, List[str]]:
    """
    Get relevant documents for each query.

    Parameters
    ----------
    queries : List[SearchQuery]
        List of queries to evaluate
    articles : List[Dict]
        List of all available articles

    Returns
    -------
    Dict[str, List[str]]
        Mapping of query prompts to lists of relevant document IDs
    """
    relevant_docs = {}

    for query in queries:
        # Simple relevance calculation based on text similarity
        # In a real scenario, this should be replaced with human annotations
        relevant = []
        search_terms = query.prompt.lower().split()

        for article in articles:
            content = article['content'].lower()
            title = article['title'].lower()

            # Check if search terms appear in title or content
            if any(term in title or term in content for term in search_terms):
                relevant.append(str(article['id']))

            # Apply section filter if present
            if query.section and article['section'] != query.section:
                continue

            # Apply date filter if present
            if query.date:
                article_date = article['published_at'].strftime('%Y-%m-%d')
                if article_date != query.date:
                    continue

        relevant_docs[query.prompt] = relevant

    return relevant_docs


def save_results(results: List[Dict], output_dir: Path):
    """
    Save evaluation results and generate comparative analysis.

    Parameters
    ----------
    results : List[Dict]
        List of evaluation results per embedder
    output_dir : Path
        Directory to save results
    """
    output_dir.mkdir(exist_ok=True)

    # Save individual reports
    for report in results:
        embedder_type = report['embedder_type']
        report_path = output_dir / f"{embedder_type}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

    # Generate comparative analysis
    df = pd.DataFrame([
        {
            'Embedder': r['embedder_type'],
            'Precision@5': r['metrics']['precision_at_k'],
            'Recall@5': r['metrics']['recall_at_k'],
            'NDCG': r['metrics']['ndcg'],
            'Queries/Second': r['metrics']['queries_per_second']
        }
        for r in results
    ])

    # Save comparison
    comparison_path = output_dir / "comparison.csv"
    df.to_csv(comparison_path, index=False)

    # Print summary
    logger.info("\nEvaluation Results Summary:")
    print(df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Evaluate embedding strategies')
    parser.add_argument('--local', action='store_true',
                        help='Use local Qdrant instance')
    parser.add_argument('--output', type=str, default='evaluation_results',
                        help='Output directory for results')
    parser.add_argument('--embedders', nargs='+',
                        default=['tfidf', 'bm25', 'dpr', 'sbert', 'minilm'],
                        help='Embedder types to evaluate')

    args = parser.parse_args()

    try:
        # Load environment variables and configuration
        load_dotenv()
        config = load_config()

        # Initialize database connection
        logger.info("Connecting to database...")
        session = SessionLocal()

        try:
            # Load articles
            logger.info("Loading articles from database...")
            articles = load_articles_from_db(
                session,
                min_words=config['processing']['min_words'],
                max_words=config['processing']['max_words']
            )

            if not articles:
                logger.error("No articles found in database!")
                return

            logger.info(f"Loaded {len(articles)} articles")

            # Load test queries
            logger.info("Loading test queries...")
            test_queries = load_test_queries()

            if not test_queries:
                logger.error("No test queries found!")
                return

            logger.info(f"Loaded {len(test_queries)} test queries")

            # Get relevant documents for evaluation
            logger.info("Getting relevant documents...")
            relevant_docs = get_relevant_documents(test_queries, articles)

            # Initialize Qdrant client
            logger.info("Connecting to Qdrant...")
            qdrant = get_qdrant_client(local=args.local)

            # Evaluate each embedding strategy
            results = []
            for embedder_type in args.embedders:
                logger.info(f"\nEvaluating {embedder_type}...")

                # Initialize and fit embedder if necessary
                embedder = get_embedder(embedder_type, config, articles)

                # Create evaluator
                evaluator = RAGEvaluator(embedder, qdrant)

                # Generate evaluation report
                report = evaluator.generate_evaluation_report(
                    test_queries,
                    relevant_docs
                )

                results.append(report)

            # Save results
            logger.info("\nSaving results...")
            save_results(results, Path(args.output))

        finally:
            session.close()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
