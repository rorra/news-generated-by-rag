"""
Embedding Evaluation Script

This script evaluates different embedding strategies using various types
of queries and generates comparative metrics including keyword-based evaluation.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd
from dotenv import load_dotenv
from config import SessionLocal
from storage.data_loader import load_articles_from_db
from evaluation.metrics import RAGEvaluator, SearchQuery
from utils.logging_config import setup_logging
from utils.common import (
    load_config,
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
    """
    test_queries_path = Path(__file__).parent.parent / "evaluation" / "test_sets" / "test_queries.json"

    if not test_queries_path.exists():
        raise FileNotFoundError(f"Test queries file not found at {test_queries_path}")

    try:
        with open(test_queries_path) as f:
            queries_data = json.load(f)

        return [
            SearchQuery(
                prompt=q.get('prompt'),
                date=q.get('date'),
                section=q.get('section'),
                keywords=q.get('keywords', []),
                min_keyword_score=q.get('min_keyword_score', 0.0)
            )
            for q in queries_data
        ]
    except Exception as e:
        logger.error(f"Error loading test queries: {str(e)}")
        raise


def get_relevant_documents(
    queries: List[SearchQuery],
    articles: List[Dict],
    keyword_match_threshold: float = 0.5
) -> Dict[str, Dict]:
    """
    Get relevant documents and keywords for each query.

    Parameters
    ----------
    queries : List[SearchQuery]
        List of queries to evaluate
    articles : List[Dict]
        List of all available articles
    keyword_match_threshold : float
        Threshold for considering a keyword match relevant

    Returns
    -------
    Dict[str, Dict]
        Mapping of query prompts to dictionaries containing:
        - doc_ids: List of relevant document IDs
        - keywords: List of relevant keywords
    """
    relevant_docs = {}

    for query in queries:
        relevant = []
        all_keywords = set()

        # Get search terms from prompt
        search_terms = query.prompt.lower().split() if query.prompt else []
        query_keywords = set(query.keywords) if query.keywords else set()

        for article in articles:
            content = article['content'].lower()
            title = article['title'].lower()
            is_relevant = False

            # Check content/title relevance
            if search_terms and any(term in title or term in content for term in search_terms):
                is_relevant = True

            # Check keyword relevance
            article_keywords = {kw for kw, score in article['keywords'] if score >= keyword_match_threshold}
            if query_keywords and article_keywords & query_keywords:
                is_relevant = True

            # Apply filters
            if query.section and article['section'] != query.section:
                is_relevant = False

            if query.date:
                article_date = article['published_at'].strftime('%Y-%m-%d')
                if article_date != query.date:
                    is_relevant = False

            if is_relevant:
                relevant.append(str(article['id']))
                all_keywords.update(article_keywords)

        relevant_docs[query.prompt or ''] = {
            'doc_ids': relevant,
            'keywords': list(all_keywords)
        }

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
            'Keyword Precision': r['metrics']['keyword_precision'],
            'Keyword Recall': r['metrics']['keyword_recall'],
            'Keyword F1': r['metrics']['keyword_f1'],
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
    parser.add_argument('--keyword-match-threshold', type=float, default=0.5,
                        help='Threshold for keyword relevance matching')
    parser.add_argument('--min-keyword-score', type=float, default=0.0,
                        help='Minimum score for including keywords in evaluation')

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
                max_words=config['processing']['max_words'],
                min_keyword_score=args.min_keyword_score
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

            # Log query distribution
            keyword_queries = len([q for q in test_queries if q.keywords])
            semantic_queries = len([q for q in test_queries if q.prompt])
            combined_queries = len([q for q in test_queries if q.prompt and q.keywords])

            logger.info("\nQuery Distribution:")
            logger.info(f"Semantic-only queries: {semantic_queries - combined_queries}")
            logger.info(f"Keyword-only queries: {keyword_queries - combined_queries}")
            logger.info(f"Combined queries: {combined_queries}")

            # Get relevant documents for evaluation
            logger.info("Getting relevant documents...")
            relevant_docs = get_relevant_documents(
                test_queries,
                articles,
                keyword_match_threshold=args.keyword_match_threshold
            )

            # Evaluate each embedding strategy
            results = []
            for embedder_type in args.embedders:
                logger.info(f"\nEvaluating {embedder_type}...")

                # Initialize and fit embedder if necessary
                embedder = get_embedder(embedder_type, config, articles)

                # Create evaluator
                evaluator = RAGEvaluator(embedder, local=args.local)

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
