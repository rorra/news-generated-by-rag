"""
Search News Script

This script provides a command-line interface for searching news articles
using semantic search, keyword-based search, or a combination of both.
"""

import argparse
from typing import List, Optional, Dict
from datetime import datetime
from dotenv import load_dotenv
from utils.common import load_config
from storage.qdrant_manager import QdrantManager
from config import SessionLocal
from storage.data_loader import load_articles_from_db
import json


def format_keywords(keywords: List[tuple]) -> str:
    """Format keyword-score pairs for display."""
    return ', '.join([f"{kw} ({score:.3f})" for kw, score in keywords])


def display_results(results: List[Dict], search_type: str, show_scores: bool = True, json_output: bool = False):
    """Display search results in a formatted manner or JSON."""
    if True: # json_output:
        # Prepare results for JSON output
        json_results = [
            {
                "id": hit.get("id"),
                "section": hit.get("section"),
                "published_at": hit.get("published_at"),
                "title": hit.get("title"),
                "score": hit.get("score") if show_scores and 'score' in hit else None,
                "keywords": hit.get("keywords"),
                "newspaper": hit.get("newspaper")
            }
            for hit in results
        ]
        print(json.dumps({"search_type": search_type, "results": json_results}, indent=4))
    else:
        # Standard formatted output
        print(f"\nSearch type: {search_type}")
        print(f"Found {len(results)} results:\n")

        for i, hit in enumerate(results, 1):
            print(f"{i}. [{hit['section']} - {hit['published_at']}] {hit['title']}")
            if show_scores and 'score' in hit:
                print(f"   Similarity Score: {hit['score']:.4f}")
            if hit['keywords']:
                print(f"   Keywords: {format_keywords(hit['keywords'])}")
            print(f"   Newspaper: {hit['newspaper']}\n")


def search_news(
    prompt: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    date: Optional[str] = None,
    section: Optional[str] = None,
    min_keyword_score: float = 0.0,
    match_any_keyword: bool = True,
    limit: int = 20,
    embedder_type: str = "minilm",
    local: bool = True,
    sort_by_keyword_score: bool = False,
    json_output = False
):
    """Search for news articles using various criteria."""
    # Load environment variables and configuration
    load_dotenv()
    config = load_config()

    # Initialize database session and load articles
    session = SessionLocal()
    try:
        articles = load_articles_from_db(
            session,
            min_words=config['processing']['min_words'],
            max_words=config['processing']['max_words']
        ) if embedder_type in ['tfidf', 'bm25'] else None

        # Initialize embedder
        from utils.common import get_embedder
        embedder = get_embedder(embedder_type, config, articles)

        # Initialize QdrantManager
        if local:
            qdrant = QdrantManager(
                local=True,
                host=config['qdrant']['local']['host'],
                port=config['qdrant']['local']['port']
            )
        else:
            qdrant = QdrantManager(
                local=False,
                url=config['qdrant']['cloud']['url'],
                api_key=config['qdrant']['cloud']['api_key']
            )

        # Prepare filter conditions
        filter_conditions = {}
        if date:
            filter_conditions['date'] = date
        if section:
            filter_conditions['section'] = section

        # Execute search based on search type
        if prompt and keywords:
            # Combined semantic and keyword search
            query_vector = embedder.embed(prompt)
            results = qdrant.search(
                collection_name=embedder.collection_name,
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                keywords=keywords,
                min_keyword_score=min_keyword_score,
                match_any_keyword=match_any_keyword,
                limit=limit
            )
            search_type = "Semantic + Keyword"

        elif prompt:
            # Semantic search only
            query_vector = embedder.embed(prompt)
            results = qdrant.search(
                collection_name=embedder.collection_name,
                query_vector=query_vector,
                filter_conditions=filter_conditions,
                limit=limit
            )
            search_type = "Semantic"

        elif keywords:
            # Keyword search only
            results = qdrant.search_by_keywords(
                collection_name=embedder.collection_name,
                keywords=keywords,
                min_keyword_score=min_keyword_score,
                filter_conditions=filter_conditions,
                match_any_keyword=match_any_keyword,
                limit=limit
            )
            search_type = "Keyword-only"

        else:
            raise ValueError("Either prompt or keywords must be provided")

        # Sort results by keyword score if requested
        if sort_by_keyword_score and keywords:
            results.sort(
                key=lambda x: max(
                    score for kw, score in x['keywords']
                    if kw in keywords
                ) if any(kw in keywords for kw, _ in x['keywords']) else -1,
                reverse=True
            )

        # Display results
        display_results(results, search_type, json_output)

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Search news articles')

    # Search criteria
    parser.add_argument('--prompt', help='Search query text')
    parser.add_argument('--keywords', nargs='+', help='Keywords to search for')

    # Filters
    parser.add_argument('--date', help='Date filter (YYYY-MM-DD)')
    parser.add_argument('--section', help='Section filter')

    # Keyword options
    parser.add_argument('--min-keyword-score', type=float, default=0.0,
                        help='Minimum keyword score threshold')
    parser.add_argument('--match-any-keyword', action='store_true', default=True,
                        help='Match any keyword instead of all keywords')
    parser.add_argument('--sort-by-keyword-score', action='store_true',
                        help='Sort results by keyword score')

    # Search parameters
    parser.add_argument('--limit', type=int, default=20,
                        help='Number of results')
    parser.add_argument('--embedder', default='minilm',
                        choices=['tfidf', 'bm25', 'dpr', 'sbert', 'minilm'],
                        help='Embedder type')
    parser.add_argument('--local', action='store_true',
                        help='Use local Qdrant')

    # Output format
    parser.add_argument('--json', action='store_true', help='Output results in JSON format')

    args = parser.parse_args()

    if not args.prompt and not args.keywords:
        parser.error("Either --prompt or --keywords must be specified")

    search_news(
        prompt=args.prompt,
        keywords=args.keywords,
        date=args.date,
        section=args.section,
        min_keyword_score=args.min_keyword_score,
        match_any_keyword=True,
        limit=args.limit,
        embedder_type=args.embedder,
        local=args.local,
        sort_by_keyword_score=args.sort_by_keyword_score,
        json_output=args.json
    )


if __name__ == "__main__":
    main()
