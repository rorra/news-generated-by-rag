import argparse
from datetime import datetime
from dotenv import load_dotenv
from utils.common import load_config, get_qdrant_client, get_embedder
from config import SessionLocal
from storage.data_loader import load_articles_from_db
from qdrant_client.models import Filter, FieldCondition, MatchValue


def search_news(prompt: str, date: str = None, section: str = None, limit: int = 20,
                embedder_type: str = "minilm", local: bool = True):
    """Search for news articles using Qdrant."""
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

        # Initialize clients
        qdrant = get_qdrant_client(local=local)
        embedder = get_embedder(embedder_type, config, articles)

        # Embed query
        query_vector = embedder.embed(prompt)

        # Prepare filters
        must_conditions = []
        if date:
            must_conditions.append(
                FieldCondition(key='published_at', match=MatchValue(value=date))
            )
        if section:
            must_conditions.append(
                FieldCondition(key='section', match=MatchValue(value=section))
            )

        # Execute search
        results = qdrant.search(
            collection_name=embedder.collection_name,
            query_vector=query_vector,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=limit
        )

        # Print results
        print(f"\nFound {len(results)} results:\n")
        for i, hit in enumerate(results, 1):
            print(f"{i}. [{hit.payload['section']} - {hit.payload['published_at']}] {hit.payload['title']}")
            print(f"   Score: {hit.score:.4f}")
            print(f"   Newspaper: {hit.payload['newspaper']}\n")

    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser(description='Search news articles')
    parser.add_argument('prompt', help='Search query')
    parser.add_argument('--date', help='Date filter (YYYY-MM-DD)')
    parser.add_argument('--section', help='Section filter')
    parser.add_argument('--limit', type=int, default=20, help='Number of results')
    parser.add_argument('--embedder', default='minilm',
                        choices=['tfidf', 'bm25', 'dpr', 'sbert', 'minilm'],
                        help='Embedder type')
    parser.add_argument('--local', action='store_true', help='Use local Qdrant')

    args = parser.parse_args()
    search_news(
        prompt=args.prompt,
        date=args.date,
        section=args.section,
        limit=args.limit,
        embedder_type=args.embedder,
        local=args.local
    )


if __name__ == "__main__":
    main()
