"""
Article Indexing Script

This script indexes articles using various embedding methods and stores them in Qdrant.
"""

import argparse
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)
from config import SessionLocal
from storage.data_loader import load_articles_from_db
from utils.logging_config import setup_logging
from utils.common import load_config, get_qdrant_client, get_embedder

# Set up logger
logger = setup_logging("indexer")


def process_articles(articles: List[Dict], embedder, batch_size: int = 32) -> List[np.ndarray]:
    """
    Process articles in batches and return embeddings.

    Parameters
    ----------
    articles : List[Dict]
        List of articles to process
    embedder : BaseEmbedder
        Embedder instance to use
    batch_size : int, optional
        Size of batches for processing

    Returns
    -------
    List[np.ndarray]
        List of embedding vectors
    """
    embeddings = []

    # Process in batches
    total_batches = (len(articles) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(articles), batch_size), total=total_batches, desc="Processing articles"):
        batch = articles[i:i + batch_size]
        try:
            batch_embeddings = [embedder.embed(article['content']) for article in batch]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error processing batch {i // batch_size}: {str(e)}")
            raise

    return embeddings


def main():
    parser = argparse.ArgumentParser(description='Index articles in Qdrant with different embedding methods')
    parser.add_argument('--embedder-type', required=True,
                        choices=['tfidf', 'bm25', 'dpr', 'sbert', 'minilm'],
                        help='Type of embedder to use')
    parser.add_argument('--local', action='store_true',
                        help='Use local Qdrant instance')
    parser.add_argument('--use-processed', action='store_true',
                        help='Use processed articles instead of raw articles')
    parser.add_argument('--batch-size', type=int,
                        help='Batch size for processing')
    parser.add_argument('--min-keyword-score', type=float, default=0.0,
                        help='Minimum score threshold for including keywords')

    args = parser.parse_args()

    try:
        # Load environment variables and configuration
        load_dotenv()
        config = load_config()

        # Initialize database connection
        logger.info("Connecting to database...")
        session = SessionLocal()

        # Initialize Qdrant client
        logger.info("Connecting to Qdrant...")
        qdrant = get_qdrant_client(local=args.local)

        # Get batch size from args or config
        batch_size = args.batch_size or config['processing']['batch_size']

        try:
            # Load articles with keyword score filtering
            logger.info("Loading articles from database...")
            articles = load_articles_from_db(
                session,
                use_processed=args.use_processed,
                min_words=config['processing']['min_words'],
                max_words=config['processing']['max_words'],
                min_keyword_score=args.min_keyword_score
            )
            logger.info(f"Loaded {len(articles)} articles")

            if not articles:
                logger.warning("No articles found matching the criteria!")
                return

            # Get embedder
            embedder = get_embedder(args.embedder_type, config, articles)

            # Create collection
            logger.info(f"Creating collection {embedder.collection_name}...")
            qdrant.recreate_collection(
                collection_name=embedder.collection_name,
                vectors_config=VectorParams(
                    size=embedder.dimension,
                    distance=Distance.COSINE
                )
            )

            # Process articles and get embeddings
            logger.info("Processing articles...")
            embeddings = process_articles(articles, embedder, batch_size)

            # Store in Qdrant
            logger.info("Storing articles in Qdrant...")
            points = []
            for idx, (article, embedding) in enumerate(zip(articles, embeddings)):
                published_date = article['published_at'].strftime('%Y-%m-%d') if article['published_at'] else None

                # Split keywords and scores for storage
                keywords = [kw for kw, _ in article['keywords']]
                keyword_scores = [score for _, score in article['keywords']]

                point = PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload={
                        'original_id': article['id'],
                        'title': article['title'],
                        'section': article['section'],
                        'keywords': keywords,
                        'keyword_scores': keyword_scores,
                        'published_at': published_date,
                        'newspaper': article['newspaper']
                    }
                )
                points.append(point)

            # Insert points in batches to avoid memory issues with large datasets
            upload_batch_size = 100  # Adjust based on your system's memory
            for i in tqdm(range(0, len(points), upload_batch_size), desc="Uploading to Qdrant"):
                batch = points[i:i + upload_batch_size]
                qdrant.upsert(
                    collection_name=embedder.collection_name,
                    points=batch
                )

            logger.info(f"Successfully indexed {len(articles)} articles with {args.embedder_type}")

            # Log keyword statistics
            total_keywords = sum(len(article['keywords']) for article in articles)
            avg_keywords = total_keywords / len(articles) if articles else 0
            logger.info(f"Average keywords per article: {avg_keywords:.2f}")

            # Log score distribution
            all_scores = [score for article in articles for _, score in article['keywords']]
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                avg_score = sum(all_scores) / len(all_scores)
                logger.info(f"Keyword score distribution:")
                logger.info(f"  Min: {min_score:.4f}")
                logger.info(f"  Max: {max_score:.4f}")
                logger.info(f"  Avg: {avg_score:.4f}")

        finally:
            session.close()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
