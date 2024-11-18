"""
Data Loader Module

This module provides functionality for loading and filtering articles from
the database. It handles both raw and preprocessed articles, applying
content length constraints and organizing the data for embedding.
"""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from models.db_models import Article, ProcessedArticle


def load_articles_from_db(
    db_session: Session,
    use_processed: bool = False,
    min_words: int = 500,
    max_words: int = 20000
) -> List[Dict]:
    """
    Load and filter articles from the database.

    This function retrieves articles from the database, optionally using
    preprocessed versions, and filters them based on content length constraints.
    Each article is returned as a dictionary containing its metadata and content.

    Parameters
    ----------
    db_session : Session
        SQLAlchemy session for database operations
    use_processed : bool, optional (default=False)
        Whether to use preprocessed articles instead of raw ones
    min_words : int, optional (default=500)
        Minimum word count for an article to be included
    max_words : int, optional (default=20000)
        Maximum word count for an article to be included

    Returns
    -------
    List[Dict]
        List of dictionaries, each containing article data with keys:
        - id: Article identifier
        - title: Article title
        - content: Article content (processed or raw)
        - section: Section name
        - published_at: Publication datetime
        - newspaper: Newspaper name

    Notes
    -----
    The function applies the following filters:
    1. Content length must be between min_words and max_words
    2. All required fields must be present
    3. When use_processed is True, joins with the ProcessedArticle table
    """
    # Construct the appropriate query based on whether we want processed articles
    if use_processed:
        stmt = select(ProcessedArticle, Article).join(Article)
    else:
        stmt = select(Article)

    result = db_session.execute(stmt).fetchall()

    articles = []
    for row in result:
        # Extract the relevant article object
        article = row[0]

        # Determine which content and title to use
        content = article.processed_content if use_processed else article.content
        title = article.processed_title if use_processed else article.title

        # Apply word count filter
        word_count = len(content.split())
        if word_count < min_words or word_count > max_words:
            continue

        # Create article dictionary with all necessary fields
        articles.append({
            'id': article.id,
            'title': title,
            'content': content,
            'section': article.section.name,
            'published_at': article.published_at,
            'newspaper': article.newspaper.name
        })

    return articles
