"""
Data Loader Module

This module provides functionality for loading and filtering articles from
the database. It handles both raw and preprocessed articles, applying
content length constraints and organizing the data for embedding.
"""

from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from models.db_models import Article, ProcessedArticle, Newspaper, Section
import re


def parse_keywords(keywords_str: str) -> List[tuple]:
    """
    Parse keywords string into a list of (keyword, score) tuples.

    Parameters
    ----------
    keywords_str : str
        String of keywords in format "(keyword1,score1),(keyword2,score2),..."

    Returns
    -------
    List[tuple]
        List of tuples containing (keyword, score)
    """
    if not keywords_str:
        return []

    # Regular expression to match (keyword,score) pairs
    pattern = r'\((.*?),([\d.]+)\)'
    matches = re.findall(pattern, keywords_str)

    # Convert matches to list of tuples with proper types
    return [(keyword, float(score)) for keyword, score in matches]


def load_articles_from_db(
    db_session: Session,
    use_processed: bool = False,
    min_words: int = 500,
    max_words: int = 20000,
    min_keyword_score: float = 0.0
) -> List[Dict]:
    """
    Load and filter articles from the database.

    Parameters
    ----------
    db_session : Session
        SQLAlchemy session for database operations
    use_processed : bool, optional
        Whether to use preprocessed articles instead of raw ones
    min_words : int, optional
        Minimum word count for an article to be included
    max_words : int, optional
        Maximum word count for an article to be included
    min_keyword_score : float
        Minimum score for keywords to be included

    Returns
    -------
    List[Dict]
        List of dictionaries, each containing article data with keys:
        - id: Article identifier
        - title: Article title
        - content: Article content (processed or raw)
        - section: Section name
        - keywords: List of (keyword, score) tuples
        - published_at: Publication datetime
        - newspaper: Newspaper name
    """
    if use_processed:
        # Start with ProcessedArticle and explicitly specify the join path
        stmt = (
            select(ProcessedArticle, Article, Section, Newspaper)
            .select_from(ProcessedArticle)
            .join(Article, ProcessedArticle.article_id == Article.id)
            .join(Section, Article.section_id == Section.id)
            .join(Newspaper, Article.newspaper_id == Newspaper.id)
        )
    else:
        # Start with Article and join other tables
        stmt = (
            select(Article, Section, Newspaper)
            .select_from(Article)
            .join(Section, Article.section_id == Section.id)
            .join(Newspaper, Article.newspaper_id == Newspaper.id)
            .outerjoin(ProcessedArticle, Article.id == ProcessedArticle.article_id)
        )

    result = db_session.execute(stmt).fetchall()
    articles = []

    for row in result:
        if use_processed:
            processed_article, article, section, newspaper = row
            content = processed_article.processed_content
            title = processed_article.processed_title
            keywords_str = processed_article.keywords
        else:
            article, section, newspaper = row
            content = article.content
            title = article.title
            # If using raw articles and they have a processed version, get keywords from there
            keywords_str = article.processed_article.keywords if article.processed_article else ""

        # Apply word count filter
        word_count = len(content.split())
        if word_count < min_words or word_count > max_words:
            continue

        # Parse and filter keywords
        keywords = parse_keywords(keywords_str)
        filtered_keywords = [
            (keyword, score) for keyword, score in keywords
            if score >= min_keyword_score
        ]

        # Create article dictionary with all necessary fields
        articles.append({
            'id': article.id,
            'title': title,
            'content': content,
            'section': section.name,
            'keywords': filtered_keywords,
            'published_at': article.published_at,
            'newspaper': newspaper.name
        })

    return articles
