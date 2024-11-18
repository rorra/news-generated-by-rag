"""
Test Set Generator

This script provides utilities for generating test queries and identifying
relevant documents for RAG system evaluation, with section-specific topics.
"""

from typing import List, Dict, Set
from datetime import datetime, timedelta
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session
import random

from evaluation.metrics import SearchQuery
from models.db_models import Article, ProcessedArticle

# Define section-specific topics
SECTION_TOPICS = {
    "Economía": [
        "apreciación del peso",
        "pesquera china",
        "pymes caputo",
        "palos verdes",
        "exportaciones industriales",
        "transferencia de emisiones",
        "créditos hipotecarios",
        "dólar MEP",
        "poder adquisitivo",
        "impuesto pais"
    ],
    "Internacional": [
        "guerra en Ucrania",
        "estados unidos y rusia",
        "francia y mercosur",
        "seguridad social en estados unidos",
        "reloj de oro del titanic",
        "cambio climatico",
        "donald trumpt",
        "g20 brasil",
        "homicidios en uruguay",
        "OTAN"
    ],
    "Política": [
        "Javier Milei",
        "Emanuel Macron",
        "Cristina Kirchner",
        "kirchnerismo",
        "reforma fiscal",
        "debate presidencial",
        "eliminación de las PASO",
        "RIGI",
        "Estados Unidos",
        "congreso nacional"
    ],
    "Sociedad": [
        "muerta al nacer",
        "oscurantismo",
        "Andrea Giunta",
        "clima",
        "inteligencia artificial",
        "hormigas voladoras",
        "propina digital",
        "Efemérides",
        "Lionsgate",
        "las mejores películas"
    ]
}

# Define cross-section topics (topics that could appear in multiple sections)
CROSS_SECTION_TOPICS = [
    "crisis económica",
    "presupuesto nacional",
    "políticas públicas",
    "impacto social",
    "desarrollo económico",
    "medidas gubernamentales",
    "sector privado",
    "reforma estatal"
]


def get_topic_variations(base_topic: str) -> List[str]:
    """
    Generate variations of a topic for more natural queries.

    Parameters
    ----------
    base_topic : str
        The base topic to generate variations for

    Returns
    -------
    List[str]
        List of topic variations
    """
    variations = [
        f"noticias sobre {base_topic}",
        f"información de {base_topic}",
        f"últimas noticias de {base_topic}",
        f"actualidad sobre {base_topic}",
        base_topic
    ]
    return variations


def generate_test_queries(
    dates: List[str] = None,
    sections: List[str] = None,
    include_cross_section: bool = True,
    queries_per_section: int = 5
) -> List[SearchQuery]:
    """
    Generate a comprehensive set of test queries with section-specific topics.

    Parameters
    ----------
    dates : List[str], optional
        List of dates in YYYY-MM-DD format
    sections : List[str], optional
        List of section names
    include_cross_section : bool, optional
        Whether to include cross-section topics
    queries_per_section : int, optional
        Number of queries to generate per section

    Returns
    -------
    List[SearchQuery]
        Generated test queries
    """
    if not dates:
        today = datetime.now()
        dates = [
            (today - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(7)
        ]

    if not sections:
        sections = ["Economía", "Internacional", "Política", "Sociedad"]

    queries = []

    # Generate section-specific queries
    for section in sections:
        section_topics = SECTION_TOPICS[section]
        selected_topics = random.sample(section_topics, min(queries_per_section, len(section_topics)))

        for topic in selected_topics:
            # Add query with both date and section
            if dates:
                date = random.choice(dates)
                queries.append(SearchQuery(
                    prompt=random.choice(get_topic_variations(topic)),
                    date=date,
                    section=section
                ))

            # Add query with section only
            queries.append(SearchQuery(
                prompt=random.choice(get_topic_variations(topic)),
                section=section
            ))

    # Add cross-section queries if requested
    if include_cross_section:
        for topic in CROSS_SECTION_TOPICS:
            # Add date-only query
            if dates:
                date = random.choice(dates)
                queries.append(SearchQuery(
                    prompt=random.choice(get_topic_variations(topic)),
                    date=date
                ))

            # Add pure topic query
            queries.append(SearchQuery(
                prompt=random.choice(get_topic_variations(topic))
            ))

    return queries


def get_relevant_documents(
    db_session: Session,
    queries: List[SearchQuery],
    use_processed: bool = False,
    relevance_threshold: float = 0.5
) -> Dict[str, List[str]]:
    """
    Identify relevant documents for test queries using database content.

    Parameters
    ----------
    db_session : Session
        SQLAlchemy database session
    queries : List[SearchQuery]
        Test queries to find relevant documents for
    use_processed : bool, optional
        Whether to use processed articles
    relevance_threshold : float, optional
        Similarity threshold for relevance

    Returns
    -------
    Dict[str, List[str]]
        Mapping of query prompts to lists of relevant document IDs
    """
    relevant_docs = {}

    for query in queries:
        # Build base query
        if use_processed:
            stmt = select(ProcessedArticle, Article).join(Article)
        else:
            stmt = select(Article)

        # Apply filters
        filters = []
        if query.date:
            date = datetime.strptime(query.date, '%Y-%m-%d')
            filters.append(Article.published_at >= date)
            filters.append(Article.published_at < date + timedelta(days=1))

        if query.section:
            filters.append(Article.section.has(name=query.section))

        if filters:
            stmt = stmt.where(and_(*filters))

        # Execute query
        results = db_session.execute(stmt).fetchall()

        # Extract the search terms from the query
        search_terms = [term.lower() for term in query.prompt.split()
                        if term.lower() not in {'noticias', 'sobre', 'información',
                                                'de', 'últimas', 'actualidad'}]

        # Get relevant documents based on content similarity
        relevant = []
        for row in results:
            article = row[0] if use_processed else row
            content = article.processed_content if use_processed else article.content
            content_lower = content.lower()

            # Check if any search term appears in the content
            if any(term in content_lower for term in search_terms):
                relevant.append(str(article.id))

        relevant_docs[query.prompt] = relevant

    return relevant_docs


def main():
    """Generate a test set and save it."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Generate evaluation test set')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for test set')
    parser.add_argument('--queries-per-section', type=int, default=5,
                        help='Number of queries to generate per section')
    parser.add_argument('--no-cross-section', action='store_true',
                        help='Disable cross-section topics')

    args = parser.parse_args()

    # Generate queries
    queries = generate_test_queries(
        queries_per_section=args.queries_per_section,
        include_cross_section=not args.no_cross_section
    )

    # Save queries
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    queries_file = output_dir / 'test_queries.json'
    with open(queries_file, 'w') as f:
        json.dump([
            {
                'prompt': q.prompt,
                'date': q.date,
                'section': q.section
            }
            for q in queries
        ], f, indent=2)

    # Generate summary
    summary = {
        'total_queries': len(queries),
        'by_section': {
            section: len([q for q in queries if q.section == section])
            for section in SECTION_TOPICS.keys()
        },
        'with_date': len([q for q in queries if q.date]),
        'with_section': len([q for q in queries if q.section]),
        'cross_section': len([q for q in queries
                              if not q.section and
                              any(topic in q.prompt
                                  for topic in CROSS_SECTION_TOPICS)])
    }

    summary_file = output_dir / 'test_set_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Generated {len(queries)} test queries")
    print(f"Saved to {queries_file}")
    print(f"Summary saved to {summary_file}")


if __name__ == '__main__':
    main()
