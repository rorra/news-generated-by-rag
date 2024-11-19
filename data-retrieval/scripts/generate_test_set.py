"""
Test Set Generator

This script provides utilities for generating test queries and identifying
relevant documents for RAG system evaluation, with section-specific topics
and keyword-based test cases.
"""

from typing import List, Dict, Set, Optional, Tuple
from datetime import datetime, timedelta
import random
import json
from pathlib import Path
from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session

from evaluation.metrics import SearchQuery
from models.db_models import Article, ProcessedArticle, Section
from storage.data_loader import parse_keywords

# Define section-specific topics with associated keywords
SECTION_TOPICS = {
    "Economía": [
        {
            "topic": "apreciación del peso",
            "keywords": ["peso", "dólar", "tipo de cambio", "mercado cambiario"]
        },
        {
            "topic": "dólar MEP",
            "keywords": ["dólar", "mep", "bolsa", "bonos"]
        },
        {
            "topic": "pesquera china",
            "keywords": ["pesca", "china", "mar", "buques"]
        },
        {
            "topic": "pymes caputo",
            "keywords": ["pymes", "caputo", "empresas", "impuestos"]
        },
        {
            "topic": "transferencia de emisiones",
            "keywords": ["emisiones", "carbono", "clima", "ambiente"]
        }
    ],
    "Internacional": [
        {
            "topic": "cambio climático",
            "keywords": ["clima", "calentamiento", "emisiones", "ambiente"]
        },
        {
            "topic": "reloj de oro del titanic",
            "keywords": ["titanic", "reloj", "subasta", "naufragio"]
        },
        {
            "topic": "seguridad social en estados unidos",
            "keywords": ["seguridad social", "eeuu", "pensiones", "jubilación"]
        },
        {
            "topic": "g20 brasil",
            "keywords": ["g20", "brasil", "cumbre", "lula"]
        },
        {
            "topic": "donald trump",
            "keywords": ["trump", "elecciones", "eeuu", "republicano"]
        }
    ],
    "Política": [
        {
            "topic": "congreso nacional",
            "keywords": ["congreso", "diputados", "senadores", "leyes"]
        },
        {
            "topic": "Emanuel Macron",
            "keywords": ["macron", "francia", "europa", "presidente"]
        },
        {
            "topic": "kirchnerismo",
            "keywords": ["kirchner", "peronismo", "política", "justicia"]
        },
        {
            "topic": "Estados Unidos",
            "keywords": ["eeuu", "biden", "washington", "política"]
        },
        {
            "topic": "Cristina Kirchner",
            "keywords": ["cristina", "kirchner", "senado", "justicialismo"]
        }
    ],
    "Sociedad": [
        {
            "topic": "Lionsgate",
            "keywords": ["lionsgate", "cine", "película", "entertainment"]
        },
        {
            "topic": "Efemérides",
            "keywords": ["efemérides", "historia", "aniversario", "conmemoración"]
        },
        {
            "topic": "hormigas voladoras",
            "keywords": ["hormigas", "insectos", "naturaleza", "clima"]
        },
        {
            "topic": "clima",
            "keywords": ["temperatura", "lluvia", "pronóstico", "meteorología"]
        },
        {
            "topic": "Andrea Giunta",
            "keywords": ["arte", "cultura", "exposición", "museo"]
        }
    ]
}

# Define cross-section topics with associated keywords
CROSS_SECTION_TOPICS = [
    {
        "topic": "crisis económica",
        "keywords": ["crisis", "economía", "inflación", "recesión"]
    },
    {
        "topic": "presupuesto nacional",
        "keywords": ["presupuesto", "gasto", "congreso", "fiscal"]
    },
    {
        "topic": "políticas públicas",
        "keywords": ["política", "estado", "gestión", "gobierno"]
    },
    {
        "topic": "impacto social",
        "keywords": ["social", "sociedad", "impacto", "comunidad"]
    }
]


def get_topic_variations(topic_info: Dict) -> List[Dict]:
    """
    Generate variations of a topic with associated keywords.

    Parameters
    ----------
    topic_info : Dict
        Dictionary containing topic and its keywords

    Returns
    -------
    List[Dict]
        List of topic variations with keywords
    """
    base_topic = topic_info["topic"]
    variations = [
        f"noticias sobre {base_topic}",
        f"información de {base_topic}",
        f"últimas noticias de {base_topic}",
        f"actualidad sobre {base_topic}",
        base_topic
    ]

    return [
        {
            "prompt": variation,
            "keywords": topic_info["keywords"]
        }
        for variation in variations
    ]


def get_keyword_scores_from_db(
    db_session: Session,
    keywords: List[str],
    min_articles: int = 5
) -> Dict[str, float]:
    """
    Get average scores for keywords from the database.

    Parameters
    ----------
    db_session : Session
        Database session
    keywords : List[str]
        List of keywords to look up
    min_articles : int
        Minimum number of articles containing the keyword

    Returns
    -------
    Dict[str, float]
        Dictionary mapping keywords to their average scores
    """
    # Query processed articles
    stmt = select(ProcessedArticle)
    results = db_session.execute(stmt).fetchall()

    keyword_scores = {}
    for keyword in keywords:
        scores = []
        for result in results:
            processed_article = result[0]
            if processed_article.keywords:
                parsed_keywords = parse_keywords(processed_article.keywords)
                for kw, score in parsed_keywords:
                    if kw.lower() == keyword.lower():
                        scores.append(score)

        if len(scores) >= min_articles:
            keyword_scores[keyword] = sum(scores) / len(scores)

    return keyword_scores


def generate_test_queries(
    db_session: Session,
    dates: Optional[List[str]] = None,
    sections: Optional[List[str]] = None,
    include_cross_section: bool = True,
    queries_per_section: int = 5,
    min_keyword_score: float = 0.1
) -> List[Dict]:
    """
    Generate a comprehensive set of test queries with sections and keywords.

    Parameters
    ----------
    db_session : Session
        Database session for getting keyword scores
    dates : Optional[List[str]]
        List of dates in YYYY-MM-DD format
    sections : Optional[List[str]]
        List of section names
    include_cross_section : bool
        Whether to include cross-section topics
    queries_per_section : int
        Number of queries to generate per section
    min_keyword_score : float
        Minimum score threshold for including keywords

    Returns
    -------
    List[Dict]
        Generated test queries with keywords
    """
    if not dates:
        today = datetime.now()
        dates = [
            (today - timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(7)
        ]

    if not sections:
        sections = list(SECTION_TOPICS.keys())

    queries = []

    # Generate section-specific queries
    for section in sections:
        section_topics = SECTION_TOPICS[section]
        selected_topics = random.sample(
            section_topics,
            min(queries_per_section, len(section_topics))
        )

        for topic_info in selected_topics:
            # Get keyword scores from database
            keyword_scores = get_keyword_scores_from_db(db_session, topic_info["keywords"])
            filtered_keywords = [
                k for k in topic_info["keywords"]
                if k in keyword_scores and keyword_scores[k] >= min_keyword_score
            ]

            # Generate variations with both prompt and keywords
            variations = get_topic_variations({
                "topic": topic_info["topic"],
                "keywords": filtered_keywords
            })

            # Add queries with date and section
            for variation in variations:
                if dates:
                    date = random.choice(dates)
                    queries.append({
                        "prompt": variation["prompt"],
                        "keywords": variation["keywords"],
                        "date": date,
                        "section": section,
                        "min_keyword_score": min_keyword_score
                    })

                # Add keyword-only query
                queries.append({
                    "prompt": None,
                    "keywords": variation["keywords"],
                    "date": None,
                    "section": section,
                    "min_keyword_score": min_keyword_score
                })

    # Add cross-section queries
    if include_cross_section:
        for topic_info in CROSS_SECTION_TOPICS:
            # Get keyword scores from database
            keyword_scores = get_keyword_scores_from_db(db_session, topic_info["keywords"])
            filtered_keywords = [
                k for k in topic_info["keywords"]
                if k in keyword_scores and keyword_scores[k] >= min_keyword_score
            ]

            variations = get_topic_variations({
                "topic": topic_info["topic"],
                "keywords": filtered_keywords
            })

            for variation in variations:
                # Add date-specific query
                if dates:
                    date = random.choice(dates)
                    queries.append({
                        "prompt": variation["prompt"],
                        "keywords": variation["keywords"],
                        "date": date,
                        "section": None,
                        "min_keyword_score": min_keyword_score
                    })

                # Add keyword-only query
                queries.append({
                    "prompt": None,
                    "keywords": variation["keywords"],
                    "date": None,
                    "section": None,
                    "min_keyword_score": min_keyword_score
                })

    return queries


def generate_test_set_summary(queries: List[Dict]) -> Dict:
    """
    Generate summary statistics for the test set.

    Parameters
    ----------
    queries : List[Dict]
        List of generated queries

    Returns
    -------
    Dict
        Summary statistics
    """
    return {
        'total_queries': len(queries),
        'by_section': {
            section: len([q for q in queries if q.get('section') == section])
            for section in SECTION_TOPICS.keys()
        },
        'query_types': {
            'semantic_only': len([q for q in queries if q.get('prompt') and not q.get('keywords')]),
            'keyword_only': len([q for q in queries if q.get('keywords') and not q.get('prompt')]),
            'combined': len([q for q in queries if q.get('prompt') and q.get('keywords')]),
        },
        'with_date': len([q for q in queries if q.get('date')]),
        'with_section': len([q for q in queries if q.get('section')]),
        'cross_section': len([q for q in queries if not q.get('section')])
    }


def main():
    """Generate a test set and save it."""
    import argparse
    from config import SessionLocal

    parser = argparse.ArgumentParser(description='Generate evaluation test set')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for test set')
    parser.add_argument('--queries-per-section', type=int, default=5,
                        help='Number of queries to generate per section')
    parser.add_argument('--no-cross-section', action='store_true',
                        help='Disable cross-section topics')
    parser.add_argument('--min-keyword-score', type=float, default=0.1,
                        help='Minimum score threshold for including keywords')

    args = parser.parse_args()

    # Initialize database session
    session = SessionLocal()

    try:
        # Generate queries
        queries = generate_test_queries(
            db_session=session,
            queries_per_section=args.queries_per_section,
            include_cross_section=not args.no_cross_section,
            min_keyword_score=args.min_keyword_score
        )

        # Save queries
        output_dir = Path(args.output)
        output_dir.mkdir(exist_ok=True)

        queries_file = output_dir / 'test_queries.json'
        with open(queries_file, 'w') as f:
            json.dump(queries, f, indent=2)

        # Generate and save summary
        summary = generate_test_set_summary(queries)
        summary_file = output_dir / 'test_set_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Generated {len(queries)} test queries")
        print("\nQuery Distribution:")
        print(f"Semantic-only queries: {summary['query_types']['semantic_only']}")
        print(f"Keyword-only queries: {summary['query_types']['keyword_only']}")
        print(f"Combined queries: {summary['query_types']['combined']}")
        print(f"\nSaved to {queries_file}")
        print(f"Summary saved to {summary_file}")

    finally:
        session.close()


if __name__ == '__main__':
    main()
