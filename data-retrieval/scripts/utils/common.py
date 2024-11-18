"""
Common utility functions shared across scripts.

This module provides common functionality used by multiple scripts,
including configuration loading and client initialization.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse


def load_config() -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Returns
    -------
    Dict[str, Any]
        Configuration dictionary

    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    """
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Replace environment variables in the config
    if '${DATABASE_URL}' in config.get('database', {}).get('url', ''):
        config['database']['url'] = os.getenv('DATABASE_URL')

    if '${QDRANT_API_KEY}' in config.get('qdrant', {}).get('cloud', {}).get('api_key', ''):
        config['qdrant']['cloud']['api_key'] = os.getenv('QDRANT_API_KEY')

    return config


def get_qdrant_client(local: bool = True) -> QdrantClient:
    """
    Initialize Qdrant client based on configuration.

    Parameters
    ----------
    local : bool
        Whether to use local or cloud instance

    Returns
    -------
    QdrantClient
        Initialized Qdrant client

    Raises
    ------
    ValueError
        If cloud configuration is missing required parameters
    """
    config = load_config()

    if local:
        local_config = config['qdrant']['local']
        return QdrantClient(
            host=local_config['host'],
            port=local_config['port']
        )
    else:
        cloud_config = config['qdrant']['cloud']
        if not os.getenv('QDRANT_API_KEY'):
            raise ValueError("QDRANT_API_KEY environment variable not set")

        return QdrantClient(
            url=cloud_config['url'],
            api_key=os.getenv('QDRANT_API_KEY')
        )


def get_embedder(embedder_type: str, config: dict, articles: Optional[list] = None):
    """
    Get the appropriate embedder instance and fit if necessary.

    Parameters
    ----------
    embedder_type : str
        Type of embedder to create
    config : dict
        Configuration dictionary
    articles : Optional[list]
        Articles to fit the embedder on (for TF-IDF and BM25)

    Returns
    -------
    BaseEmbedder
        Initialized embedder instance
    """
    from embedders import (
        TfidfEmbedder,
        SBERTEmbedder,
        DPREmbedder,
        MiniLMEmbedder,
        BM25Embedder
    )

    embedder_config = config['embedders'].get(embedder_type, {})

    if embedder_type == 'tfidf':
        embedder = TfidfEmbedder(max_features=embedder_config.get('max_features', 384))
        if articles:
            print(f"Fitting {embedder_type} embedder on {len(articles)} articles...")
            embedder.fit([article['content'] for article in articles])
    elif embedder_type == 'bm25':
        embedder = BM25Embedder()
        if articles:
            print(f"Fitting {embedder_type} embedder on {len(articles)} articles...")
            embedder.fit([article['content'] for article in articles])
    elif embedder_type == 'sbert':
        embedder = SBERTEmbedder(model_name=embedder_config['model_name'])
    elif embedder_type == 'dpr':
        embedder = DPREmbedder(model_name=embedder_config['model_name'])
    elif embedder_type == 'minilm':
        embedder = MiniLMEmbedder(model_name=embedder_config['model_name'])
    else:
        raise ValueError(f"Invalid embedder type: {embedder_type}")

    return embedder
