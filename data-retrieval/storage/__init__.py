"""
Storage Package

This package provides functionality for data storage and retrieval operations,
including database access and vector storage management.

Available Modules
---------------
data_loader
    Functions for loading and filtering articles from the database
qdrant_manager
    Manages interactions with the Qdrant vector database

The storage package handles all data persistence and retrieval operations,
providing a clean interface between the database and the application logic.
"""

from .data_loader import load_articles_from_db
from .qdrant_manager import QdrantManager

__all__ = [
    'load_articles_from_db',
    'QdrantManager'
]
