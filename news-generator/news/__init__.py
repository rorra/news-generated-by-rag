"""News module initialization."""

from .fetcher import NewsFetcher
from .selector import NewsSelector

__all__ = ['NewsFetcher', 'NewsSelector']
