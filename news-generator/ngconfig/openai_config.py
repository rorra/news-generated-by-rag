"""OpenAI configuration settings."""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_CONFIG = {
    "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "api_key": os.getenv("OPENAI_API_KEY"),
    "temperature": 0.3,
}

SECTIONS = ["Economía", "Internacional", "Política", "Sociedad"]

NEWS_SETTINGS = {
    "news_per_section": 5,
    "similarity_threshold": 0.8,
    "max_fetch_limit": 100,
    "default_embedder": "minilm",
}
