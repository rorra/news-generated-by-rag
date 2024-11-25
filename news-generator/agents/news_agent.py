"""
News Selection Agent

Uses OpenAI and LangChain to select important news from each section based on RAG retrieval.
"""

from datetime import datetime
from typing import List, Dict, Optional
import openai
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from news.fetcher import NewsFetcher
from news.selector import NewsSelector
from storage.qdrant_manager import QdrantManager


class NewsSelectionAgent:
    def __init__(
        self,
        openai_model: str = "gpt-4o-mini",
        local_qdrant: bool = True,
        news_per_section: int = 5
    ):
        """Initialize the news selection agent."""
        self.llm = ChatOpenAI(model=openai_model)
        self.fetcher = NewsFetcher(local=local_qdrant)
        qdrant = QdrantManager(local=local_qdrant)
        self.selector = NewsSelector(qdrant_manager=qdrant)
        self.news_per_section = news_per_section

        # Define tool for news retrieval
        self.tools = [
            Tool(
                name="fetch_section_news",
                func=self._fetch_section_news,
                description="Fetches news for a specific section and today's date"
            )
        ]

    def _fetch_section_news(self, section: str) -> List[Dict]:
        """Fetch and filter news for a section."""
        today = datetime.now().strftime("%Y-%m-%d")
        raw_news = self.fetcher.fetch_news(date=today, section=section)
        return self.selector.select_unique_news(raw_news)

    def _create_selection_prompt(self, section: str) -> str:
        """Create prompt for news selection."""
        return f"""Analiza las noticias de la sección {section} y selecciona las {self.news_per_section} más importantes.
        Ten en cuenta:
        - Relevancia para el mercado argentino
        - Impacto en la actualidad
        - Diversidad de temas
        - Evita noticias similares o redundantes

        Devuelve solo los títulos de las noticias seleccionadas. No númeres las noticias, solo escribe el título."""

    def select_news_for_section(self, section: str) -> List[str]:
        """Select important news titles for a section."""
        news_data = self._fetch_section_news(section)

        if not news_data:
            return []

        prompt = self._create_selection_prompt(section)
        titles = [news['title'] for news in news_data]

        # Ask LLM to select important news
        response = self.llm.predict(prompt + "\n\nNoticias disponibles:\n" + "\n".join(titles))

        # Extract selected titles from response
        selected_titles = [title.strip() for title in response.split("\n") if title.strip()]
        return selected_titles[:self.news_per_section]

    def select_all_sections(self) -> Dict[str, List[str]]:
        """Select news for all sections."""
        sections = ["Economía", "Internacional", "Política", "Sociedad"]
        return {
            section: self.select_news_for_section(section) for section in sections
        }
