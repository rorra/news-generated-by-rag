"""
Writer Agents Module

Contains different writer agents that generate news articles using different styles and perspectives.
Implements RAG pattern by retrieving relevant articles from Qdrant before generation.
"""

import random
from typing import List, Dict, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from storage.qdrant_manager import QdrantManager
from models.db_models import GeneratedNews, Article
from ngconfig.openai_config import OPENAI_CONFIG, WRITER_PROMPTS
from embedders import MiniLMEmbedder


class BaseWriterAgent:
    """Base class for all writer agents."""

    def __init__(
        self,
        qdrant: QdrantManager,
        db_session: Session,
        collection_name: str = "news_minilm",
        max_documents: int = 5,
        model: str = "gpt-4o-mini"
    ):
        self.qdrant = qdrant
        self.db_session = db_session
        self.collection_name = collection_name
        self.max_documents = max_documents
        self.llm = ChatOpenAI(model=model, temperature=OPENAI_CONFIG["temperature"])
        self.embedder = MiniLMEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def fetch_relevant_articles(self, title: str, section: str, date: str) -> List[Dict]:
        """Fetch relevant articles from Qdrant for RAG."""
        filter_conditions = {
            'section': section
        }

        # Embed the title to get relevant articles
        embedded_title = self.embedder.embed(title)

        # Get relevant articles from Qdrant
        qdrant_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=embedded_title,
            filter_conditions=filter_conditions,
            limit=self.max_documents
        )

        # Extract original ids from Qdrant results
        original_ids = [result['original_id'] for result in qdrant_results]

        # Fetch full articles from database
        articles = self.db_session.query(Article).filter(
            Article.id.in_(original_ids)
        ).all()

        # Convert to dictionary format
        full_articles = []
        for article in articles:
            full_articles.append({
                'id': article.id,
                'title': article.title,
                'content': article.content,
                'section': article.section.name,
                'published_at': article.published_at,
                'score': next((r['score'] for r in qdrant_results if r['original_id'] == article.id), 0)
            })

        return full_articles

    def _prepare_context(self, articles: List[Dict]) -> str:
        """Prepare context from retrieved articles."""
        # Sort articles by score
        sorted_articles = sorted(articles, key=lambda x: x['score'], reverse=True)

        context = []
        for article in sorted_articles:
            context.append(
                f"**{article['title']}**\n"
                f"{article['content']}\n\n"
                f"Fecha: {article['published_at']}\n"
                f"Score: {article['score']:.2f}\n"
                "---\n"
            )
        return "\n".join(context)

    def generate_article(self, title: str, section: str, date: str, style_prompt: str) -> str:
        """Generate article using RAG and specific style."""
        relevant_articles = self.fetch_relevant_articles(title, section, date)
        context = self._prepare_context(relevant_articles)

        prompt = ChatPromptTemplate.from_template(
            """Eres un periodista escribiendo un artículo sobre: {title}
 
            Contexto de artículos relacionados:
            {context}
 
            Estilo de escritura:
            {style}
 
            Genera un artículo completo manteniendo este estilo. El artículo debe ser informativo y 
            mantener un tono profesional. Utiliza la información proporcionada en el contexto para 
            enriquecer el artículo con datos y detalles relevantes, priorizando la información
            de los artículos con mayor puntaje de relevancia.
            
            Responde en formato json con las claves titulo y cuerpo, no agregues el ```, solo necesito el json para 
            parsearlo.
            
            Artículo:"""
        )

        chain = prompt | self.llm

        return chain.invoke({
            "title": title,
            "context": context,
            "style": style_prompt
        })


class NYTimesWriter(BaseWriterAgent):
    """NY Times style writer agent."""

    def generate_article(self, title: str, section: str, date: str) -> str:
        return super().generate_article(
            title,
            section,
            date,
            WRITER_PROMPTS["nytimes"]
        )


class LeftWingWriter(BaseWriterAgent):
    """Left-wing perspective writer agent."""

    def generate_article(self, title: str, section: str, date: str) -> str:
        return super().generate_article(
            title,
            section,
            date,
            WRITER_PROMPTS["leftwing"]
        )


class RightWingWriter(BaseWriterAgent):
    """Right-wing perspective writer agent."""

    def generate_article(self, title: str, section: str, date: str) -> str:
        return super().generate_article(
            title,
            section,
            date,
            WRITER_PROMPTS["rightwing"]
        )


class WriterFactory:
    """Factory class to create and select writer agents."""

    def __init__(self, qdrant: QdrantManager, db_session: Session):
        self.qdrant = qdrant
        self.db_session = db_session
        self.writers = {
            'nytimes': NYTimesWriter(qdrant, db_session),
            'leftwing': LeftWingWriter(qdrant, db_session),
            'rightwing': RightWingWriter(qdrant, db_session)
        }

    def get_random_writer(self) -> BaseWriterAgent:
        """Get a random writer agent."""
        writer_type = random.choice(list(self.writers.keys()))
        return self.writers[writer_type]
