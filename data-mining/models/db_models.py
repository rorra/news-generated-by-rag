from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Newspaper(Base):
    __tablename__ = "newspapers"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    url = Column(String(255), nullable=False)
    articles = relationship("Article", back_populates="newspaper")

    def __repr__(self):
        return f"<Newspaper(id={self.id}, name='{self.name}')>"


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    articles = relationship("Article", back_populates="section")

    def __repr__(self):
        return f"<Section(id={self.id}, name='{self.name}')>"


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    link = Column(String(255), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)

    # Foreign keys
    newspaper_id = Column(Integer, ForeignKey("newspapers.id"), nullable=False)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)

    # Relationships
    newspaper = relationship("Newspaper", back_populates="articles")
    section = relationship("Section", back_populates="articles")
    processed_article = relationship("ProcessedArticle", uselist=False, back_populates="article")

    # Indexes
    __table_args__ = (
        Index("ix_articles_newspaper_id", "newspaper_id"),
        Index("ix_articles_section_id", "section_id"),
        Index("ix_articles_published_at", "published_at"),
    )

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title}', newspaper_id={self.newspaper_id}, section_id={self.section_id})>"


class ProcessedArticle(Base):
    """Model for storing preprocessed articles."""
    __tablename__ = "processed_articles"

    id = Column(Integer, primary_key=True)
    article_id = Column(Integer, ForeignKey("articles.id"), unique=True, nullable=False)
    article_created_at = Column(DateTime, nullable=False)
    processed_title = Column(String(255), nullable=False)
    processed_content = Column(Text, nullable=False)
    processed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    keywords = Column(String(255), nullable=False)

    # Relationship to original article
    article = relationship("Article", back_populates="processed_article")

    # Indexes
    __table_args__ = (
        Index("ix_processed_articles_article_id", "article_id"),
        Index("ix_processed_articles_article_created_at", "article_created_at"),
        Index("ix_processed_articles_processed_at", "processed_at"),
    )

    def __repr__(self):
        return f"<ProcessedArticle(id={self.id}, article_id={self.article_id})>"

    @classmethod
    def from_article(cls, article: Article, processed_title: str, processed_content: str,
                     keywords: str) -> "ProcessedArticle":
        """Create a ProcessedArticle instance from an original Article."""
        return cls(
            article_id=article.id,
            article_created_at=article.created_at,
            processed_title=processed_title,
            processed_content=processed_content,
            keywords=keywords
        )


class GeneratedNews(Base):
    """Model for storing generated news articles."""
    __tablename__ = "generated_news"

    id = Column(Integer, primary_key=True)
    section_id = Column(Integer, ForeignKey("sections.id"), nullable=False)
    title = Column(String(255), nullable=False)
    body = Column(Text, nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    section = relationship("Section", back_populates="generated_news")

    # Indexes
    __table_args__ = (
        Index("ix_generated_news_section_id", "section_id"),
        Index("ix_generated_news_generated_at", "generated_at"),
    )

    def __repr__(self):
        return f"<GeneratedNews(id={self.id}, section_id={self.section_id}, title='{self.title}')>"
   