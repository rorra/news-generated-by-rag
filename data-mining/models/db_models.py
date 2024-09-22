from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Newspaper(Base):
    __tablename__ = 'newspapers'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    url = Column(String(255), nullable=False)
    articles = relationship("Article", back_populates="newspaper")

    def __repr__(self):
        return f"<Newspaper(id={self.id}, name='{self.name}')>"


class Section(Base):
    __tablename__ = 'sections'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    articles = relationship("Article", back_populates="section")

    def __repr__(self):
        return f"<Section(id={self.id}, name='{self.name}')>"


class Article(Base):
    __tablename__ = 'articles'

    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    link = Column(String(255), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)  # New field to store publication datetime

    # Foreign keys
    newspaper_id = Column(Integer, ForeignKey('newspapers.id'), nullable=False)
    section_id = Column(Integer, ForeignKey('sections.id'), nullable=False)

    # Relationships
    newspaper = relationship("Newspaper", back_populates="articles")
    section = relationship("Section", back_populates="articles")

    # Indexes
    __table_args__ = (
        Index('ix_articles_newspaper_id', 'newspaper_id'),
        Index('ix_articles_section_id', 'section_id'),
        Index('ix_articles_published_at', 'published_at'),
    )

    def __repr__(self):
        return f"<Article(id={self.id}, title='{self.title}', newspaper_id={self.newspaper_id}, section_id={self.section_id})>"
