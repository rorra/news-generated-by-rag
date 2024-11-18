import sys
import os
import re
import language_tool_python
import argparse
import json
import spacy
from datetime import datetime
from typing import Dict, Optional
from config import SessionLocal
from models.db_models import Newspaper, Section, Article

# NLTK data downloaded for NLP tasks
import nltk

nltk.download('punkt')
nltk.download('punkt_tab')

# spaCy model for spanish words
nlp = spacy.load('es_core_news_sm')

db_session = SessionLocal()

# Inicializar LanguageTool una vez, asi no se crean n= articulos de instancias
# Se aumenta mucho la velocidad y ahorramos memoria.
tool = language_tool_python.LanguageTool('es')


def correct_spelling(text: str) -> str:
    """
    Corrects spelling and grammar in the given text using LanguageTool.
    """
    corrected_text = tool.correct(text)  # Usa la misma instancia para cada noticia
    return corrected_text


def remove_duplicates(text: str) -> str:
    """
    Removes duplicate sentences from the text.

    Args:
        text (str): The input text containing potential duplicate sentences.

    Returns:
        str: The text with duplicate sentences removed.
    """
    sentences = nltk.sent_tokenize(text, language='spanish')
    unique_sentences = list(dict.fromkeys(sentences))
    return ' '.join(unique_sentences)


def normalize_text(text: str) -> str:
    """
    Cleans and normalizes text:
        - Removes unnecessary spaces.
        - Removes stop words.
        - Applies lemmatization.
    
    Args:
        text (str): Input text.

    Returns:
        str: Cleaned and normalized text.
    """

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Process the text with spaCy
    doc = nlp(text)

    # Filter tokens: remove stop words and punctuation, apply lemmatization
    normalized_words = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct
    ]

    # Join normalized words into a single string
    return ' '.join(normalized_words)


def remove_irrelevant(text: str) -> str:
    """
    Removes irrelevant information or broken links from the text.

    Args:
        text (str): The input text from which to remove irrelevant content.

    Returns:
        str: The text without irrelevant parts.
    """
    text = re.sub(r'Lee también.*?\. ', '', text)
    return text


def remove_links(text: str) -> str:
    """
    Removes hyperlinks from the text.

    Args:
        text (str): The input text that may contain hyperlinks.

    Returns:
        str: The text with hyperlinks removed.
    """
    # Regular expression pattern to identify URLs
    url_pattern = r'http[s]?://\S+|www\.\S+'
    text = re.sub(url_pattern, '', text)
    return text


def segment_paragraphs(text: str) -> str:
    """
    Segments the text into paragraphs every four sentences to improve readability.

    Args:
        text (str): The input text to segment into paragraphs.

    Returns:
        str: The text segmented into paragraphs.
    """
    sentences = nltk.sent_tokenize(text, language='spanish')
    paragraphs = []
    current_paragraph = ''
    for i, sentence in enumerate(sentences):
        current_paragraph += sentence + ' '
        if (i + 1) % 4 == 0:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = ''
    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    return '\n\n'.join(paragraphs)


def preprocess_news(publish_date: Optional[datetime] = None) -> Dict:
    """
    Preprocesses news articles published after a certain date by retrieving them from the database
    and applying text preprocessing functions.

    Args:
        publish_date (datetime, optional): The date from which to retrieve and preprocess articles.
                                           Defaults to the current date if not provided.

    Returns:
        Dict: A dictionary containing the preprocessed articles.

    Note:
        This function assumes that you have a database session and ORM models defined.
    """
    db_session = SessionLocal()
    try:
        # Limitar la consulta a las primeras 10 noticias
        articles = db_session.query(Article).filter(
            Article.published_at >= publish_date
        ).order_by(Article.published_at.desc()).all()

        print(f"Found {len(articles)} articles")
        preprocessed_articles = {}
        for article in articles:
            print(f"Processing article: {article.title}")
            text = article.content
            text = correct_spelling(text)
            text = remove_duplicates(text)
            text = normalize_text(text)
            text = remove_links(text)
            text = remove_irrelevant(text)
            text = segment_paragraphs(text)

            preprocessed_articles[article.title] = {
                'newspaper': article.newspaper.name,
                'section': article.section.name,
                'published_at': article.published_at.isoformat(),
                'title': article.title,
                'content': text
            }
        return preprocessed_articles
    except Exception as e:
        print(f"Error to query database: {e}")
    finally:
        db_session.close()


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess news articles.')
    parser.add_argument('--date', type=str, required=False, help='Publish date in YYYY-MM-DD format')

    args = parser.parse_args()

    # Convert the input date string to a datetime object
    try:
        if args.date:
            # User provided a date, set time to midnight
            publish_date = datetime.strptime(args.date, '%Y-%m-%d')
        else:
            # Use current date at midnight
            publish_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    except ValueError:
        print("Incorrect date format. Please use YYYY-MM-DD.")
        exit(1)

    print(f"Start preprocessing with publish date {publish_date}")
    preprocessed = preprocess_news(publish_date)
    print(f"Preprocessed {len(preprocessed)} files")

    # Save preprocessed files
    formatted_date = publish_date.strftime('%Y%m%d')
    output_dir = os.path.join(os.path.dirname(__file__), 'preprocessed_files')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{formatted_date}_preprocessed_files.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed, f, ensure_ascii=False, indent=4)

    print(f"Preprocessed articles saved to {output_file}")
