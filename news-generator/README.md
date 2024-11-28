# AI News Generator

An intelligent news generation system that uses RAG (Retrieval-Augmented Generation) and multiple AI writer agents to
create news articles in different writing styles.

## Overview

This system processes and generates news articles by:

1. Selecting important news from different sections using AI
2. Using RAG to retrieve relevant context from a news database
3. Generating new articles in different writing styles (NY Times, Left-wing, Right-wing)
4. Storing the generated content in a structured database

## Architecture

The project is organized into several key components:

### Agents

- `NewsSelectionAgent`: Selects important news from each section using OpenAI and LangChain
- `WriterAgents`: Different writer personalities (NYTimes, Left-wing, Right-wing) that generate articles

### News Processing

- `NewsFetcher`: Handles RAG retrieval using Qdrant
- `NewsSelector`: Filters and selects unique news articles
- `NewsGenerator`: Orchestrates the article generation process

### Database Models

- `Newspaper`: Stores newspaper sources
- `Section`: Different news sections (Economy, International, Politics, Society)
- `Article`: Original news articles
- `ProcessedArticle`: Preprocessed article content
- `GeneratedNews`: AI-generated news articles

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment variables:

```bash
cp env.sample .env
```

Edit `.env` with your configurations:

```
DATABASE_URL=mysql+mysqlconnector://root:password@localhost/news_db
QDRANT_API_KEY=your_api_key
OPENAI_API_KEY="your-openai-api-key"
```

## Configuration

The system uses several configuration files in the `ngconfig` directory:

### OpenAI Configuration

```python
OPENAI_CONFIG = {
    "model": "gpt-4o-mini",
    "temperature": 0.3
}
```

### News Settings

```python
NEWS_SETTINGS = {
    "news_per_section": 5,
    "similarity_threshold": 0.8,
    "max_fetch_limit": 100
}
```

## Usage

Run the main script to start the news generation process:

```bash
export PYTHONPATH="$(pwd):$(pwd)/../data-mining:$(pwd)/../data-retrieval"
python main.py
```

The system will:

1. Select important news from each section
2. Generate new articles using different writer styles
3. Store the generated content in the database

## Writer Styles

The system supports multiple writing styles:

- **NY Times Style**: Formal, objective, and detailed
- **Left-Wing**: Progressive perspective focusing on social justice
- **Right-Wing**: Conservative perspective emphasizing market principles
