# News Article RAG System

A Retrieval-Augmented Generation (RAG) system for Spanish news articles, implementing multiple embedding strategies and
vector search capabilities with support for keyword-based retrieval.

## Overview

This project implements a RAG system designed to work with Spanish news articles. It provides multiple embedding
approaches and vector search capabilities, allowing for efficient semantic search, keyword-based search, and hybrid
retrieval of news content.

## Features

- Multiple search approaches:
  - Semantic search using embeddings
  - Keyword-based search with relevance scores
  - Hybrid search combining both approaches
  
- Multiple embedding strategies:
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - BM25 (Best Match 25 algorithm)
  - DPR (Dense Passage Retrieval)
  - SBERT (Spanish-tuned Sentence-BERT)
  - MiniLM (Efficient multilingual embeddings)

- Advanced keyword functionality:
  - Keyword scoring and relevance ranking
  - Keyword extraction and filtering
  - Score-based keyword matching
  - Minimum score thresholds

- Content filtering:
  - Word count constraints (500-20,000 words)
  - Section-based filtering
  - Date-based filtering
  - Keyword score filtering

- Database integration:
  - SQLAlchemy models for articles and processed content
  - Support for both raw and preprocessed articles
  - Keyword storage and retrieval

- Vector storage:
  - Qdrant integration for efficient vector search
  - Metadata filtering capabilities
  - Cosine similarity search
  - Keyword score indexing

## Setup

1. Setup PYTHONPATH environment variable:

```bash
export PYTHONPATH="$(pwd):$(pwd)/../data-mining"
```

2. Create a `.env` file with your configuration:
```plaintext
DATABASE_URL=mysql+mysqlconnector://root:password@localhost/news_db
QDRANT_API_KEY=your_api_key
```

3. Create your config.yaml file based on the sample:

```bash
cp config/config.yaml.sample config/config.yaml
```

Then edit config/config.yaml with your settings:

```yaml
database:
  url: ${DATABASE_URL}

qdrant:
  # For local development (Docker)
  local:
    host: localhost
    port: 6333
  # For production (Cloud)
  cloud:
    url: "https://YOUR-CLUSTER-URL.qdrant.io"
    api_key: ${QDRANT_API_KEY}

embedders:
  tfidf:
    max_features: 384

  sbert:
    model_name: "hiiamsid/sentence_similarity_spanish_es"

  dpr:
    model_name: "facebook/dpr-question_encoder-single-nq-base"

  minilm:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"

processing:
  batch_size: 32
  min_words: 500
  max_words: 20000
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Managing Qdrant Collections

List collections:
```bash
# Simple list
python scripts/manage_qdrant.py --local list

# Detailed information
python scripts/manage_qdrant.py --local list --detailed
```

Delete collections:
```bash
# Delete specific collections
python scripts/manage_qdrant.py --local delete --collections news_sbert news_tfidf

# Delete all collections
python scripts/manage_qdrant.py --local delete --force

# Delete all except specific ones
python scripts/manage_qdrant.py --local delete --exclude news_sbert news_tfidf
```

### Indexing Articles

Basic indexing:
```bash
# Index using specific embedder (local Qdrant)
python scripts/index_articles.py --local --embedder-type minilm

# Use processed articles with keyword filtering
python scripts/index_articles.py --local --embedder-type minilm --use-processed --min-keyword-score 0.1

# Specify batch size
python scripts/index_articles.py --local --embedder-type minilm --batch-size 64
```

Multiple embedder indexing:
```bash
# Index with all embedders
for embedder in tfidf bm25 dpr sbert minilm; do
    python scripts/index_articles.py --local --embedder-type $embedder --use-processed
done
```

### Searching Articles

Basic search:
```bash
# Semantic search
python scripts/search_news.py --prompt "Noticias importantes" --embedder minilm --local

# Keyword search
python scripts/search_news.py --keywords "economía" "inflación" --min-keyword-score 0.2 --local

# Combined search
python scripts/search_news.py --prompt "crisis económica" --keywords "economía" "inflación" --local
```

Advanced search options:
```bash
# Search with filters
python scripts/search_news.py --prompt "análisis económico" \
    --keywords "g20" "dólar" \
    --date 2024-11-18 \
    --section "Economía" \
    --min-keyword-score 0.3 \
    --match-any-keyword \
    --sort-by-keyword-score \
    --limit 20 \
    --embedder minilm \
    --local
```

### Evaluating Embedders

Generate test set:
```bash
# Basic test set generation
python scripts/generate_test_set.py --output evaluation/test_sets/

# Advanced test set with keyword settings
python scripts/generate_test_set.py \
    --output evaluation/test_sets/ \
    --queries-per-section 10 \
    --min-keyword-score 0.2
```

Run evaluation:
```bash
# Evaluate all embedders
python scripts/evaluate_embeddings.py \
    --local \
    --output evaluation/results/ \
    --keyword-match-threshold 0.3 \
    --min-keyword-score 0.2

# Evaluate specific embedders
python scripts/evaluate_embeddings.py \
    --local \
    --output evaluation/results/ \
    --embedders tfidf sbert \
    --keyword-match-threshold 0.3
```

### Visualizing Results

Generate visualizations:
```bash
# Generate complete report
python scripts/visualize_results.py \
    --results-dir evaluation/results \
    --output-dir evaluation/visualizations

# Generate specific formats
python scripts/visualize_results.py \
    --results-dir evaluation/results \
    --output-dir evaluation/visualizations \
    --format html  # or 'png' or 'all'
```

## Understanding Keyword Scores

Keywords in articles are stored with relevance scores in the format:
```
(keyword1,0.306),(keyword2,0.216),(keyword3,0.205)
```

- Scores range from 0 to 1, indicating keyword relevance
- Higher scores indicate stronger keyword relevance
- Typical threshold ranges:
  - 0.3+ : Strong relevance
  - 0.2-0.3 : Moderate relevance
  - <0.2 : Weak relevance

## Common Issues

1. **Keyword Score Filtering**: If no results are returned when using keyword search, try lowering the `min-keyword-score` threshold.

2. **Memory Issues**: When processing large datasets with keywords, adjust the batch sizes in index_articles.py and evaluate_embeddings.py.

3. **Missing Test Queries**: Ensure you've generated test queries with appropriate keyword settings:
```bash
python scripts/generate_test_set.py --output evaluation/test_sets/ --min-keyword-score 0.2
```

4. **Qdrant Connection**: For local development, ensure Qdrant is running. For cloud, verify your API key is set in .env.

## Development

### Common Utilities

Common functionalities are centralized in `scripts/utils/common.py`:
- Configuration loading
- Qdrant client initialization
- Embedder initialization
- Keyword processing utilities

Example:
```python
from utils.common import load_config, get_qdrant_client, get_embedder

# Load configuration
config = load_config()

# Get Qdrant client
qdrant = get_qdrant_client(local=True)

# Initialize embedder with keyword support
embedder = get_embedder('minilm', config, articles)
```