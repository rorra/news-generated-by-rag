# News Article RAG System

A Retrieval-Augmented Generation (RAG) system for Spanish news articles, implementing multiple embedding strategies and
vector search capabilities.

## Overview

This project implements a RAG system designed to work with Spanish news articles. It provides multiple embedding
approaches and vector search capabilities, allowing for efficient semantic search and retrieval of news content.

## Features

- Multiple embedding strategies:
    - TF-IDF (Term Frequency-Inverse Document Frequency)
    - BM25 (Best Match 25 algorithm)
    - DPR (Dense Passage Retrieval)
    - SBERT (Spanish-tuned Sentence-BERT)
    - MiniLM (Efficient multilingual embeddings)

- Content filtering:
    - Word count constraints (500-20,000 words)
    - Section-based filtering
    - Date-based filtering

- Database integration:
    - SQLAlchemy models for articles and processed content
    - Support for both raw and preprocessed articles

- Vector storage:
    - Qdrant integration for efficient vector search
    - Metadata filtering capabilities
    - Cosine similarity search

## Setup

1. Setup PYTHONPATH environment variable:

```bash
export PYTHONPATH="$(pwd):$(pwd)/../data-mining"
```

2. Create a `.env` file with your configuration:
   ```plaintext
   DATABASE_URL=postgresql://user:password@localhost:5432/your_db
   QDRANT_API_KEY=your-qdrant-api-key
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

```bash
# Index using specific embedder (local Qdrant)
python scripts/index_articles.py --local --embedder-type tfidf

# Use processed articles
python scripts/index_articles.py --local --embedder-type tfidf --use-processed

# Specify batch size
python scripts/index_articles.py --local --embedder-type tfidf --batch-size 64

# Use cloud Qdrant
python scripts/index_articles.py --embedder-type tfidf
```

### Evaluating Embedders

Generate test set:
```bash
python scripts/generate_test_set.py --output evaluation/test_sets/
```

Run evaluation:
```bash
# Evaluate all embedders
python scripts/evaluate_embeddings.py --local --output evaluation/results/

# Evaluate specific embedders
python scripts/evaluate_embeddings.py --local --output evaluation/results/ --embedders tfidf sbert

# Use cloud Qdrant
python scripts/evaluate_embeddings.py --output evaluation/results/
```

### Visualizing Results

```bash
# Generate both HTML report and PNG images
python scripts/visualize_results.py \
    --results-dir evaluation/results \
    --output-dir evaluation/visualizations

# Generate HTML report only
python scripts/visualize_results.py \
    --results-dir evaluation/results \
    --output-dir evaluation/visualizations \
    --format html

# Generate PNG images only
python scripts/visualize_results.py \
    --results-dir evaluation/results \
    --output-dir evaluation/visualizations \
    --format png
```

## Common Issues

1. **TF-IDF/BM25 Not Fitted**: If you see "Vectorizer needs to be fitted first", ensure you're passing articles when initializing these embedders.

2. **Memory Issues**: When processing large datasets, adjust the batch sizes in index_articles.py and evaluate_embeddings.py.

3. **Missing Test Queries**: Ensure you've generated test queries before running evaluation:
```bash
python scripts/generate_test_set.py --output evaluation/test_sets/
```

4. **Qdrant Connection**: For local development, ensure Qdrant is running. For cloud, verify your API key is set in .env.

## Development

### Common Utilities

Common functionalities are centralized in `scripts/utils/common.py`:
- Configuration loading
- Qdrant client initialization
- Embedder initialization

Example:
```python
from utils.common import load_config, get_qdrant_client, get_embedder

# Load configuration
config = load_config()

# Get Qdrant client
qdrant = get_qdrant_client(local=True)

# Initialize embedder
embedder = get_embedder('tfidf', config, articles)
```
