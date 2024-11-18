# News Article Preprocessor

A robust, modular Python application for preprocessing Spanish news articles. This system processes raw news articles
through a configurable pipeline of text preprocessing steps and stores the results in a database for later use in RAG (
Retrieval-Augmented Generation) systems.

## Features

- **Modular Architecture**: Each preprocessing step is encapsulated in its own class
- **Batch Processing**: Efficiently handles large numbers of articles
- **Error Handling**: Robust error handling and logging
- **Progress Tracking**: Real-time processing status updates
- **Database Integration**: Direct integration with SQL databases
- **Configurable Pipeline**: Easy to add, remove, or modify preprocessing steps

## Architecture

```
/
├── preprocessors/             # Preprocessing components
│   ├── base.py                # Abstract base class
│   ├── spelling.py            # Spelling correction
│   ├── duplicate_remover.py   # Duplicate sentence removal
│   ├── text_normalizer.py     # Text normalization
│   ├── content_cleaner.py     # Content cleaning
│   └── paragraph_segmenter.py # Paragraph segmentation
├── services/                  # Business logic
│   ├── article_processor.py   # Main processing service
│   └── keyword_extractor.py   # Keyword extractor from articles
├── config.py                  # Configuration settings
└── main.py                    # Application entry point
```

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
   ```

3. Install Spanish spaCy model:
   ```bash
   python -m spacy download es_core_news_sm
   ```

## Configuration

1. Configure your database connection in `.env`:
   ```
   DATABASE_URL=mysql+mysqlconnector://root:password@localhost/news_db
   ```

2. (Optional) Adjust preprocessing parameters in each preprocessor class:
    - `sentences_per_paragraph` in ParagraphSegmenter
    - `batch_size` in ArticleProcessor
    - Language settings in various preprocessors

## Usage

Run the preprocessor:

```bash
export PYTHONPATH="$(pwd):$(pwd)/../data-mining"
python main.py
```

The system will:

1. Find all unprocessed articles in the database
2. Apply the preprocessing pipeline
3. Extract keywords from the processed text
3. Store processed results
4. Log progress and any errors

## Preprocessing Pipeline

Current preprocessing steps:

1. **Spelling Correction** (`SpellingCorrector`)
    - Corrects spelling and grammar errors
    - Uses LanguageTool for Spanish

2. **Duplicate Removal** (`DuplicateRemover`)
    - Removes duplicate sentences
    - Preserves original sentence order

3. **Text Normalization** (`TextNormalizer`)
    - Removes extra whitespace
    - Removes stop words
    - Applies lemmatization
    - Uses spaCy's Spanish model

4. **Content Cleaning** (`ContentCleaner`)
    - Removes irrelevant phrases
    - Removes URLs and broken links

5. **Paragraph Segmentation** (`ParagraphSegmenter`)
    - Groups sentences into paragraphs
    - Improves readability

## Adding New Preprocessors

1. Create a new class in the `preprocessors` directory
2. Inherit from `TextPreprocessor`
3. Implement the `process` method
4. Add to the pipeline in `create_preprocessor_pipeline()`

Example:

```python
from .base import TextPreprocessor


class MyNewPreprocessor(TextPreprocessor):
    def process(self, text: str) -> str:
        # Your preprocessing logic here
        return processed_text
```

## Outside the pipeline

- **Keyword Generation**:
    - Identifies meaningful words
    - Uses  BERT model to produce embeddings for document and words.
    - Calculate the similarity between the document and the words to determine relevance.
    - Weights the values with a TF-IDF matrix to refine the results.

```python
from services.keyword_generator import KeywordGenerator

keygen = KeywordGenerator(model_path='path/to-bert-model')
keywords = keygen.generate_keywords(processed_text, top_n=5)
```

## Performance Considerations

- **Memory Usage**:
    - Batch processing limits memory consumption
    - LanguageTool and spaCy models are cached
    - Each batch is committed independently

- **Processing Speed**:
    - Parallel processing can be added for larger datasets
    - Batch size can be adjusted based on system capabilities

## Error Handling

- Each batch is processed independently
- Failed batches are logged and rolled back
- Processing continues with the next batch
- Detailed error logging for debugging
