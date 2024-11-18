Here’s the updated `README` with steps for the vectorizer and things to keep in mind:

---

## News Preprocessor

This project is a Python script designed to preprocess news articles stored in a database. The script applies various cleaning and normalization functions to the content of the articles and saves the preprocessed data into a JSON file. It also supports adding embeddings using a vectorization step for later retrieval tasks.

### Prerequisites

- Python 3.8 or higher.
- Access to a SQLAlchemy-compatible database containing the `Article`, `Newspaper`, and `Section` tables.
- Install the necessary dependencies from `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

- Download NLTK data:
  ```python
  nltk.download('punkt')
  nltk.download('punkt_tab')
  ```

install spacy model
```bash
python -m spacy download es_core_news_sm
```

### Usage

1. **Update PYTHONPATH:**

   Before running the script, ensure that the project directory is in the `PYTHONPATH` environment variable. You can do this by running the following command in the project directory:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/../data-mining"
   ```

2. **Preprocess the news articles:**

   You can specify the publication date to run the script from a given date to the present. For example, to run it from `2024-01-01`:
   ```bash
   python main.py --date YYYY-MM-DD
   ```
   Or run without a date to process articles from the current day:
   ```bash
   python main.py
   ```

3. **Generate embeddings with the vectorizer:**

   After preprocessing, you can use the `vectorize.py` script to generate embeddings for the articles.

   - By default, the vectorizer uses the `stsb-xlm-r-multilingual` model, which is highly accurate for multilingual texts, including Spanish.
   - You can also use the `distiluse-base-multilingual-cased-v1` model for faster processing with lower resource usage, depending on your use case.

   **Steps:**
   1. Ensure that the `preprocessed_files` folder contains the JSON file generated from the preprocessing step (e.g., `20241001_preprocessed_files.json`).
   2. Run the vectorizer:
      ```bash
      python data-preprocessing/vectorize.py
      ```

   **Key Points:**
   - The vectorizer will generate embeddings for each article and save the output in a new JSON file (e.g., `noticias_con_embeddings.json`).
   - Choose the embedding model based on your context:
     - **For high accuracy in Spanish:** Use `stsb-xlm-r-multilingual`.
     - **For faster and resource-limited tasks:** Use `distiluse-base-multilingual-cased-v1`.

### Things to Keep in Mind:

- **Memory Usage:** For large datasets, ensure that your system has enough RAM, especially if using Java-based tools like `language_tool_python` for spelling correction. You may limit the number of articles processed or adjust memory settings in Java if needed.
- **Embedding Model Choice:** The model impacts performance and accuracy. Use `stsb-xlm-r-multilingual` for better semantic understanding or switch to `distiluse` for speed and efficiency.
