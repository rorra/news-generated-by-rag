import argparse
from datetime import datetime
from typing import List, Tuple
from gensim.models import KeyedVectors
from gensim.downloader import load
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_preprocessed_json_file
import numpy as np

def compute_document_embedding(document: str) -> np.ndarray:
    """
    Compute the embedding for a document by averaging word embeddings.
    
    Args:
        document (str): The input document text.
    Returns:
        np.ndarray: The averaged Word2Vec embedding for the document.
    """
    words = document.split()
    word_embeddings = [model[word] for word in words if word in model]

    if not word_embeddings:
        return np.zeros(model.vector_size)

    return np.mean(word_embeddings, axis=0)


def search_word2vec(query: str, top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Search documents in the corpus using Word2Vec embeddings and cosine similarity.
    
    Args:
        query (str): The search query.
        top_n (int): Number of top relevant results to return.
    Returns:
        List[Tuple[str, float]]: List of document titles and their similarity scores.
    """
    # Compute the query embedding
    query_embedding = compute_document_embedding(query)

    # Compute cosine similarity between the query and all document embeddings
    similarities = cosine_similarity([query_embedding], document_embeddings).flatten()

    # Sort results by similarity in descending order
    top_indices = similarities.argsort()[::-1][:top_n]

    # Return the top titles and their similarity scores
    results = [(titles[i], similarities[i]) for i in top_indices]
    return results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Search documents using Word2Vec.')
    parser.add_argument('--date', type=str, required=True, help='Publish date in YYYYMMDD format (e.g., 20241116)')
    parser.add_argument('--query', type=str, required=True, help='Query to be done')

    args = parser.parse_args()
    query = args.query

    # Validate and convert the input date string to ensure it matches the YYYYMMDD format
    try:
        if args.date:
            # Validate that the date matches the YYYYMMDD format
            publish_date = datetime.strptime(args.date, '%Y%m%d')
        else:
            # Use the current date in YYYYMMDD format
            publish_date = datetime.now().strftime('%Y%m%d')
    except ValueError:
        print("Incorrect date format. Please use YYYYMMDD (e.g., 20241116).")
        exit(1)

    # Use the date for file retrieval
    date = publish_date if isinstance(publish_date, str) else publish_date.strftime('%Y%m%d')
    try:
        preprocessed_articles = get_preprocessed_json_file(date)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Initialize containers for documents
    documents = []
    titles = []
    contents = []

    # Extract titles and content from the loaded data
    for title, article_data in preprocessed_articles.items():
        documents.append((title, article_data["content"]))

    titles = [doc[0] for doc in documents]
    contents = [doc[1] for doc in documents]

    # Load pre-trained Word2Vec model from gensim-data
    print("Loading pre-trained Word2Vec model from gensim-data...")
    model = load("word2vec-google-news-300")
    print("Pre-trained Word2Vec model loaded successfully.")

    # Generate embeddings for all documents
    document_embeddings = np.array([compute_document_embedding(content) for content in contents])

    # Perform the search
    results = search_word2vec(query)

    # Display the search results
    print(f"\nQuery: {query}")
    print("\nRelevant results:")
    for title, score in results:
        print(f"Title: {title} - Similarity: {score:.4f}")