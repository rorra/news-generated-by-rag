import argparse
from datetime import datetime
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import get_preprocessed_json_file


def initialize_tfidf(corpus: List[str]) -> Tuple[TfidfVectorizer, 'scipy.sparse.csr.csr_matrix']:
    """
    Initializes the TF-IDF vectorizer and generates the TF-IDF matrix for the corpus.
    
    Args:
        corpus (List[str]): List of document contents.
    
    Returns:
        Tuple[TfidfVectorizer, csr_matrix]: Fitted TF-IDF vectorizer and the TF-IDF matrix.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix


def search_tfidf(query: str, vectorizer: TfidfVectorizer, tfidf_matrix, titles: List[str], top_n: int = 5) -> List[Tuple[str, float]]:
    """
    Searches documents in the corpus using TF-IDF and cosine similarity.
    
    Args:
        query (str): The search query.
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        tfidf_matrix (csr_matrix): TF-IDF matrix of the corpus.
        titles (List[str]): List of document titles.
        top_n (int): Number of top relevant results to return.
    
    Returns:
        List[Tuple[str, float]]: List of document titles and their similarity scores.
    """
    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and the corpus
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Sort results by similarity in descending order
    top_indices = cosine_similarities.argsort()[::-1][:top_n]
    results = [(titles[i], cosine_similarities[i]) for i in top_indices]
    return results


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess news articles.')
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

    # Generate the TF-IDF matrix for the corpus
    vectorizer, tfidf_matrix = initialize_tfidf(contents)

    results = search_tfidf(query, vectorizer, tfidf_matrix, titles)

    # Display the search results
    print(f"\nQuery: {query}")
    if results:
        print("\nRelevant results:")
        for title, score in results:
            print(f"Title: {title} - Similarity coissine: {score:.4f}")
    else:
        print("No relevant articles found.")
