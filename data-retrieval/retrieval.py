from qdrant_client import QdrantClient
from vectorize import get_embedding  # Importa la función de embeddings

client = QdrantClient(host="localhost", port=6333)

def search_news(query: str, top_k: int = 5):
    """Realiza una búsqueda de noticias basadas en la consulta."""
    # Genera el embedding de la consulta
    query_embedding = get_embedding(query)
    
    # Busca en Qdrant
    search_result = client.search(
        collection_name="news_collection",
        query_vector=query_embedding,
        limit=top_k
    )
    
    # Procesa los resultados
    articles = []
    for result in search_result:
        articles.append({
            "id": result.id,
            "title": result.payload["title"],
            "newspaper": result.payload["newspaper"],
            "section": result.payload["section"],
            "published_at": result.payload["published_at"],
            "score": result.score
        })
    return articles

if __name__ == "__main__":
    query = "política"
    results = search_news(query, top_k=3)
    for article in results:
        print(article)
