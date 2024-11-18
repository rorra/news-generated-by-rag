from qdrant_client import QdrantClient
import json

client = QdrantClient(host="localhost", port=6333)

def insert_articles(json_file):
    """Inserta las noticias preprocesadas y vectorizadas en Qdrant."""
    with open(json_file, 'r') as f:
        preprocessed_articles = json.load(f)

    points = []
    for article_id, article_data in enumerate(preprocessed_articles.values()):
        embedding = article_data['embedding']
        point = {
            "id": article_id,
            "vector": embedding,
            "payload": {
                "title": article_data['title'],
                "newspaper": article_data['newspaper'],
                "section": article_data['section'],
                "published_at": article_data['published_at'],
                "keywords": article_data['keywords']
            }
        }
        points.append(point)

    # Inserta los puntos en el índice de Qdrant
    client.upsert(collection_name="news_collection", points=points)

if __name__ == "__main__":
    insert_articles('noticias_con_embeddings.json')
