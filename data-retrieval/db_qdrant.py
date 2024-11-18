from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

def create_index():
    """Crea un índice para almacenar embeddings en Qdrant."""
    client.recreate_collection(
        collection_name="news_collection",
        vectors_config=VectorParams(
            size=384,  # Tamaño del embedding (depende del modelo que uses)
            distance=Distance.COSINE  # Usamos distancia coseno para la búsqueda
        )
    )
