import json
from sentence_transformers import SentenceTransformer
import os

# Función para cargar el modelo de embeddings basado en la preferencia
def load_embedding_model(preferred_model: str = 'stsb-xlm-r-multilingual'):
    """
    Carga el modelo de embeddings según la preferencia del usuario.

    Usar stsb-xlm-r para precisión y  sacrificar tiempo de procesamiento.
    Usar distiluse para tareas más ligeras o con limitaciones de recursos.
    
    Args:
        preferred_model (str): Nombre del modelo. Por defecto 'stsb-xlm-r-multilingual'.
    
    Returns:
        model: Modelo de SentenceTransformer cargado.
    """
    if preferred_model == 'distiluse':
        return SentenceTransformer('distiluse-base-multilingual-cased-v1')
    else:
        # Por defecto usa el modelo más preciso para español
        return SentenceTransformer('stsb-xlm-r-multilingual')
    


# Cargar el modelo preentrenado
model = load_embedding_model()

def get_embedding(text: str):
    """Genera el embedding para el texto preprocesado."""
    return model.encode(text)

def load_preprocessed_articles(file_path: str):
    """Carga el archivo JSON con las noticias preprocesadas."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def add_embeddings_to_articles(preprocessed_articles):
    """Añade embeddings a las noticias preprocesadas."""
    for title, article_data in preprocessed_articles.items():
        text = article_data['content']
        embedding = get_embedding(text)
        article_data['embedding'] = embedding.tolist()  # Convertir a lista para guardar en JSON
    return preprocessed_articles

def save_articles_with_embeddings(preprocessed_articles, output_file: str):
    """Guarda las noticias con embeddings en un nuevo archivo JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_articles, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Ruta del archivo de noticias preprocesadas
    input_file = os.path.join('data-preprocessing', 'preprocessed_files', '20241001_preprocessed_files.json')  # Modifica el nombre de acuerdo a tu fecha

    # Cargar noticias preprocesadas
    preprocessed_articles = load_preprocessed_articles(input_file)

    # Añadir embeddings a las noticias
    articles_with_embeddings = add_embeddings_to_articles(preprocessed_articles)

    # Guardar las noticias con embeddings
    output_file = os.path.join('data-preprocessing','preprocessed_files', 'noticias_con_embeddings.json')
    save_articles_with_embeddings(articles_with_embeddings, output_file)

    print(f"Embeddings generados y guardados en {output_file}")