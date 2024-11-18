import json
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings preentrenado
model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embedding(text: str):
    """Genera el embedding para el texto de entrada."""
    return model.encode(text)


def load_preprocessed_articles(file_path: str):
    """Carga el archivo JSON con las noticias preprocesadas."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_embeddings_to_articles(preprocessed_articles):
    """Añade embeddings a las noticias preprocesadas."""
    for title, article_data in preprocessed_articles.items():
        text = article_data['content']  # Usa el texto preprocesado
        embedding = get_embedding(text)  # Genera el embedding
        article_data['embedding'] = embedding.tolist()  # Añade el embedding como lista
    return preprocessed_articles


def save_articles_with_embeddings(preprocessed_articles, output_file: str):
    """Guarda las noticias con embeddings en un nuevo archivo JSON."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(preprocessed_articles, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    # Ruta del archivo JSON con las noticias preprocesadas
    input_file = 'ruta_al_json_preprocesado.json'

    # Cargar las noticias preprocesadas
    preprocessed_articles = load_preprocessed_articles(input_file)

    # Añadir embeddings
    articles_with_embeddings = add_embeddings_to_articles(preprocessed_articles)

    # Guardar el resultado en un nuevo archivo
    output_file = 'noticias_con_embeddings.json'
    save_articles_with_embeddings(articles_with_embeddings, output_file)

    print(f"Embeddings generados y guardados en {output_file}")
