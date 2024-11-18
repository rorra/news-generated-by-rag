from transformers import pipeline, logging
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

nlp = spacy.load("es_core_news_sm")
logging.set_verbosity_error()

# remover puntuación y convertir a minúsculas
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Lematizar para agrupar palabras similares
def normalize_word(word):
    doc = nlp(word)
    return doc[0].lemma_ if doc else word

# Agrupar palabras similares y sumar sus puntajes
def group_similar_words(keywords_with_scores):
    grouped = defaultdict(lambda: {"score": 0.0, "examples": []})
    for word, score in keywords_with_scores:
        normalized_word = normalize_word(word)
        grouped[normalized_word]["score"] += score
        grouped[normalized_word]["examples"].append(word)
    grouped_sorted = sorted(
        [(examples["examples"][0], examples["score"]) for _, examples in grouped.items()],
        key=lambda x: x[1],
        reverse=True
    )
    return grouped_sorted

# Extraer palabras clave con BERT y TF-IDF
model_name = "dccuchile/bert-base-spanish-wwm-uncased"

embedding_pipeline = pipeline("feature-extraction", model=model_name, tokenizer=model_name, max_length=512, truncation=True, device=0)

def extract_keywords(text, num_keywords=5):
    text = preprocess_text(text)

    doc = nlp(text)

    # extraer sustantivos y adjetivos para mejorar la calidad de las palabras clave
    content_words = [token.text for token in doc if token.pos_ in {"NOUN", "ADJ"}]

    # Cargar modelo de BERT para extracción de características
    # model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Generrar embeddings para el texto
    embeddings = embedding_pipeline(text)
    aggregated_embeddings = np.mean(np.array(embeddings[0]), axis=0)

    # Generar embeddings para cada palabra en el texto
    word_embeddings = {}
    for word in set(content_words):
        try:
            word_embeddings[word] = np.mean(embedding_pipeline(word)[0], axis=0)
        except Exception as e:
            print(f"Error generating embedding for word '{word}': {e}")

    # Calcular similitud coseno entre cada palabra y el texto
    similarities = {
        word: cosine_similarity([embedding], [aggregated_embeddings])[0][0]
        for word, embedding in word_embeddings.items()
    }

    # Normalizar similitudes para que estén en el rango [0, 1]
    max_similarity = max(similarities.values()) if similarities else 1
    normalized_similarities = {word: score / max_similarity for word, score in similarities.items()}

    # Calcular puntajes TF-IDF para cada palabra para ponderar la importancia
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text])
    tfidf_scores = dict(zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

    # Combinar similitudes y puntajes TF-IDF para obtener puntajes con peso
    combined_scores = {
        word: normalized_similarities.get(word, 0) * tfidf_scores.get(word, 0)
        for word in content_words
    }

    # ordenar, extraer y agrupar palabras clave
    sorted_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    keywords_with_scores = [(word, combined_scores[word]) for word, _ in sorted_words[:num_keywords]]
    grouped_keywords = group_similar_words(keywords_with_scores)


    return grouped_keywords[:num_keywords]

# Ejemplo
if __name__ == "__main__":
    text = """mes Facundo Font planear novia vida preparar convivir decidir desprender él cosa liberar espacio juntar dinero ir necesitar mudanza contexto 11 junio vida cambiar usuario decidir venta PlayStation usar plataforma venta lejos encontrar solución problema encontrar inconveniente Lee Alerta peligroso estafa llegar mails AFIP pedir revalidar dato robar información hombre pasar comprador engañar conseguir foto DNI momento utilizar identidad cometer delito 40 persona estafar Facundo Font realidad estafó publicación pareja venta plataforma necesitar tirar pileta hora hacer publicación contactar persona diciéndome interesar Play 5 habir publicar 5 4 preguntar querer pasar número sobrino interesado pasar llamar WhatsApp detallar chico rápidamente percibir persona edad 28 años entender producto empezar pregunta técnico mensaje estafador Foto gentileza Facundo Font viernes re interesar pedir guarde bajar publicación miedo venda decir hacer asegurar ir comprar finalmente bajar publicación joven periodista situación presunto comprador volver escribir primita querer PlayStation venta ir comprar lunes atareado decir venir horario ir mandar auto parecer raro decir dejar martes martes comprador volver poder acercar papá prestar auto motivo ir mandar auto retirar ambos consola juego empezar parecer raro plata seguridad ir estafar bajar hall entrada edificio ver auto poner baliza mandar captura transferencia plata llegar banco parecer extraño acreditar sumado comprobante decir poder llegar acreditar 24 hora víctima momento situación comenzar volver él incómodo empezar irascible tratar jarca decir transferencia padre ir enterar ir venir buscar momento decir poder confiar momento confiar vos pasamir foto DNI Facu parecer razonable poder quedar tranquilo pedir foto presunto interesado consola enviar foto persona sostener documento Facundo conversación tener ver bloqueó borrar mensaje banco existir momento sentir alivio haber robar Play joven semana hablar persona conocer colegio vo sos Facundo Font vo sos jarca jugar gente trabajar rememorar chico entender caer momento novia foto DNI caer pensar tipo usar dato cagar efectivamente estafado creer Facundo acercar comisaría explicar suceder principal problema principio querer tomar denuncia usurpación identidad decir prueba suficiente 10 aparecer 10 persona estafada mediados noviembre mes víctima ascender 40 recién permitir efectuar denuncia respaldo joven mensaje víctima Foto gentileza Facundo Font pese Justicia causa mano chico continuar vivir tortura transcurso escribir Instagram novia familiar contactar Facebook venir casa detallar empezar caer realidad miedo persona foto DNI dirección miedo vengar buscarme tono violento mayoría estafado contactar terminar creer ayudar dato número WhatsApp estafador captura pantalla brindar fiscalía manera tomar robo acabar atravesar Facundo trabajo exhaustivo crear situación tensión vivir diario generar problema psicológico emocional llegar notificación ver él miedo estafado tocar portero casa generar palpitación poner 28 año vivo Parque Patricios caminer tranquilo barrio camino mirar lado alguien quedar viéndomir segundo fijo pensar buscar lamentar joven miedo miedo bajar persona auto puerta casa buscar tener pegar cartel casa venir estafa bajar cambio bajar explicar precisar recurso implementar vivir miedo pareja manejar sostener denuncia avanzar Facundo contacto persona pasar robar él asegurar pese atado mano fiscalía causa ir pasar nivel nacional cuestión burocrático proceso investigación ubicar él tipo caso hablar mil gente entender cantidad tecnología rastrear esperar damnificado declarar seguir completar Instagram posteo fijado suceder diverso recorte programa radio participar llegar poder entender persona honesta estafó mensaje llegar foto gentileza Facundo Font char él estafado atípico pararte hab él alguien sacar plata pedir él disculpa sentirte culpable situación incómodo entender entender yo acotar joven aguardo asignir juzgado causa 40 estafado contacto volver escribir avance llevar decepción respuesta negativo Policía poder tomar denuncia hacer nombre real denunciar estafador lograr anexar  reclamo Facu conseguir delito informático bache legal pandemia aumentar muchísimo caso solución sentir pillo mundo decir ir cagar pasar boludo gente vivir cosa paso seguir funcionar estafar producto identidad confiar previsto pasar sentido estancamiento caso número estafador gana hablar él gano hablar él confiar desear Justicia hacer número único forma rastrear él hablar bloquear pierdo lamentar afectar vida labró pareja insistir chico padecer cuidado injusto intenta fingir demencia seguir sufrí llorir montón tener crisis nervio hablar gano terminar enojar sentido pensar único forma parir cansar pasar tanto año foto DNI coincido cara lamentar Facundo advertencia utilizar plataforma venta pasar foto DNI tarjeta pedir mamá internet raro confiar completar persona común pasar situación horrible deseo volver anónimo vivir tranquilo"""

    keywords = extract_keywords(text, num_keywords=10)
    print("Extracted Keywords with Scores:", keywords)
