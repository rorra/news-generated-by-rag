import pandas as pd
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from config import SessionLocal
from models.db_models import ProcessedArticle


load_dotenv()
client = OpenAI()


def get_news() -> dict:
    """Retrieve all processed news from the database."""
    news = {}
    db = SessionLocal()
    try:
        processed_articles = db.query(ProcessedArticle).all()
        for article in processed_articles:
            news[article.processed_title] = article.processed_content
        
    except Exception as e:
        print(f"Error retrieving news: {e}")
        raise
    finally:
        db.close()

    return news


def generate_query_to_retrieve_new(new: str, max_tries: int = 3) -> List[str]:
    """Generate a query to retrieve a given news."""

    if max_tries != 0:
        prompt = f"""
                Given this new in Spanish. Generate 5 question in human language to retrieve it in Spanihs. Don't add 1. or 2. etc. Just the questions.

                Input example Spanish new:

                    23 septiembre 2024 gobierno javier milei cerrar agosto 2024 superávit fiscal $ 899660 millón informe centro economía político argentino cepa saldo positivo cuenta público reflejar panorama complejo 99,6% superávit destinar pago interés deuda resultado neto $ 3531 millón país equilibrio fiscal pilar política económico gobierno calle ajuste palpable 50% pobreza semestre fondo destinar completo cubrir compromiso deuda $ 899660 millón superávits $ 896130 millón usar pagar interés dejar margen necesidad fiscal excluir pasivo interés letra tesoro leer seco saldo real deficitario señal superávit ajuste gasto público principal razón equilibrio agosto gasto público reducir $ 7.8 billón caída 23,7% término real ajustado inflación comparación mes 2023 fuerte contracción impactar drástico área clave educación reducción 92% vivienda caída 92% transporte -73% transferencia provincia -41% universidades -31% ver gravemente afectado leer dato caber industria pyme ver caer 8,7% interanual agosto ajuste generalizado área lograr escapar tijera gasto público subsidio transporte fondo destinado beneficiario asignación universal hijo rubro experimentar incremento aumento compensar drástico reducción partida dramático caída ingreso informe cepa señalar caída preocupante ingreso sector público nacional spn agosto alcanzar $ 8.7 billón representar disminución 13,9% interanual ajustado inflación caída significativo año reflejar deterioro recaudación área clave principal componente merma baja aporte contribución sector privado público seguridad social desplomar 71% término interanual caída importante incluir transferencia corriente -56% derecho importación -33% recaudación bien personal -29% leer gira economista reunir javier milei york contraste tendencia general derecho exportación único ítem mostrar incremento recaudación profundo caída junio -43.7% apartado mostrar signo recuperación agosto crecer 24% término interanual repunte positivo lograr contrarrestar magnitud caída rubro ingreso

                Output example Spanish questions:

                    ¿Cuál es el superávit fiscal de agosto de 2024?, 
                    ¿Cuál es el saldo real deficitario?, 
                    ¿Qué área clave experimentó una reducción del 92%?, 
                    ¿Qué área logró escapar de la tijera del gasto público?,
                    ¿Qué rubro experimentó un incremento en agosto de 2024?

                input: {new}

                output:

                """
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates questions, related with news in order to test a retrieval system."},
                {"role": "user", "content": prompt},
            ]
        )

        questions = response.choices[0].message.content.split("\n")

        if len(questions) == 5:
            return questions
        else:
            return generate_query_to_retrieve_new(new, max_tries - 1)
    else:
        return []


def build_dataset() -> list:
    news = get_news()
    dataset = []
    for new in news:
        questions = generate_query_to_retrieve_new(news[new])
        if questions:
            dataset.append((new, questions))
    return dataset


def save_csv_dataset(dataset: list):
    df = pd.DataFrame(dataset, columns=['new', 'questions'])
    df.to_csv('test_dataset.csv', index=False, encoding='utf-8')
    dataset = pd.read_csv('test_dataset.csv', encoding='utf-8')
    print(dataset.head())
    print(dataset.shape)


def main():
    dataset = build_dataset()
    save_csv_dataset(dataset)
    
    
if __name__ == "__main__":
    main()