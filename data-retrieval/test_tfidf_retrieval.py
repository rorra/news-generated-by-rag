import unittest
from sklearn.feature_extraction.text import TfidfVectorizer

# Import the functions and variables you need to test
from tfidf_retrieval import search_tfidf
from utils import get_preprocessed_json_file

class TestSearchTfidf(unittest.TestCase):

    def setUp(self):
        # Prepare data and TF-IDF matrix
        date = "20241116"
        preprocessed_articles = get_preprocessed_json_file(date)

        self.documents = []
        self.titles = []
        self.contents = []
        for title, article_data in preprocessed_articles.items():
            self.documents.append((title, article_data["content"]))
        self.titles = [doc[0] for doc in self.documents]
        self.contents = [doc[1] for doc in self.documents]

        # Initialize the vectorizer and compute TF-IDF matrix
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.contents)

    def test_search_tfidf(self):
        # Test cases: List of tuples (query, expected_titles)
        test_cases = [
            ('Movimientos del dolar', ['🔴 Dólar blue y dólar hoy: a cuánto cotizan este viernes 15 de noviembre']),
            ('Dolar blue', ['🔴 Dólar blue y dólar hoy: a cuánto cotizan este viernes 15 de noviembre']),
            ('Eliminatorias del futbol', ['Así están las posiciones de las Eliminatorias tras el triunfo de Uruguay ante Colombia: todos los goles y el cronograma de la fecha 12']),
            ('pobreza en America', ['Cae la pobreza en América latina']),
            ('sustentabilidad', ['Compras de productos sustentables: un nicho que cobra relevancia en Argentina']),
            ('Taylor Swift', ['Taylor Swift no pasará el Día de Acción de Gracias con Travis Kelce ni su familia, aseguró la madre del jugador del Kansas City Chiefs']),
            ('corea del sur', ['Joe Biden elogió la cooperación con Corea del Sur y Japón en medio de la escalada de tensión con Pyongyang']),
            ('Mauricio Macri', ['Duro golpe a Mauricio Macri: la Justicia falló a favor de Chiqui Tapia y quedó firme su reelección como presidente de la AFA'])
        
        ]

        for query, expected_titles in test_cases:
            with self.subTest(query=query):
                results = search_tfidf(query, self.vectorizer, self.tfidf_matrix, self.titles)
                result_titles = [title for title, score in results]

                # Assert that the expected titles are in the results
                for expected_title in expected_titles:
                    print("\nExpected title: ")
                    print(expected_title)
                    print("\nResult titles: ")
                    print(result_titles)
                    self.assertIn(expected_title, result_titles)

if __name__ == '__main__':
    unittest.main()



