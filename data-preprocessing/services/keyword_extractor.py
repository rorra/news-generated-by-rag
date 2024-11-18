# FILE: data-preprocessing/preprocessors/keyword_extractor.py

from transformers import pipeline, logging
from concurrent.futures import ThreadPoolExecutor
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict

logging.set_verbosity_error()

class KeywordExtractor():
    """
    A preprocessor for extracting keywords from text.
    """
    def __init__(self, model="dccuchile/bert-base-spanish-wwm-uncased"):
        print("Initializing KeywordExtractor...")
        self.nlp = spacy.load("es_core_news_sm")
        self.pipeline = pipeline("feature-extraction", model=model, tokenizer=model, max_length=512, truncation=True, device=0)

        self.word_embedding_cache = {}
        self.vectorizer = TfidfVectorizer()
        print("KeywordExtractor initialized")

    def preprocess_text(self, text):
        """
        Remove punctuation and convert to lowercase.
        """
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def normalize_word(self, word):
        """
        Lemmatize to group similar words.
        """
        doc = self.nlp(word)
        return doc[0].lemma_ if doc else word

    def group_similar_words(self, keywords_with_scores):
        """
        Group similar words and sum their scores.
        """
        grouped = defaultdict(lambda: {"score": 0.0, "examples": []})
        for word, score in keywords_with_scores:
            normalized_word = self.normalize_word(word)
            grouped[normalized_word]["score"] += score
            grouped[normalized_word]["examples"].append(word)
        grouped_sorted = sorted(
            [(examples["examples"][0], examples["score"]) for _, examples in grouped.items()],
            key=lambda x: x[1],
            reverse=True
        )
        return grouped_sorted

    def get_word_embeddings_batch(self, words):
        """
        Generate embeddings for a batch of words.
        """
        uncached_words = [word for word in words if word not in self.word_embedding_cache]
        if uncached_words:
            embeddings = self.pipeline(uncached_words)
            for word, embedding in zip(uncached_words, embeddings):
                self.word_embedding_cache[word] = np.mean(embedding, axis=0)
        return {word: self.word_embedding_cache[word] for word in words}
    
    def extract_keywords(self, title: str, text: str, top_k=5):
        text = self.preprocess_text(title + " " + text)
        
        chunks = [text[i:i+512] for i in range(0, len(text), 512)]
        embeddings = []
        for chunk in chunks:
            chunk_embeddings = self.pipeline(chunk)
            embeddings.extend(chunk_embeddings)
        
        cls_embedding = np.mean(np.array(embeddings[0]), axis=0)
        
        # Content words
        doc = self.nlp(text)
        # content_words = [(token.text, token.i) for token in doc if token.pos_ in {"NOUN", "ADJ"} and token.i < 510]
        content_words = [(token.text, token.i) for token in doc if token.pos_ in {"NOUN", "ADJ", "PROPN"}]
        
        
        word_embeddings = {}
        for word, i in content_words:
            try:
                word_embeddings[word] = np.mean(self.pipeline([word])[0], axis=0)
            except Exception as e:
                print(f"Error generating embedding for word '{word}': {e}")

        similarities = {
            word: cosine_similarity([embedding[0]], [cls_embedding])[0][0]
            for word, embedding in word_embeddings.items()
        }

        max_similarity = max(similarities.values()) if similarities else 1
        normalized_similarities = {word: score / max_similarity for word, score in similarities.items()}

        tfidf_matrix = self.vectorizer.fit_transform([text])
        tfidf_scores = dict(zip(self.vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0]))

        combined_scores = {
            word: normalized_similarities.get(word, 0) * tfidf_scores.get(word, 0)
            for word, _ in content_words
        }

        sorted_words = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        keywords_with_scores = [(word, combined_scores[word]) for word, _ in sorted_words[:top_k]]
        grouped_keywords = self.group_similar_words(keywords_with_scores)

        data_str = ''
        for item in grouped_keywords[:top_k]:
            w = item[0]
            s = str(item[1])
            data_str += f"({w},{s}),"
        return data_str[:-1]