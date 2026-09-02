"""
TF-IDF Search Engine Indexer Module
"""

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LexicalIndexManager:
    """Manages TF-IDF Sparse Matrix Indexing."""

    def __init__(self):
        self.vectorizer: TfidfVectorizer | None = None
        self.tfidf_matrix = None

    def build_index(self, corpus: List[str]):
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=10_000,
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print("[LexicalIndexManager] TF-IDF lexical index built.")

    def search_lexical(self, expanded_query_str: str) -> np.ndarray:
        if not self.vectorizer or self.tfidf_matrix is None:
            return np.array([])
        query_vec = self.vectorizer.transform([expanded_query_str])
        return cosine_similarity(query_vec, self.tfidf_matrix).flatten().astype(np.float32)

    def reindex_corpus(self, corpus: List[str]):
        """Re-fits TF-IDF vectorizer when catalog corpus updates."""
        self.build_index(corpus)
