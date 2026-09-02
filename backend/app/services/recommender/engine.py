"""
Neural Hybrid Recommendation Orchestrator Engine Module
"""

import json
from typing import List, Dict, Any

import numpy as np

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from app.core.config import PRODUCTS_FILE_PATH, MODEL_NAME, MMR_LAMBDA
from app.services.recommender.preprocessing import (
    normalize_occasion,
    build_product_text,
    is_low_quality,
)
from app.services.recommender.indexers import LexicalIndexManager
from app.services.vector_db import get_vector_db
from app.services.recommender.re_ranker import apply_mmr_diversity_reranking


class ProductRecommender:
    def __init__(self, products_path: str = str(PRODUCTS_FILE_PATH), model_name: str = MODEL_NAME):
        self.products_path = products_path
        self.model_name = model_name
        self.products: List[Dict[str, Any]] = []
        self.corpus: List[str] = []
        self.known_occasions: List[str] = []

        # Plug-and-play Vector DB & Lexical Indexer
        self.vector_db = get_vector_db(model_name=model_name)
        self.lexical_indexer = LexicalIndexManager()
        self._load_and_index()

    def _load_and_index(self):
        with open(self.products_path, "r", encoding="utf-8") as f:
            all_products: List[Dict[str, Any]] = json.load(f)

        self.products = [
            p for p in all_products
            if p.get("status") in ("ACTIVE", "REVIEW_REQUIRED")
            and not p.get("deletedAt")
            and not is_low_quality(p)
        ]

        self.corpus = [build_product_text(p) for p in self.products]
        print(f"[ProductRecommender] Loaded {len(self.products)} products for indexing (shortDescription EXCLUDED).")

        self.known_occasions = self._harvest_occasions()
        print(f"[ProductRecommender] Dynamically harvested {len(self.known_occasions)} occasion themes.")

        # Build Vector DB & Lexical Indexes
        self.vector_db.build_index(self.corpus)
        self.lexical_indexer.build_index(self.corpus)

    def _harvest_occasions(self) -> List[str]:
        occasions = set()
        common_themes = [
            "birthday", "wedding", "diwali", "valentine", "anniversary",
            "christmas", "holi", "baby shower", "graduation", "mothers day",
            "fathers day", "eid", "raksha bandhan", "new year", "housewarming"
        ]
        for p in self.products:
            text = (p.get("name", "") + " " + p.get("slug", "")).lower()
            for theme in common_themes:
                if theme in text:
                    occasions.add(theme)
        return list(occasions)

    def _expand_occasion_fuzzy(self, occasion: str) -> str:
        normalized = normalize_occasion(occasion)
        if not HAS_RAPIDFUZZ or not self.known_occasions:
            return normalized

        match = process.extractOne(normalized, self.known_occasions, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 70:
            matched_occasion = match[0]
            return f"{normalized} {matched_occasion}"
        
        return normalized

    def recommend(self, occasion: str, limit: int = 12, lambda_mult: float = MMR_LAMBDA) -> List[Dict[str, Any]]:
        if not self.products:
            return []

        n_products = len(self.products)
        normalized_query = normalize_occasion(occasion)
        query_tokens = set(normalized_query.split())

        # 1. Vector DB Dense Similarity Search
        dense_scores = self.vector_db.search(occasion, n_products)

        # 2. TF-IDF Lexical Similarity with RapidFuzz expansion
        expanded_str = self._expand_occasion_fuzzy(occasion)
        tfidf_scores = self.lexical_indexer.search_lexical(expanded_str)
        if len(tfidf_scores) == 0:
            tfidf_scores = np.zeros(n_products, dtype=np.float32)

        # 3. Title Match Boost
        title_boosts = np.zeros(n_products, dtype=np.float32)
        for i, p in enumerate(self.products):
            title_text = p.get("name", "").lower()
            if normalized_query in title_text:
                title_boosts[i] = 1.0
            else:
                if HAS_RAPIDFUZZ:
                    fz_score = fuzz.partial_ratio(normalized_query, title_text) / 100.0
                    if fz_score > 0.7:
                        title_boosts[i] = 0.5 * fz_score
                else:
                    matches = sum(1 for t in query_tokens if t in title_text)
                    if matches > 0:
                        title_boosts[i] = 0.5 * (matches / len(query_tokens))

        # 4. Keyword Boost
        keyword_boosts = np.array([
            sum(1 for t in query_tokens if t in p.get("name", "").lower()) / (len(query_tokens) or 1)
            for p in self.products
        ], dtype=np.float32)

        # 5. Mismatch Penalty
        mismatch_penalties = np.zeros(n_products, dtype=np.float32)
        for i, p in enumerate(self.products):
            title_lower = p.get("name", "").lower()
            for other_occ in self.known_occasions:
                if other_occ not in normalized_query and other_occ in title_lower:
                    mismatch_penalties[i] = 0.15
                    break

        # Pad all score arrays to n_products length (safety)
        def _pad(arr, n):
            if len(arr) < n:
                padded = np.zeros(n, dtype=np.float32)
                padded[:len(arr)] = arr
                return padded
            return arr[:n].astype(np.float32)

        dense_scores       = _pad(dense_scores, n_products)
        tfidf_scores       = _pad(tfidf_scores, n_products)
        title_boosts       = _pad(title_boosts, n_products)
        keyword_boosts     = _pad(keyword_boosts, n_products)
        mismatch_penalties = _pad(mismatch_penalties, n_products)

        # 6. Ensemble Hybrid Scoring
        if getattr(self.vector_db, "is_ready", False) and float(dense_scores.max()) > 0:
            rel_scores = (
                (0.55 * dense_scores) +
                (0.20 * tfidf_scores) +
                (0.15 * title_boosts) +
                (0.10 * keyword_boosts) -
                mismatch_penalties
            )
        else:
            rel_scores = (
                (0.60 * tfidf_scores) +
                (0.25 * title_boosts) +
                (0.15 * keyword_boosts) -
                mismatch_penalties
            )

        # Always return top-N — no hard cutoff (dataset products use generic occasion language)
        candidate_indices = list(np.argsort(rel_scores)[::-1][:limit])
        if not candidate_indices:
            return []

        # 7. MMR & RapidFuzz Category Diversity Re-ranking Step
        dense_embeddings = self.vector_db.get_dense_embeddings()
        selected_indices = apply_mmr_diversity_reranking(
            products=self.products,
            candidate_indices=candidate_indices,
            rel_scores=rel_scores,
            dense_embeddings=dense_embeddings,
            limit=limit,
            lambda_mult=lambda_mult,
        )

        results = []
        for idx in selected_indices:
            score = float(rel_scores[idx])
            product = dict(self.products[idx])
            product["relevance_score"] = round(score, 4)
            product["dense_score"] = round(float(dense_scores[idx]), 4) if getattr(self.vector_db, "is_ready", False) else 0.0
            product["lexical_score"] = round(float(tfidf_scores[idx]), 4)
            product["keyword_boost"] = round(float(keyword_boosts[idx]), 4)
            product["title_boost"] = round(float(title_boosts[idx]), 4)
            product["mismatch_penalty"] = round(float(mismatch_penalties[idx]), 4)
            results.append(product)

        return results

    def get_all_products(self, limit: int = 15) -> List[Dict[str, Any]]:
        return self.products[:limit]

    def add_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamically adds a new product to the catalog:
        1. Appends product to self.products and persists to products.json
        2. Builds product text document
        3. Encodes new dense vector and upserts to Vector DB (Qdrant/FAISS)
        4. Re-indexes lexical TF-IDF matrix
        """
        import uuid
        if "id" not in product_data or not product_data["id"]:
            product_data["id"] = f"prod_{uuid.uuid4().hex[:8]}"
        
        if "status" not in product_data:
            product_data["status"] = "ACTIVE"
            
        self.products.append(product_data)
        doc_id = len(self.products) - 1

        # Persist to disk
        try:
            with open(self.products_path, "w", encoding="utf-8") as f:
                json.dump(self.products, f, indent=2)
            print(f"[ProductRecommender] Saved new product '{product_data.get('name')}' to {self.products_path}")
        except Exception as e:
            print(f"[ProductRecommender] Error persisting product to disk: {e}")

        # Build text & insert vector
        new_text = build_product_text(product_data)
        self.corpus.append(new_text)

        self.vector_db.add_product_vector(text=new_text, doc_id=doc_id)
        self.lexical_indexer.reindex_corpus(self.corpus)

        # Harvest occasion themes if any
        name_text = (product_data.get("name", "") + " " + product_data.get("slug", "")).lower()
        for theme in ["birthday", "wedding", "diwali", "valentine", "anniversary"]:
            if theme in name_text and theme not in self.known_occasions:
                self.known_occasions.append(theme)

        return product_data
