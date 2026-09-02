"""
Vector Database Abstraction Layer (Pluggable FAISS & Qdrant Engine)
-------------------------------------------------------------------
Provides a unified, pluggable interface (`BaseVectorDB`) for vector storage and retrieval.
Supports:
  1. `FAISSVectorDB`: High-performance local memory FAISS IndexFlatIP index.
  2. `QdrantVectorDB`: Qdrant vector database (In-Memory or Remote Server mode).

To switch providers, change `VECTOR_DB_TYPE` in `config.py` or set environment variable `VECTOR_DB_TYPE=qdrant`.
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from app.core.config import MODEL_NAME, VECTOR_DB_TYPE, QDRANT_CLUSTER_ENDPOINT, QDRANT_API_KEY

# Prevent transformers from importing broken global tensorflow package
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# 1. Check SentenceTransformers & FAISS availability
HAS_NEURAL = False
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    HAS_NEURAL = True
except Exception:
    HAS_NEURAL = False

# 2. Check Qdrant Client availability
HAS_QDRANT = False
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    HAS_QDRANT = True
except ImportError:
    HAS_QDRANT = False


class BaseVectorDB(ABC):
    """Abstract Base Class defining the vector database contract."""

    @abstractmethod
    def build_index(self, corpus: List[str]):
        """Encode text corpus into embeddings and index them."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int) -> np.ndarray:
        """Search for top_k most similar items to query vector. Returns array of similarity scores."""
        pass

    @abstractmethod
    def get_dense_embeddings(self) -> np.ndarray | None:
        """Retrieve stored L2-normalized dense embedding matrix for MMR re-ranking."""
        pass

    @abstractmethod
    def add_product_vector(self, text: str, doc_id: int):
        """Dynamically encode new product text and insert vector embedding into index."""
        pass


class FAISSVectorDB(BaseVectorDB):
    """
    FAISS + SentenceTransformers implementation of BaseVectorDB.
    Uses IndexFlatIP with L2 normalized vectors for Cosine Similarity.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self.encoder: SentenceTransformer | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.dense_embeddings: np.ndarray | None = None
        self.is_ready: bool = False

    def build_index(self, corpus: List[str]):
        if not HAS_NEURAL or not corpus:
            print("[VectorDB:FAISS] Neural dependencies unavailable or corpus empty. Skipping dense index.")
            return

        try:
            print(f"[VectorDB:FAISS] Loading embedding model '{self.model_name}'...")
            self.encoder = SentenceTransformer(self.model_name)

            print(f"[VectorDB:FAISS] Encoding {len(corpus)} documents into dense vectors...")
            raw_embeddings = self.encoder.encode(corpus, show_progress_bar=False, convert_to_numpy=True)

            faiss.normalize_L2(raw_embeddings)
            self.dense_embeddings = raw_embeddings.astype(np.float32)

            dimension = self.dense_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.dense_embeddings)
            self.is_ready = True
            print(f"[VectorDB:FAISS] Successfully indexed {self.index.ntotal} vectors in FAISS IndexFlatIP (dim={dimension}).")
        except Exception as e:
            print(f"[VectorDB:FAISS] Error building FAISS index: {e}")
            self.is_ready = False

    def search(self, query: str, top_k: int) -> np.ndarray:
        if not self.is_ready or not self.encoder or not self.index:
            return np.zeros(top_k, dtype=np.float32)

        query_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)

        scores = np.zeros(top_k, dtype=np.float32)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < top_k:
                scores[idx] = float(dist)
        return scores

    def get_dense_embeddings(self) -> np.ndarray | None:
        return self.dense_embeddings

    def add_product_vector(self, text: str, doc_id: int):
        if not self.encoder or not self.index:
            return
        new_emb = self.encoder.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(new_emb)
        self.index.add(new_emb)
        if self.dense_embeddings is not None:
            self.dense_embeddings = np.vstack([self.dense_embeddings, new_emb])
        else:
            self.dense_embeddings = new_emb
        self.is_ready = True
        print(f"[VectorDB:FAISS] Dynamic vector inserted for doc_id {doc_id}.")


class QdrantVectorDB(BaseVectorDB):
    """
    Qdrant Vector Database implementation of BaseVectorDB.
    Supports both remote Qdrant Cloud cluster (via endpoint + API key) and local in-memory mode.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        url: str = QDRANT_CLUSTER_ENDPOINT,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = "products",
    ):
        self.model_name = model_name
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client: Optional[QdrantClient] = None
        self.encoder: Optional[SentenceTransformer] = None
        self.dense_embeddings: np.ndarray | None = None
        self.is_ready: bool = False

    def build_index(self, corpus: List[str]):
        if not HAS_QDRANT:
            print("[VectorDB:Qdrant] qdrant-client not installed. Falling back...")
            return
        if not HAS_NEURAL or not corpus:
            print("[VectorDB:Qdrant] Neural dependencies unavailable or corpus empty.")
            return

        try:
            if self.url and self.api_key:
                print(f"[VectorDB:Qdrant] Connecting to Remote Qdrant Cloud Cluster at '{self.url}'...")
                self.client = QdrantClient(url=self.url, api_key=self.api_key, timeout=30.0)
            elif self.url:
                print(f"[VectorDB:Qdrant] Connecting to Remote Qdrant Server at '{self.url}'...")
                self.client = QdrantClient(url=self.url, timeout=30.0)
            else:
                print(f"[VectorDB:Qdrant] Initializing Local In-Memory Qdrant Client...")
                self.client = QdrantClient(location=":memory:")

            print(f"[VectorDB:Qdrant] Loading embedding model '{self.model_name}'...")
            self.encoder = SentenceTransformer(self.model_name)

            # Check if collection already exists on Qdrant Cloud to avoid redundant re-encoding & cloud uploads
            if self.client.collection_exists(collection_name=self.collection_name):
                info = self.client.get_collection(collection_name=self.collection_name)
                if info.points_count >= len(corpus):
                    print(f"[VectorDB:Qdrant] Collection '{self.collection_name}' exists on Qdrant Cloud with {info.points_count} points. Re-using cloud embeddings!")
                    # Cache local dense matrix for MMR re-ranking
                    raw_embeddings = self.encoder.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
                    faiss.normalize_L2(raw_embeddings)
                    self.dense_embeddings = raw_embeddings.astype(np.float32)
                    self.is_ready = True
                    return

            print(f"[VectorDB:Qdrant] Encoding {len(corpus)} documents for initial upload...")
            raw_embeddings = self.encoder.encode(corpus, show_progress_bar=False, convert_to_numpy=True)
            faiss.normalize_L2(raw_embeddings)
            self.dense_embeddings = raw_embeddings.astype(np.float32)

            vector_size = self.dense_embeddings.shape[1]

            # Re-create collection in Qdrant
            if self.client.collection_exists(collection_name=self.collection_name):
                self.client.delete_collection(collection_name=self.collection_name)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

            # Insert vector points in small batches to avoid network timeout
            points = [
                PointStruct(
                    id=idx,
                    vector=vector.tolist(),
                    payload={"doc_id": idx, "text_snippet": corpus[idx][:100]}
                )
                for idx, vector in enumerate(self.dense_embeddings)
            ]
            
            batch_size = 50
            for b_idx in range(0, len(points), batch_size):
                batch_points = points[b_idx : b_idx + batch_size]
                self.client.upsert(collection_name=self.collection_name, points=batch_points)

            self.is_ready = True
            print(f"[VectorDB:Qdrant] Successfully indexed {len(points)} vectors into Remote Qdrant collection '{self.collection_name}'.")
        except Exception as e:
            print(f"[VectorDB:Qdrant] Error building Qdrant index: {e}")
            self.is_ready = False

    def search(self, query: str, top_k: int) -> np.ndarray:
        if not self.is_ready or not self.client or not self.encoder:
            total_docs = len(self.dense_embeddings) if self.dense_embeddings is not None else 100
            return np.zeros(total_docs, dtype=np.float32)

        total_docs = len(self.dense_embeddings) if self.dense_embeddings is not None else 100

        query_emb = self.encoder.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)
        query_vector = query_emb[0].tolist()

        # Query Qdrant (compatible with qdrant-client 1.19+)
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )

        scores = np.zeros(total_docs, dtype=np.float32)
        for hit in search_result.points:
            doc_id = hit.payload.get("doc_id", hit.id)
            if 0 <= doc_id < total_docs:
                scores[doc_id] = float(hit.score)
        return scores

    def get_dense_embeddings(self) -> np.ndarray | None:
        return self.dense_embeddings

    def add_product_vector(self, text: str, doc_id: int):
        if not self.client or not self.encoder:
            return
        new_emb = self.encoder.encode([text], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(new_emb)
        query_vec = new_emb[0].tolist()

        point = PointStruct(
            id=doc_id,
            vector=query_vec,
            payload={"doc_id": doc_id, "text_snippet": text[:100]}
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])

        if self.dense_embeddings is not None:
            self.dense_embeddings = np.vstack([self.dense_embeddings, new_emb])
        else:
            self.dense_embeddings = new_emb
        self.is_ready = True
        print(f"[VectorDB:Qdrant] Dynamic vector inserted into collection '{self.collection_name}' for doc_id {doc_id}.")


def get_vector_db(model_name: str = MODEL_NAME, db_type: str = VECTOR_DB_TYPE) -> BaseVectorDB:
    """
    Factory function for vector database provider.
    Switches between FAISS and Qdrant based on `VECTOR_DB_TYPE` config or argument.
    """
    db_type_lower = db_type.lower().strip()
    if db_type_lower == "qdrant":
        print(f"[VectorDB Provider] Selected Qdrant Vector Engine.")
        return QdrantVectorDB(model_name=model_name)
    else:
        print(f"[VectorDB Provider] Selected FAISS Vector Engine.")
        return FAISSVectorDB(model_name=model_name)
