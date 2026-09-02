# OccasionAI — AI/ML Hybrid Product Recommendation System

> **AI/ML Engineer Take-Home Assignment**
> An intelligent, hybrid semantic gift recommendation system powered by **SentenceTransformers (`all-MiniLM-L6-v2`)**, **FAISS Vector Search**, and **TF-IDF Lexical Matching**.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.10+ · FastAPI · Uvicorn |
| **Embedding Model** | `all-MiniLM-L6-v2` (Sentence Transformers, 384-d dense vectors) |
| **Vector Search** | FAISS (`IndexFlatIP` — Cosine Similarity on L2-normalized vectors) |
| **Lexical Engine** | TF-IDF (word + n-grams) + Synonym Expansion |
| **Frontend** | Vanilla HTML5 · CSS3 (Glassmorphism) · JavaScript (Fetch API) |
| **Data** | `products.json` (108 active product items) |

---

## 🏗️ Recommendation Engine Architecture

```
                               Occasion Input (e.g. "Birthday")
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
       Dense Vector Embedding                                Lexical TF-IDF Engine
     (SentenceTransformers 384-d)                           (Word + N-gram Vector)
                    │                                                   │
                    ▼                                                   ▼
            FAISS Index Search                                 Lexical Similarity
        (faiss.IndexFlatIP cosine)                              (scikit-learn)
                    │                                                   │
                    └─────────────────────────┬─────────────────────────┘
                                              ▼
                                   Hybrid Score Ensemble
                      (0.50 * FAISS_Dense + 0.35 * TFIDF_Lexical + 0.15 * Keyword_Boost)
                                              │
                                              ▼
                                    Top N Ranked Products
```

---

## 🚀 Quick Start

### 1. Requirements

Ensure dependencies are installed in your environment:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
python -m uvicorn main:app --reload --port 8000
```

*Console log on startup:*
```text
[Recommender] Loaded 108 products for indexing.
[Recommender] TF-IDF lexical matrix indexed.
[Recommender] Loading SentenceTransformer model 'all-MiniLM-L6-v2'...
[Recommender] Encoding product embeddings...
[Recommender] FAISS IndexFlatIP constructed with 108 vectors (dim=384).
[API] Recommendation engine ready.
```

### 3. Open Frontend

Open `frontend/index.html` in your web browser.

---

## 🔌 API Reference

### `POST /recommend`

```json
// Request Body
{
  "occasion": "Wedding",
  "limit": 12
}

// Response (Sample item)
{
  "id": "cmkzaq7ea0068ns01f4me9ffh",
  "name": "Rose gold watch",
  "brand": "KA",
  "price": "2499",
  "relevance_score": 0.3842,
  "dense_score": 0.4125,
  "lexical_score": 0.1205
}
```

---

## 📸 Screen Screenshots & Deliverables

- **Backend API Docs**: `http://localhost:8000/docs`
- **Frontend App**: `frontend/index.html`
