# OccasionAI — Neural Hybrid Product Recommendation Engine

> **Production-Ready Neural Hybrid Gift Recommendation System**  
> Powered by **FastAPI**, **SentenceTransformers (`all-MiniLM-L6-v2`)**, **Qdrant Vector DB / FAISS**, **TF-IDF Lexical Matching**, **RapidFuzz Typo Tolerance**, and **MMR Diversity Re-Ranking**.

---

## 🛠️ Tech Stack & Architecture

| Layer | Technology & Implementation Details |
|-------|-------------------------------------|
| **Backend Framework** | Python 3.10+ · FastAPI · Uvicorn (Asynchronous REST API) |
| **Dense Vector Model** | `all-MiniLM-L6-v2` (SentenceTransformers, 384-dimensional dense embeddings) |
| **Vector Engines** | **Qdrant Cloud** (Remote Cluster) / **FAISS** (`IndexFlatIP` Cosine Similarity) |
| **Lexical Engine** | TF-IDF (Unigram + Bigram sparse matrix) |
| **Diversity & Re-Ranking** | Maximal Marginal Relevance (MMR) + RapidFuzz category deduplication |
| **Frontend SPA** | Single Page Application (HTML5, Vanilla CSS3 Glassmorphism, JS Fetch API) |
| **Product Database** | `data/products.json` (Dynamic read/write persistence with real-time vector indexing) |

---

## 🏗️ Recommender Pipeline Architecture

```
                                Occasion Query Input (e.g. "Birthday", "Diwali")
                                                       │
                           ┌───────────────────────────┴───────────────────────────┐
                           ▼                                                       ▼
             Dense Vector Search (Qdrant/FAISS)                         Lexical Search (TF-IDF)
          Cosine Similarity on 384-d Embeddings                       Sparse N-gram Matrix Matching
                           │                                                       │
                           ▼                                                       ▼
                      Dense Scores                                            Lexical Scores
                           │                                                       │
                           └───────────────────────────┬───────────────────────────┘
                                                       ▼
                                            Ensemble Scoring Engine
                                  (0.55 * Dense + 0.20 * TF-IDF + 0.15 * Title 
                                   + 0.10 * Keyword - Mismatch Penalties)
                                                       │
                                                       ▼
                                            MMR Diversity Re-Ranking
                                    (Maximal Marginal Relevance Lambda = 0.75)
                                                       │
                                                       ▼
                                            Top Recommended Products
```

---

## 📁 Modular Folder Structure

```text
Product_Recommendation_System/
├── backend/
│   ├── .env                       # Environment credentials (Qdrant Endpoint, API Key, VECTOR_DB_TYPE)
│   ├── requirements.txt           # Python dependencies
│   └── app/
│       ├── main.py                # FastAPI entrypoint & REST API endpoints
│       ├── core/
│       │   └── config.py          # Centralized configuration & environment setup
│       └── services/
│           ├── vector_db.py       # Pluggable Vector DB abstraction (BaseVectorDB, FAISSVectorDB, QdrantVectorDB)
│           └── recommender/
│               ├── __init__.py
│               ├── engine.py       # ProductRecommender orchestrator & scoring ensemble
│               ├── indexers.py     # LexicalIndexManager (TF-IDF sparse index)
│               ├── preprocessing.py# Text normalization & product document construction
│               └── re_ranker.py    # MMR diversity re-ranking algorithm
├── frontend/
│   ├── index.html                 # Multi-view SPA (Discover, Admin Catalog, Add Product, Error Page)
│   ├── style.css                  # Modern dark glassmorphism design system
│   └── script.js                  # Frontend SPA routing, API fetch client, pagination, & search
└── data/
    └── products.json              # Product catalog database
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Start the FastAPI application:

```bash
# Ensure command is run from the 'backend' directory
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

*Backend runs at `http://localhost:8000` (Swagger docs available at `http://localhost:8000/docs`).*

### 2. Frontend Setup

Serve the static single page application using Python's built-in HTTP server:

```bash
cd frontend
python -m http.server 3000
```

*Access the web UI at `http://localhost:3000`.*

### 3. Docker Deployment (Recommended)

Run the full stack with Docker Compose:

```bash
# Build and run containers
docker-compose up --build
```

- **Web Frontend**: `http://localhost:3000`
- **FastAPI Backend**: `http://localhost:8000` (Docs at `http://localhost:8000/docs`)

---

## 🔌 API Reference

### 1. `GET /api/recommend`
Fetches hybrid AI product recommendations based on occasion.

**Parameters:**
- `occasion` (query string, required): e.g. `Birthday`, `Diwali`, `Wedding`
- `limit` (query int, default 12): Number of products to return.

### 2. `GET /api/products`
Retrieves product catalog items for administrative display.

**Parameters:**
- `limit` (query int, default 100): Maximum catalog products to retrieve.

### 3. `POST /api/products`
Dynamically adds a new product to the catalog, updates `products.json`, and triggers real-time vector embedding generation & store upsert.

**Payload:**
```json
{
  "name": "Birthday Party Gift Set",
  "brand": "Joy Gifts",
  "price": "$39.99",
  "mainImage": "https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=500&auto=format&fit=crop",
  "description": "Premium curated gift box containing celebratory treats and candles."
}
```

---

## 🛡️ Admin Portal Features

- **Single Page View Switching**: Seamless navigation between Gift Finder and Admin Portal.
- **Admin Search Bar**: Filter products live by title or brand.
- **Catalog Pagination**: 10 products per page with intuitive Next/Prev controls.
- **Dynamic Product Creation**: Instant embedding vector generation upon addition.
- **Robust Media Fallback**: Automatic image error handling with DOM initial placeholders.

---

## 🤖 AI Assistance Statement

AI coding assistants (such as ChatGPT and GitHub Copilot) were used selectively for frontend development, UI refinement, debugging, and development guidance. The architectural design, recommendation engine logic, vector database integrations, and final implementation were reviewed, tested, and validated.

