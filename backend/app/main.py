"""
FastAPI Backend Application Entrypoint
"""

import sys
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure app package is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import PRODUCTS_FILE_PATH, DEFAULT_RECOMMENDATION_LIMIT
from app.services.recommender import ProductRecommender

app = FastAPI(
    title="Neural Hybrid Product Recommendation API",
    description="Occasion-based product recommendation engine combining SentenceTransformers dense vector search (FAISS) + TF-IDF lexical search + RapidFuzz typo-tolerance + MMR diversity re-ranking.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender: Optional[ProductRecommender] = None


@app.on_event("startup")
def startup_event():
    global recommender
    print(f"[API Startup] Initializing recommender engine with catalog from: {PRODUCTS_FILE_PATH}")
    if not PRODUCTS_FILE_PATH.exists():
        print(f"[API Warning] Catalog file not found at {PRODUCTS_FILE_PATH}")
    recommender = ProductRecommender(products_path=str(PRODUCTS_FILE_PATH))
    print("[API Startup] Recommender engine ready.")


class ProductResponse(BaseModel):
    id: str
    name: str
    slug: Optional[str] = None
    description: Optional[str] = None
    shortDescription: Optional[str] = None
    mainImage: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[str] = None
    discountPercentage: Optional[str] = None
    stock: Optional[int] = None
    relevance_score: Optional[float] = None
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    keyword_boost: Optional[float] = None
    title_boost: Optional[float] = None
    mismatch_penalty: Optional[float] = None


class RecommendationResponse(BaseModel):
    occasion: str
    total_results: int
    recommendations: List[ProductResponse]


class CreateProductRequest(BaseModel):
    name: str = Field(..., min_length=2, description="Product Name")
    brand: Optional[str] = "Generic"
    description: Optional[str] = ""
    shortDescription: Optional[str] = ""
    mainImage: Optional[str] = "https://images.unsplash.com/photo-1549465220-1a8b9238cd48?w=500&auto=format&fit=crop"
    price: Optional[str] = "$29.99"
    discountPercentage: Optional[str] = "0"
    stock: Optional[int] = 50


@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "Neural Hybrid Product Recommendation Engine",
        "docs_url": "/docs",
    }


@app.get("/api/products", response_model=List[ProductResponse])
def get_all_products(limit: int = Query(100, ge=1, le=500)):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender engine is not initialized.")
    return recommender.get_all_products(limit=limit)


@app.post("/api/products", response_model=ProductResponse, status_code=201)
def create_product(product: CreateProductRequest):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender engine is not initialized.")
    
    product_dict = product.model_dump()
    created_product = recommender.add_product(product_dict)
    return created_product


@app.get("/api/recommend", response_model=RecommendationResponse)
def get_recommendations(
    occasion: str = Query(..., min_length=1, description="Occasion query (e.g. Birthday, Wedding, Diwali)"),
    limit: int = Query(DEFAULT_RECOMMENDATION_LIMIT, ge=1, le=50, description="Max recommendations to return"),
):
    if not recommender:
        raise HTTPException(status_code=500, detail="Recommender engine is not initialized.")
    
    results = recommender.recommend(occasion=occasion, limit=limit)
    return RecommendationResponse(
        occasion=occasion,
        total_results=len(results),
        recommendations=results,
    )
