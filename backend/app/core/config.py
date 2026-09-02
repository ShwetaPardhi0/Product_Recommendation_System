"""
Application Configuration Settings
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PRODUCTS_FILE_PATH = DATA_DIR / "products.json"

# Load backend/.env file if present
ENV_PATH = BASE_DIR / "backend" / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)

# Recommender System Parameters
DEFAULT_RECOMMENDATION_LIMIT = 12
MODEL_NAME = "all-MiniLM-L6-v2"
MMR_LAMBDA = 0.75
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "qdrant")  # Default to qdrant when credentials present

# Qdrant Cloud Credentials
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT", "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or os.getenv("QDRABT_API_KEY") or "").strip()
