"""
Text Preprocessing, Boilerplate Cleaning, & Quality Filtering Module
"""

import re
from typing import Dict, Any, List

_PLACEHOLDER_DESCRIPTIONS = {
    "good quality", "good quailty", "deepesh created",
    "this group is created by deepesh for clothes",
}


def normalize_occasion(occasion: str) -> str:
    return re.sub(r"[^\w\s]", "", occasion.lower()).strip()


def is_low_quality(product: Dict[str, Any]) -> bool:
    name = (product.get("name") or "").strip().lower()
    desc = (product.get("description") or "").strip().lower()
    if desc in _PLACEHOLDER_DESCRIPTIONS:
        return True
    if desc == f"{name} description":
        return True
    return False


def build_product_text(product: Dict[str, Any]) -> str:
    """
    Constructs rich searchable product text for TF-IDF & dense vector indexing.
    Weights: Name x3, Brand/Slug x2, Description x1, shortDescription x1
    """
    name = product.get("name", "").strip()
    brand = product.get("brand", "").strip()
    slug = (product.get("slug") or "").replace("-", " ").strip()
    description = (product.get("description") or "").strip()
    short_desc = (product.get("shortDescription") or "").strip()

    parts = [
        name, name, name,       # Title x3 weight
        brand, brand,           # Brand x2 weight
        slug, slug,             # Slug x2 weight
        description,            # Full description x1
        short_desc,             # Short description x1
    ]
    return " ".join(p for p in parts if p).strip()
