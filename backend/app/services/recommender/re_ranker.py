"""
Maximal Marginal Relevance (MMR) & RapidFuzz Category Diversity Re-ranker Module
"""

import re
from typing import List, Dict, Any

import numpy as np

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


def extract_product_type_hybrid(name: str) -> str:
    words = set(re.findall(r"[a-z]+", name.lower()))
    if "gift" in words and "card" in words or "cards" in words and "gift" in words:
        return "gift_card"
    
    main_words = [w for w in words if w not in {"the", "a", "an", "and", "or", "for", "by", "of", "with", "gift", "gifts"}]
    if not main_words:
        return "general"

    return main_words[0]


def apply_mmr_diversity_reranking(
    products: List[Dict[str, Any]],
    candidate_indices: List[int],
    rel_scores: np.ndarray,
    dense_embeddings: np.ndarray | None,
    limit: int = 12,
    lambda_mult: float = 0.75,
) -> List[int]:
    """
    Selects top candidates while penalizing feature vector similarity and category repetition.
    """
    selected_indices: List[int] = []
    selected_types: Dict[str, int] = {}

    remaining_candidates = list(candidate_indices)

    while len(remaining_candidates) > 0 and len(selected_indices) < limit:
        best_score = -999.0
        best_candidate = -1

        for cand_idx in remaining_candidates:
            cand_rel = rel_scores[cand_idx]

            if not selected_indices:
                max_sim = 0.0
            else:
                if dense_embeddings is not None:
                    sims = [
                        float(np.dot(dense_embeddings[cand_idx], dense_embeddings[sel_idx]))
                        for sel_idx in selected_indices
                    ]
                    max_sim = max(sims)
                else:
                    max_sim = 0.0

            p_type = extract_product_type_hybrid(products[cand_idx].get("name", ""))
            type_count = selected_types.get(p_type, 0)
            category_penalty = 0.12 * type_count

            mmr_score = (lambda_mult * cand_rel) - ((1 - lambda_mult) * max_sim) - category_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = cand_idx

        if best_candidate != -1:
            selected_indices.append(best_candidate)
            remaining_candidates.remove(best_candidate)
            p_type = extract_product_type_hybrid(products[best_candidate].get("name", ""))
            selected_types[p_type] = selected_types.get(p_type, 0) + 1

    return selected_indices
