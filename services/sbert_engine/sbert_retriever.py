from sentence_transformers.util import cos_sim
import numpy as np
from typing import List, Tuple


def get_top_cosine_matches(
    query_embedding: np.ndarray, corpus_embeddings: np.ndarray, top_k: int = 5
) -> List[Tuple[int, float]]:
    """
    Use SBERT's built-in cosine similarity to return top-k matches.

    Args:
        query_embedding: (dim,) — SBERT embedding of the query
        corpus_embeddings: (n, dim) — precomputed corpus embeddings
        top_k: number of top matches to return

    Returns:
        List of (index, similarity score) sorted by highest score
    """
    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be a 1D array")
    if corpus_embeddings.ndim != 2:
        raise ValueError("Corpus embeddings must be a 2D array")

    scores = cos_sim(query_embedding, corpus_embeddings)[0]
    scores_np = scores.cpu().numpy() if hasattr(scores, "cpu") else scores.numpy()
    top_indices = np.argsort(scores_np)[::-1][:top_k]

    return [(int(i), float(scores_np[i])) for i in top_indices]
