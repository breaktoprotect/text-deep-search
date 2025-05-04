from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from services.sbert_engine.sbert_model_registry import (
    SUPPORTED_MODELS,
    DEFAULT_MODEL_ID,
)


# ? Model cache (per HuggingFace model path)
_model_cache = {}


def get_model(model_key: str = DEFAULT_MODEL_ID) -> SentenceTransformer:
    """Load and cache model by friendly key name."""
    if model_key not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model key: {model_key}")
    model_name = SUPPORTED_MODELS[model_key]
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_sentences(
    sentences: List[str], model_key: str = DEFAULT_MODEL_ID
) -> np.ndarray:
    """Embed a list of corpus sentences."""
    model = get_model(model_key)
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def embed_query(query: str, model_key: str = DEFAULT_MODEL_ID) -> np.ndarray:
    """Embed a single query string."""
    model = get_model(model_key)
    return model.encode([query], convert_to_numpy=True)[0]  # shape: (dim,)
