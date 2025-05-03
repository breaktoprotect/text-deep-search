from sentence_transformers import SentenceTransformer
from typing import List, Tuple
import numpy as np
import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Registry of human-readable keys → HuggingFace model names
SUPPORTED_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "MPNet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}
DEFAULT_MODEL_ID = os.getenv("TEDDY_SEARCH_DEFAULT_MODEL", "MiniLM-L6-v2")

# ? Model cache
# TODO: move it somewhere more global?
_model_cache = {}  # Model cache (per backend name)


def list_supported_models() -> List[str]:
    """Return list of model keys for UI dropdown or config."""
    return list(SUPPORTED_MODELS.keys())


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
