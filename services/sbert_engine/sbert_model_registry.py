import os
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Registry of human-readable keys → HuggingFace model names
SUPPORTED_MODELS = {
    "MiniLM-L6-v2": "sentence-transformers/paraphrase-MiniLM-L6-v2",
    "MPNet-base-v2": "sentence-transformers/all-mpnet-base-v2",
}

DEFAULT_MODEL_ID = os.getenv("TEDDY_SEARCH_DEFAULT_MODEL", "MiniLM-L6-v2")


def list_supported_models():
    return list(SUPPORTED_MODELS.keys())
