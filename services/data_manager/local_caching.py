from pathlib import Path
import json
import hashlib
import shutil
import numpy as np
from typing import List, Optional
from models.embeddings_metadata import EmbeddingMetadata


CACHE_ROOT = Path("cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def generate_file_hash(file_path: Path) -> str:
    """Generate MD5 hash of the raw file content."""
    file_bytes = file_path.read_bytes()
    return hashlib.md5(file_bytes).hexdigest()


def compute_embedding_id(
    file_hash: str,
    columns: List[str],
    model_key: str,
    sheet_name: Optional[str] = None,
) -> str:
    """Compute deterministic embedding ID based on config."""
    fingerprint = {
        "file_hash": file_hash,
        "columns": columns,
        "model_key": model_key,
        "sheet_name": sheet_name,
    }
    json_str = json.dumps(fingerprint, sort_keys=True)
    return hashlib.md5(json_str.encode("utf-8")).hexdigest()


def get_cache_dir(embedding_id: str) -> Path:
    return CACHE_ROOT / embedding_id


def is_cached(embedding_id: str) -> bool:
    """Check if metadata exists for the embedding."""
    return (get_cache_dir(embedding_id) / "metadata.json").exists()


def save_metadata(metadata: EmbeddingMetadata) -> None:
    cache_dir = get_cache_dir(metadata.embedding_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def save_sentences(embedding_id: str, sentences: List[str]) -> None:
    cache_dir = get_cache_dir(embedding_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "sentences.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2)


def save_embeddings(embedding_id: str, embeddings: np.ndarray) -> None:
    cache_dir = get_cache_dir(embedding_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)


def save_records(embedding_id: str, records: List[dict]) -> None:
    cache_dir = get_cache_dir(embedding_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_metadata(embedding_id: str) -> EmbeddingMetadata:
    path = get_cache_dir(embedding_id) / "metadata.json"
    return EmbeddingMetadata.model_validate_json(path.read_text())


def load_sentences(embedding_id: str) -> List[str]:
    with open(
        get_cache_dir(embedding_id) / "sentences.json", "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def load_embeddings(embedding_id: str) -> np.ndarray:
    return np.load(get_cache_dir(embedding_id) / "embeddings.npy")


def load_records(embedding_id: str) -> List[dict]:
    with open(get_cache_dir(embedding_id) / "records.json", "r", encoding="utf-8") as f:
        return json.load(f)


def delete_cache(embedding_id: str) -> None:
    shutil.rmtree(get_cache_dir(embedding_id), ignore_errors=True)
