from pathlib import Path
import json
import numpy as np
from typing import List
import shutil
import hashlib

from models.file_metadata import FileMetadata

CACHE_ROOT = Path("cache")
CACHE_ROOT.mkdir(parents=True, exist_ok=True)  # Ensure ./cache exists


def get_cache_dir(file_id: str) -> Path:
    return CACHE_ROOT / file_id


def generate_file_id(path: Path) -> str:
    file_bytes = path.read_bytes()
    return hashlib.md5(file_bytes).hexdigest()


def is_file_cached(file_id: str) -> bool:
    """Check if the file's metadata exists in the cache (used as cache presence signal)."""
    return (get_cache_dir(file_id) / "metadata.json").exists()


def save_metadata(file_id: str, metadata: FileMetadata) -> None:
    cache_dir = get_cache_dir(file_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        f.write(metadata.model_dump_json(indent=2))


def save_sentences(file_id: str, sentences: List[str]) -> None:
    cache_dir = get_cache_dir(file_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "sentences.json", "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2)


def save_embeddings(file_id: str, embeddings: np.ndarray) -> None:
    cache_dir = get_cache_dir(file_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "embeddings.npy", embeddings)


def save_records(file_id: str, records: List[dict]) -> None:
    cache_dir = get_cache_dir(file_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def load_metadata(file_id: str) -> FileMetadata:
    path = get_cache_dir(file_id) / "metadata.json"
    return FileMetadata.model_validate_json(path.read_text())


def load_sentences(file_id: str) -> List[str]:
    with open(get_cache_dir(file_id) / "sentences.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_embeddings(file_id: str) -> np.ndarray:
    return np.load(get_cache_dir(file_id) / "embeddings.npy")


def load_records(file_id: str) -> List[dict]:
    with open(get_cache_dir(file_id) / "records.json", "r", encoding="utf-8") as f:
        return json.load(f)


def delete_cache(file_id: str) -> None:
    shutil.rmtree(get_cache_dir(file_id), ignore_errors=True)
