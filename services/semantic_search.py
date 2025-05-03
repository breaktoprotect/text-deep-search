from pathlib import Path
from typing import List, Tuple, Optional, Dict

from services.data_manager import load_data, local_caching, output_export
from services.data_manager.data_loaders import load_excel, load_csv
from services.sbert_engine import sbert_embedder, sbert_retriever
from models.file_metadata import FileMetadata


def list_cached_file_ids() -> List[FileMetadata]:
    cached = []
    for cache_dir in local_caching.CACHE_ROOT.iterdir():
        metadata_path = cache_dir / "metadata.json"
        if metadata_path.exists():
            try:
                meta = local_caching.load_metadata(cache_dir.name)
                cached.append(meta)
            except Exception:
                continue
    return cached


def inspect_file(file_path: Path) -> Dict[str, Optional[List[str]]]:
    ext = file_path.suffix.lower()
    if ext in (".xlsx", ".xls"):
        sheets = load_data.list_sheets(file_path)
        first_sheet = sheets[0] if sheets else None
        columns = load_data.list_columns(file_path, first_sheet) if first_sheet else []
        return {"sheets": sheets, "columns": columns}
    elif ext == ".csv":
        columns = load_data.list_columns(file_path)
        return {"sheets": None, "columns": columns}
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def prepare_corpus(
    file_path: Path, sheet_name: Optional[str], columns: List[str], model_key: str
) -> str:
    file_id = local_caching.generate_file_id(file_path)
    cache_dir = local_caching.get_cache_dir(file_id)

    records = load_data.extract_data(file_path, sheet_name, columns)
    sentences = [
        " ".join(str(r[col]) for col in columns if col in r and r[col] is not None)
        for r in records
    ]

    embeddings_path = cache_dir / "embeddings.npy"
    metadata_path = cache_dir / "metadata.json"

    needs_embedding = True
    if metadata_path.exists() and embeddings_path.exists():
        meta = local_caching.load_metadata(file_id)
        if meta.model_key == model_key and meta.columns == columns:
            needs_embedding = False

    if needs_embedding:
        embeddings = sbert_embedder.embed_sentences(sentences, model_key)
        local_caching.save_embeddings(file_id, embeddings)
        local_caching.save_sentences(file_id, sentences)
        local_caching.save_records(file_id, records)
        local_caching.save_metadata(
            file_id,
            FileMetadata(
                file_id=file_id,
                file_name=file_path.name,
                model_key=model_key,
                sheet_name=sheet_name,
                columns=columns,
            ),
        )

    return file_id


def query_corpus(
    query: str, file_id: str, model_key: str, top_k: int = 5
) -> List[Tuple[dict, float]]:
    query_vec = sbert_embedder.embed_query(query, model_key)
    corpus_vecs = local_caching.load_embeddings(file_id)
    records = local_caching.load_records(file_id)

    matches = sbert_retriever.get_top_cosine_matches(query_vec, corpus_vecs, top_k)
    return [(records[idx], score) for idx, score in matches]


def export_results(results: List[Tuple[dict, float]], out_path: Path) -> None:
    ext = out_path.suffix.lower()
    if ext == ".csv":
        output_export.export_results_to_csv(results, out_path)
    elif ext == ".txt":
        output_export.export_results_to_txt(results, out_path)
    elif ext == ".json":
        output_export.export_results_to_json(results, out_path)
    else:
        raise ValueError(f"Unsupported export format: {ext}")
