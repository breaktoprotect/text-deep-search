import pytest
import numpy as np
from pathlib import Path
import json
from services.data_manager import local_caching
from models.embeddings_metadata import EmbeddingMetadata


def test_generate_file_hash_consistency(tmp_path):
    file1 = tmp_path / "sample1.csv"
    file2 = tmp_path / "sample2.csv"
    file1.write_text("hello world")
    file2.write_text("hello world")

    file3 = tmp_path / "sample3.csv"
    file3.write_text("different content")

    h1 = local_caching.generate_file_hash(file1)
    h2 = local_caching.generate_file_hash(file2)
    h3 = local_caching.generate_file_hash(file3)

    assert h1 == h2
    assert h1 != h3


def test_compute_embedding_id_changes_on_input_variants():
    file_hash = "abcd1234"
    base_columns = ["col1", "col2"]
    model_key = "MiniLM-L6-v2"
    sheet_name = "Sheet1"

    id1 = local_caching.compute_embedding_id(
        file_hash, base_columns, model_key, sheet_name
    )
    id2 = local_caching.compute_embedding_id(file_hash, ["col1"], model_key, sheet_name)
    id3 = local_caching.compute_embedding_id(
        file_hash, base_columns, "MPNet", sheet_name
    )

    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_save_and_load_metadata(tmp_path):
    local_caching.CACHE_ROOT = tmp_path

    metadata = EmbeddingMetadata(
        embedding_id="abc123",
        file_hash="f00dbabe",
        file_name="test.xlsx",
        model_key="MiniLM-L6-v2",
        columns=["A", "B"],
        sheet_name="Data",
    )

    local_caching.save_metadata(metadata)
    loaded = local_caching.load_metadata("abc123")

    assert loaded == metadata


def test_full_cache_cycle(tmp_path):
    local_caching.CACHE_ROOT = tmp_path
    embedding_id = "abc123"

    metadata = EmbeddingMetadata(
        embedding_id=embedding_id,
        file_hash="f00dbabe",
        file_name="demo.csv",
        model_key="MiniLM-L6-v2",
        columns=["Name", "Score"],
        sheet_name=None,
    )
    sentences = ["This is row one", "This is row two"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    records = [{"Name": "Alice", "Score": "90"}, {"Name": "Bob", "Score": "80"}]

    local_caching.save_metadata(metadata)
    local_caching.save_sentences(embedding_id, sentences)
    local_caching.save_embeddings(embedding_id, embeddings)
    local_caching.save_records(embedding_id, records)

    assert local_caching.load_metadata(embedding_id) == metadata
    assert local_caching.load_sentences(embedding_id) == sentences
    assert np.array_equal(local_caching.load_embeddings(embedding_id), embeddings)
    assert local_caching.load_records(embedding_id) == records


def test_delete_cache(tmp_path):
    local_caching.CACHE_ROOT = tmp_path
    embedding_id = "to_delete"
    local_caching.save_sentences(embedding_id, ["x", "y"])
    assert local_caching.get_cache_dir(embedding_id).exists()

    local_caching.delete_cache(embedding_id)
    assert not local_caching.get_cache_dir(embedding_id).exists()


def test_is_cached(tmp_path):
    local_caching.CACHE_ROOT = tmp_path
    embedding_id = "check123"
    cache_dir = local_caching.get_cache_dir(embedding_id)
    cache_dir.mkdir(parents=True)
    (cache_dir / "metadata.json").write_text("{}")

    assert local_caching.is_cached(embedding_id)
    local_caching.delete_cache(embedding_id)
    assert not local_caching.is_cached(embedding_id)
