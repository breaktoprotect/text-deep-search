import pytest
import numpy as np
from pathlib import Path
from services.data_manager import local_caching
import json

from models.file_metadata import FileMetadata


def test_generate_file_id_uniqueness_and_consistency(tmp_path):
    file1 = tmp_path / "file1.xlsx"
    file2 = tmp_path / "file2.xlsx"
    file1.write_text("Hello world")
    file2.write_text("Hello world")

    file3 = tmp_path / "file3.xlsx"
    file3.write_text("Different content")

    id1 = local_caching.generate_file_id(file1)
    id2 = local_caching.generate_file_id(file2)
    id3 = local_caching.generate_file_id(file3)

    assert id1 == id2, "Identical file contents should produce same file_id"
    assert id1 != id3, "Different file contents should produce different file_id"


def test_save_records_creates_cache_dir(tmp_path):
    # Setup
    file_id = "test_id_record_write"
    test_dir = tmp_path / "cache"  # Use isolated tmp dir
    local_caching.CACHE_ROOT = test_dir  # Temporarily override

    sample_records = [{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}]

    # Act
    local_caching.save_records(file_id, sample_records)

    # Assert
    records_path = test_dir / file_id / "records.json"
    assert records_path.exists(), "records.json was not created"

    with open(records_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data == sample_records


def test_load_metadata_raises_when_missing(tmp_path):
    fake_file_id = "nonexistent_file_id"
    # Ensure the directory does not exist
    cache_dir = tmp_path / fake_file_id
    assert not (cache_dir / "metadata.json").exists()

    with pytest.raises(FileNotFoundError):
        local_caching.load_metadata(fake_file_id)


def test_local_caching_full_cycle():
    file_id = "d41d8cd98f00b204e9800998ecf8427e_3f2a5e142abbc1f197d66b19fca1aa60"
    cache_dir = Path("cache") / file_id

    metadata = FileMetadata(
        file_id=file_id,
        file_name="test_file.xlsx",  # ← corrected
        model_key="MiniLM-L6-v2",  # ← added
        sheet_name="Sheet1",
        columns=["A", "B"],
    )
    sentences = ["Row one text", "Row two text"]
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]])
    records = [{"A": "value1", "B": "value2"}, {"A": "value3", "B": "value4"}]

    # Save all
    local_caching.save_metadata(file_id, metadata)
    local_caching.save_sentences(file_id, sentences)
    local_caching.save_embeddings(file_id, embeddings)
    local_caching.save_records(file_id, records)

    # Load and validate
    loaded_metadata = local_caching.load_metadata(file_id)
    assert loaded_metadata == metadata
    assert local_caching.load_sentences(file_id) == sentences
    assert np.array_equal(local_caching.load_embeddings(file_id), embeddings)
    assert local_caching.load_records(file_id) == records

    # Cleanup
    local_caching.delete_cache(file_id)
    assert not cache_dir.exists()


def test_is_file_cached(tmp_path):
    file_id = "fake123"
    cache_dir = local_caching.get_cache_dir(file_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "metadata.json").write_text("{}")

    assert local_caching.is_file_cached(file_id) is True

    local_caching.delete_cache(file_id)
    assert local_caching.is_file_cached(file_id) is False
