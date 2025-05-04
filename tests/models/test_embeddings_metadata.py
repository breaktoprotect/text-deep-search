import pytest
from models.embeddings_metadata import EmbeddingMetadata


def test_valid_embedding_metadata():
    metadata = EmbeddingMetadata(
        embedding_id="abc123",
        file_hash="f00dbabe",
        file_name="sample.xlsx",
        model_key="MiniLM-L6-v2",
        columns=["col1", "col2"],
        sheet_name="Sheet1",
    )

    assert metadata.embedding_id == "abc123"
    assert metadata.file_hash == "f00dbabe"
    assert metadata.file_name.endswith(".xlsx")
    assert metadata.model_key == "MiniLM-L6-v2"
    assert metadata.columns == ["col1", "col2"]
    assert metadata.sheet_name == "Sheet1"


def test_optional_sheet_name_is_none():
    metadata = EmbeddingMetadata(
        embedding_id="hash123",
        file_hash="deadbeef",
        file_name="data.csv",
        model_key="all-MiniLM-L6-v2",
        columns=["Name", "Age"],
        sheet_name=None,
    )

    assert metadata.sheet_name is None


def test_invalid_missing_fields():
    with pytest.raises(ValueError):
        EmbeddingMetadata(
            embedding_id="missing",
            file_hash="deadbeef",
            file_name="file.csv",
            model_key="some-model",
            columns=None,  # Invalid: columns must be a list
        )
