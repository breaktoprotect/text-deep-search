import pytest
from models.file_metadata import FileMetadata


def test_valid_metadata():
    metadata = FileMetadata(
        file_id="abc123",
        file_name="example.csv",
        model_key="MiniLM-L6-v2",
        columns=["Policy", "Description"],
        sheet_name=None,  # should pass since Optional[str]
    )
    assert metadata.file_id == "abc123"
    assert metadata.columns == ["Policy", "Description"]
    assert metadata.sheet_name is None


def test_missing_required_field():
    with pytest.raises(Exception) as exc:
        FileMetadata(
            file_id="abc123", model="MiniLM-L6-v2", columns=["ColA"], sheet_name=None
        )
    assert "file_name" in str(exc.value)


def test_invalid_column_type():
    with pytest.raises(Exception) as exc:
        FileMetadata(
            file_id="abc123",
            file_name="ok.xlsx",
            model_key="MiniLM-L6-v2",
            columns="not-a-list",  # should fail
            sheet_name="Sheet1",
        )
    assert "columns" in str(exc.value)


def test_allow_empty_columns_list():
    metadata = FileMetadata(
        file_id="abc123",
        file_name="none.xlsx",
        model_key="MiniLM-L6-v2",
        columns=[],
        sheet_name="Sheet1",
    )
    assert metadata.columns == []
