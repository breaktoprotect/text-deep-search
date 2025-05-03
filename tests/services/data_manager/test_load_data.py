import pytest
from services.data_manager import load_data
from services.data_manager.load_data import UnsupportedFileTypeError
from pathlib import Path


def test_dispatch_csv_list_columns():
    path = Path("tests/test_files/sample.csv")
    result = load_data.list_columns(path)
    assert result == ["Name", "Age", "City", "Department", "Active"]


def test_dispatch_excel_list_columns():
    path = Path("tests/test_files/sample.xlsx")
    result = load_data.list_columns(path, sheet_name="Sheet1")
    assert result == ["Name", "Age", "City", "Department", "Active"]


def test_dispatch_csv_extract_data():
    path = Path("tests/test_files/sample.csv")
    result = load_data.extract_data(path, sheet_name=None, columns=["Name", "City"])
    assert isinstance(result, list)
    assert result[0]["Name"] == "Alice"
    assert result[0]["City"] == "Singapore"


def test_dispatch_excel_extract_data():
    path = Path("tests/test_files/sample.xlsx")
    result = load_data.extract_data(path, sheet_name="Sheet1", columns=["Name", "City"])
    assert isinstance(result, list)
    assert result[0]["City"] == "Singapore"


def test_dispatch_invalid_format_raises(tmp_path):
    dummy_path = tmp_path / "sample.unsupported"
    dummy_path.write_text("invalid content")
    with pytest.raises(UnsupportedFileTypeError):
        load_data.extract_data(dummy_path, None, ["Dummy"])
