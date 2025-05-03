from services.data_loaders import load_excel
from pathlib import Path


def test_list_sheets_excel():
    path = Path("tests/test_files/sample.xlsx")
    sheets = load_excel.list_sheets(path)
    assert "Sheet1" in sheets


def test_list_columns_excel():
    path = Path("tests/test_files/sample.xlsx")
    columns = load_excel.list_columns(path, "Sheet1")
    assert columns == ["Name", "Age", "City", "Department", "Active"]


def test_extract_data_excel_subset_columns():
    path = Path("tests/test_files/sample.xlsx")
    columns = ["Name", "City", "Active"]
    rows = load_excel.extract_data(path, "Sheet1", columns)
    assert isinstance(rows, list)
    assert rows[0]["Name"] == "Alice"
    assert rows[0]["City"] == "Singapore"
    assert rows[0]["Active"] == "Yes"


def test_extract_data_excel_full_columns():
    path = Path("tests/test_files/sample.xlsx")
    columns = ["Name", "Age", "City", "Department", "Active"]
    rows = load_excel.extract_data(path, "Sheet1", columns)
    assert len(rows) == 5
    assert rows[-1]["Name"] == "Evan"
    assert rows[-1]["Department"] == "IT"
