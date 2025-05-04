from services.data_manager.data_loaders import load_csv
from pathlib import Path


def test_list_columns_csv():
    path = Path("tests/test_files/sample.csv")
    columns = load_csv.list_columns(path)
    assert columns == ["Name", "Age", "City", "Department", "Active"]


def test_extract_data_csv_subset_columns():
    path = Path("tests/test_files/sample.csv")
    columns = ["Name", "City", "Active"]
    rows = load_csv.extract_data(path, columns)
    assert isinstance(rows, list)
    assert all(isinstance(row, dict) for row in rows)
    assert rows[0]["Name"] == "Alice"
    assert rows[0]["City"] == "Singapore"
    assert rows[0]["Active"] == "Yes"


def test_extract_data_csv_full_columns():
    path = Path("tests/test_files/sample.csv")
    columns = ["Name", "Age", "City", "Department", "Active"]
    rows = load_csv.extract_data(path, columns)
    assert len(rows) == 5
    assert rows[-1]["Name"] == "Evan"
    assert rows[-1]["Active"] == "No"
