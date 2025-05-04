import pytest
from pathlib import Path
import json
from services import semantic_search

TEST_FILES = Path("tests/test_files")
OUTPUT_DIR = Path("tests/test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_inspect_file_csv():
    csv_path = TEST_FILES / "sample.csv"
    info = semantic_search.inspect_file(csv_path)
    assert info["sheets"] is None
    assert "Name" in info["columns"]
    assert "City" in info["columns"]


def test_inspect_file_excel():
    xlsx_path = TEST_FILES / "sample.xlsx"
    info = semantic_search.inspect_file(xlsx_path)
    assert "Sheet1" in info["sheets"]
    assert "Name" in info["columns"]
    assert "City" in info["columns"]


def test_prepare_and_query_corpus_csv():
    path = TEST_FILES / "sample.csv"
    columns = ["Name", "City"]
    model_key = "MiniLM-L6-v2"

    embedding_id = semantic_search.prepare_corpus(path, None, columns, model_key)
    assert embedding_id

    results = semantic_search.query_corpus(
        "Singapore", embedding_id, model_key, top_k=3
    )
    assert len(results) > 0
    assert isinstance(results[0][0], dict)
    assert isinstance(results[0][1], float)


def test_export_results_to_json(tmp_path):
    results = [
        ({"Name": "Alice", "City": "Singapore"}, 0.95),
        ({"Name": "Bob", "City": "Tokyo"}, 0.88),
    ]
    out_file = tmp_path / "exported_results.json"
    semantic_search.export_results(results, out_file)
    assert out_file.exists()

    data = json.loads(out_file.read_text())
    assert isinstance(data, list)
    assert data[0]["_score"] == 0.95


def test_export_results_invalid_extension():
    results = [({"Test": "Value"}, 1.0)]
    out_path = OUTPUT_DIR / "invalid.xyz"
    with pytest.raises(ValueError):
        semantic_search.export_results(results, out_path)
