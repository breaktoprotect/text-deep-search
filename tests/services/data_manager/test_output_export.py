import pytest
from pathlib import Path
import csv
import json

from services.data_manager import output_export

OUTPUT_DIR = Path("tests/test_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "results_test.csv"
TXT_OUTPUT = OUTPUT_DIR / "results_test.txt"
JSON_OUTPUT = OUTPUT_DIR / "results_test.json"


@pytest.fixture
def sample_results():
    return [
        ({"Policy": "Enable UAC", "Description": "Ensure UAC is on."}, 0.89),
        (
            {
                "Policy": "Set password length",
                "Description": "Passwords must be 14 chars.",
            },
            0.76,
        ),
    ]


def test_export_results_to_csv(sample_results):
    output_export.export_results_to_csv(sample_results, CSV_OUTPUT)
    assert CSV_OUTPUT.exists()

    with open(CSV_OUTPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 2
    assert "_score" in rows[0]
    assert rows[0]["Policy"] == "Enable UAC"


def test_export_results_to_txt(sample_results):
    output_export.export_results_to_txt(sample_results, TXT_OUTPUT)
    assert TXT_OUTPUT.exists()

    content = TXT_OUTPUT.read_text(encoding="utf-8")
    assert "Score: 0.89" in content
    assert "Policy: Enable UAC" in content
    assert "Description: Ensure UAC is on." in content


def test_export_results_to_json(sample_results):
    output_export.export_results_to_json(sample_results, JSON_OUTPUT)
    assert JSON_OUTPUT.exists()

    data = json.loads(JSON_OUTPUT.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert data[0]["Policy"] == "Enable UAC"
    assert data[0]["_score"] == 0.89
