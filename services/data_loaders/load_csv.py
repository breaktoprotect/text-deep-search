import csv
from typing import List, Dict, Union
from pathlib import Path


def list_columns(file_path: Union[str, Path]) -> List[str]:
    """
    Returns the column headers from a CSV file using built-in csv module.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)


def extract_data(
    file_path: Union[str, Path],
    columns: List[str],
) -> List[Dict[str, Union[str, float]]]:
    """
    Extracts rows using only the specified columns, skipping completely empty rows.
    """
    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        selected_rows = []
        for row in reader:
            selected_row = {col: row[col] for col in columns if row[col]}
            if selected_row:
                selected_rows.append(selected_row)
        return selected_rows
