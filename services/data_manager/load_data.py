from pathlib import Path
from typing import List, Dict, Union, Optional
from services.data_manager.data_loaders import load_excel, load_csv


class UnsupportedFileTypeError(Exception):
    pass


def _get_extension(file_path: Union[str, Path]) -> str:
    return str(file_path).lower().split(".")[-1]


def list_sheets(file_path: Union[str, Path]) -> Optional[List[str]]:
    ext = _get_extension(file_path)
    if ext in ("xlsx", "xls"):
        return load_excel.list_sheets(file_path)
    elif ext == "csv":
        return None  # CSV has no sheets
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_path}")


def list_columns(
    file_path: Union[str, Path], sheet_name: Optional[str] = None
) -> List[str]:
    ext = _get_extension(file_path)
    if ext in ("xlsx", "xls"):
        return load_excel.list_columns(file_path, sheet_name)
    elif ext == "csv":
        return load_csv.list_columns(file_path)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_path}")


def extract_data(
    file_path: Union[str, Path], sheet_name: Optional[str], columns: List[str]
) -> List[Dict]:
    ext = _get_extension(file_path)
    if ext in ("xlsx", "xls"):
        return load_excel.extract_data(file_path, sheet_name, columns)
    elif ext == "csv":
        return load_csv.extract_data(file_path, columns)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type: {file_path}")
