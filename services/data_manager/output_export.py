import csv
import json
from pathlib import Path
from typing import List, Tuple


def export_results_to_csv(results: List[Tuple[dict, float]], out_path: Path) -> None:
    if not results:
        return

    fieldnames = list(results[0][0].keys()) + ["_score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record, score in results:
            writer.writerow({**record, "_score": round(score, 4)})


def export_results_to_txt(results: List[Tuple[dict, float]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for record, score in results:
            f.write(f"Score: {round(score, 4)}\n")
            for key, val in record.items():
                f.write(f"{key}: {val}\n")
            f.write("\n")


def export_results_to_json(results: List[Tuple[dict, float]], out_path: Path) -> None:
    json_data = [{**record, "_score": round(score, 4)} for record, score in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
