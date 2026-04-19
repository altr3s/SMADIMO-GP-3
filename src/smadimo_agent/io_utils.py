from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str, limit: int = 60) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-").lower()
    return normalized[:limit] or "run"


def now_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def normalize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.int64)):
        return int(value)
    if isinstance(value, (np.floating, np.float64)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(normalize_for_json(payload), file, ensure_ascii=False, indent=2)


def read_json(path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def copy_dataset_to_workspace(dataset_path: Path, workspace_dir: Path) -> Path:
    destination = ensure_dir(workspace_dir / "data" / "raw") / dataset_path.name
    if destination.resolve() != dataset_path.resolve():
        shutil.copy2(dataset_path, destination)
    return destination


def _detect_csv_delimiter(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as file:
        sample = file.read(8192)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;|\t")
        return dialect.delimiter
    except csv.Error:
        first_line = sample.splitlines()[0] if sample else ""
        candidates = {
            ";": first_line.count(";"),
            ",": first_line.count(","),
            "\t": first_line.count("\t"),
            "|": first_line.count("|"),
        }
        return max(candidates, key=candidates.get)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        delimiter = _detect_csv_delimiter(path)
        return pd.read_csv(
            path,
            sep=delimiter,
            engine="python",
            na_values=["null", "NULL", "None", "none"],
            keep_default_na=True,
        )
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")


def safe_resolve(path: str) -> Path:
    return Path(path).expanduser().resolve()
