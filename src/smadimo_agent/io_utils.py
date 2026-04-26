import csv
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

CSV_DELIMITER_CANDIDATES = ",;|\t"
CSV_NA_VALUES = ["null", "NULL", "None", "none"]


def slugify(value, limit=60):
    normalized = re.sub(r"[^a-zA-Z0-9а-яА-Я_-]+", "-", value.strip())
    normalized = re.sub(r"-{2,}", "-", normalized).strip("-").lower()
    return normalized[:limit] or "run"


def now_run_id():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def normalize_for_json(value):
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


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as file:
        json.dump(normalize_for_json(payload), file, ensure_ascii=False, indent=2)


def read_json(path, default=None):
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _detect_csv_delimiter(path):
    with path.open("r", encoding="utf-8", newline="") as file:
        sample = file.read(8192)

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=CSV_DELIMITER_CANDIDATES)
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


def read_table(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        delimiter = _detect_csv_delimiter(path)
        return pd.read_csv(
            path,
            sep=delimiter,
            engine="python",
            na_values=CSV_NA_VALUES,
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


def write_table(df, path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        return
    raise ValueError(f"Unsupported output format: {path.suffix}")
