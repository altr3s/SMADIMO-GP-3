import itertools
import json
import math
import pickle
import random
import os
import shutil
from typing import Any, List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.metrics import (
    accuracy_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from langchain.tools import ToolRuntime, tool


def write_json(path, data):
    def fix(v):
        if isinstance(v, dict):
            return {str(k): fix(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        if hasattr(v, "__fspath__"): #
            return str(v)
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return v
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fix(data), f, ensure_ascii=False, indent=2)


def read_json(path, default=None):
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        if default is not None:
            return default
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def read_csv(path):
    path = os.path.abspath(os.path.expanduser(str(path)))
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    if not first.strip():
        return pd.DataFrame()
    sem, com = first.count(";"), first.count(",")
    sep = ";" if sem > com else ","
    return pd.read_csv(path, sep=sep)


def infer_roles(df, target=None):
    roles = {}
    for col in df.columns:
        if target and col == target:
            roles[col] = "target"
            continue
        s = df[col]
        non_null = s.dropna()
        unique_ratio = non_null.nunique() / max(len(non_null), 1)
        name = col.lower()

        if pd.api.types.is_bool_dtype(s):
            roles[col] = "categorical"
        elif pd.api.types.is_datetime64_any_dtype(s):
            roles[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(s):
            if unique_ratio > 0.98 and any(t in name for t in ("id", "uuid", "code", "number")):
                roles[col] = "identifier"
            else:
                roles[col] = "numeric"
        else:
            text = non_null.astype(str)
            avg_len = text.str.len().mean() if len(text) else 0
            avg_words = text.str.split().str.len().mean() if len(text) else 0
            parsed = pd.to_datetime(text, errors="coerce", format="mixed")
            parse_ratio = parsed.notna().mean() if len(text) else 0

            if unique_ratio > 0.98 and any(t in name for t in ("id", "uuid", "code", "number")):
                roles[col] = "identifier"
            elif parse_ratio >= 0.8 and avg_len >= 6:
                roles[col] = "datetime_candidate"
            elif avg_len >= 35 or avg_words >= 4 or unique_ratio >= 0.5:
                roles[col] = "text"
            else:
                roles[col] = "categorical"
    return roles


SUPPORTED_MODELS = {
    "classification": [
        "logistic_regression", "random_forest_classifier",
        "gradient_boosting_classifier", "linear_svc", "k_neighbors_classifier",
    ],
    "regression": [
        "ridge_regression", "sgd_regressor", "random_forest_regressor",
        "gradient_boosting_regressor", "k_neighbors_regressor",
    ],
    "clustering": ["kmeans", "agglomerative_clustering", "dbscan"],
}

DEFAULT_SPACES = {
    "ridge_regression": {"alpha": [0.1, 1.0, 10.0, 100.0]},
    "sgd_regressor": {"alpha": [0.0001, 0.001]},
    "random_forest_regressor": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "gradient_boosting_regressor": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "k_neighbors_regressor": {"n_neighbors": [5, 9, 11, 15]},
    "logistic_regression": {"C": [0.1, 1.0, 3.0]},
    "random_forest_classifier": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
    "gradient_boosting_classifier": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]},
    "linear_svc": {"C": [0.1, 1.0, 3.0]},
    "k_neighbors_classifier": {"n_neighbors": [5, 9, 11, 15]},
    "kmeans": {"n_clusters": [2, 3, 4, 5]},
    "agglomerative_clustering": {"n_clusters": [2, 3, 4, 5]},
    "dbscan": {"eps": [0.5, 0.7, 1.0], "min_samples": [5, 10]},
}


def build_model(task_type, model_name, params=None):
    if task_type == "classification":
        models = {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "random_forest_classifier": RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1),
            "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
            "linear_svc": LinearSVC(),
            "k_neighbors_classifier": KNeighborsClassifier(n_neighbors=11),
        }
    elif task_type == "regression":
        models = {
            "ridge_regression": Ridge(alpha=1.0),
            "sgd_regressor": SGDRegressor(random_state=42, max_iter=2000, penalty="elasticnet", early_stopping=True),
            "random_forest_regressor": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
            "k_neighbors_regressor": KNeighborsRegressor(n_neighbors=11),
        }
    else:
        models = {
            "kmeans": KMeans(n_clusters=3, random_state=42, n_init=20),
            "agglomerative_clustering": AgglomerativeClustering(n_clusters=3),
            "dbscan": DBSCAN(eps=0.7, min_samples=10),
        }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    est = models[model_name]
    if params:
        valid = est.get_params()
        safe = {k: v for k, v in params.items() if k in valid}
        if safe:
            est.set_params(**safe)
    return est


def prepare_df(df, target=None):
    roles = infer_roles(df, target)
    drop = [
        c for c, r in roles.items()
        if r in {"identifier", "text", "datetime", "datetime_candidate"} and c != target
    ]
    return df.drop(columns=drop, errors="ignore").copy()


def _rank_value(row: dict) -> float:
    if row.get("rank_score") is not None:
        try:
            return float(row["rank_score"])
        except (TypeError, ValueError):
            pass
    try:
        return float(row.get("score", float("-inf")))
    except (TypeError, ValueError):
        return float("-inf")


def build_preprocessor(df, target):
    num_cols = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [
        c for c in df.columns
        if c != target
        and not pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    if not transformers:
        raise ValueError("No usable columns for modeling after preprocessing.")
    return ColumnTransformer(transformers, remainder="drop")


def _fmt_ml_params(params: Optional[dict]) -> str:
    p = dict(params or {})
    if not p:
        return "{} (значения по умолчанию sklearn для этой модели)"
    return json.dumps(p, ensure_ascii=False, sort_keys=True, default=str)


def _ml_log(lines: str | List[str]) -> None:
    if isinstance(lines, str):
        lines = lines.splitlines() or [lines]
    for raw in lines:
        print(f"[ML] {raw}")


def fit_supervised(
    model_name,
    task_type,
    train_df,
    val_df,
    target,
    params=None,
    *,
    run_label: str = "Обучение",
):
    feat_train = prepare_df(train_df, target)
    feat_val = prepare_df(val_df, target)
    params = dict(params or {})
    n_features = int(feat_train.drop(columns=[target]).shape[1])

    preprocessor = build_preprocessor(feat_train, target)
    estimator = build_model(task_type, model_name, params)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", estimator)])

    _ml_log([
        f"{run_label}: «{model_name}» ({task_type})",
        f"  таргет: {target}; гиперпараметры: {_fmt_ml_params(params)}",
        f"  выборка после prepare_df: train={len(feat_train)} строк, val={len(feat_val)} строк, признаков={n_features}",
    ])

    y_train = feat_train[target].to_numpy()
    y_val = feat_val[target].to_numpy()
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)

    pipeline.fit(feat_train.drop(columns=[target]), y_train)
    preds = pipeline.predict(feat_val.drop(columns=[target]))

    if task_type == "classification":
        avg = "binary" if len(np.unique(y_val)) == 2 else "weighted"
        metrics = {
            "accuracy": float(accuracy_score(y_val, preds)),
            "precision": float(precision_score(y_val, preds, average=avg, zero_division=0)),
            "recall": float(recall_score(y_val, preds, average=avg, zero_division=0)),
            "f1": float(f1_score(y_val, preds, average=avg, zero_division=0)),
            "roc_auc": None,
        }
        if len(np.unique(y_val)) == 2 and hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(feat_val.drop(columns=[target]))[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_val, proba))
            except Exception:
                metrics["roc_auc"] = None
        rank_score = float(metrics.get("roc_auc") or metrics.get("f1") or -math.inf)
    else:
        mse = float(mean_squared_error(y_val, preds))
        metrics = {
            "rmse": float(math.sqrt(mse)),
            "mae": float(mean_absolute_error(y_val, preds)),
        }
        rank_score = -metrics["rmse"]

    _ml_log(f"  готово «{model_name}». Метрики на валидации: {json.dumps(metrics, ensure_ascii=False, default=str)}")

    return {"pipeline": pipeline, "label_encoder": le, "metrics": metrics, "rank_score": rank_score}


def fit_clustering(model_name, train_df, params=None, *, run_label: str = "Обучение"):
    feat = prepare_df(train_df)
    feat["_dummy"] = 0
    preprocessor = build_preprocessor(feat, "_dummy")
    params = dict(params or {})
    _ml_log([
        f"{run_label}: «{model_name}» (кластеризация)",
        f"  гиперпараметры: {_fmt_ml_params(params)}",
        f"  строк после prepare_df: {len(feat)}",
    ])
    X = np.asarray(preprocessor.fit_transform(feat))
    _ml_log(f"  размерность признаков после препроцессинга: {X.shape[0]}×{X.shape[1]}")

    estimator = build_model("clustering", model_name, params)
    if hasattr(estimator, "fit_predict"):
        labels = estimator.fit_predict(X)
    else:
        estimator.fit(X)
        labels = estimator.predict(X)

    unique = set(labels.tolist())
    if len(unique) > 1:
        metrics = {
            "silhouette": float(silhouette_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        }
        rank_score = metrics["silhouette"]
    else:
        metrics = {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
        rank_score = -math.inf

    _ml_log(f"  готово «{model_name}». Метрики: {json.dumps(metrics, ensure_ascii=False, default=str)}")

    return {"preprocessor": preprocessor, "model": estimator, "metrics": metrics, "rank_score": rank_score}


def active_dataset(runtime):
    ws = runtime.state["workspace_dir"]
    candidates = [
        os.path.join(ws, "data", "featured", "featured_dataset.csv"),
        os.path.join(ws, "data", "cleaned", "cleaned_dataset.csv"),
        runtime.state["dataset_path"],
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return runtime.state["dataset_path"]


def get_goal(runtime):
    ws = runtime.state["workspace_dir"]
    return read_json(os.path.join(ws, "analysis", "modeling_goal.json"), default={})


class FeatureSpec(BaseModel):
    name: str
    kind: Literal[
        "ratio", "difference", "product", "sum",
        "text_length", "text_word_count", "date_part",
        "is_missing", "category_frequency",
    ]
    source_columns: List[str]
    date_part: Optional[Literal["year", "month", "day", "dayofweek", "quarter"]] = None
    denominator_guard: float = 1e-6


@tool
def profile_dataset(runtime: ToolRuntime) -> str:
    """Profile the dataset and save column statistics."""
    ws = runtime.state["workspace_dir"]
    df = read_csv(runtime.state["dataset_path"])
    roles = infer_roles(df)

    by_role = {}
    for col, role in roles.items():
        by_role.setdefault(role, []).append(col)

    columns_info = []
    for col in df.columns:
        s = df[col]
        columns_info.append({
            "name": col,
            "dtype": str(s.dtype),
            "role": roles[col],
            "missing": int(s.isna().sum()),
            "unique": int(s.nunique()),
            "samples": s.dropna().astype(str).head(3).tolist(),
        })

    write_json(os.path.join(ws, "analysis", "dataset_profile.json"), {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "columns": columns_info,
        "roles": roles,
        "columns_by_role": by_role,
        "duplicates": int(df.duplicated().sum()),
        "candidate_targets": {
            "regression": [c for c in df.columns if roles[c] == "numeric" and df[c].nunique() > 10][:10],
            "classification": [c for c in df.columns if roles[c] == "categorical" and 2 <= df[c].nunique() <= 25][:10],
        },
    })
    return f"Dataset profiled: {len(df)} rows, {len(df.columns)} columns."


@tool
def get_dataset_schema(runtime: ToolRuntime) -> str:
    """Get schema of the current dataset (cleaned/featured if available)."""
    ws = runtime.state["workspace_dir"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")
    df = read_csv(active_dataset(runtime))
    roles = infer_roles(df, target)

    by_role = {}
    for col, role in roles.items():
        by_role.setdefault(role, []).append(col)

    parts = [f"rows={len(df)}, cols={len(df.columns)}", f"target={target or 'not set'}"]
    for role in ["numeric", "categorical", "text", "datetime", "identifier"]:
        if role in by_role:
            parts.append(f"{role}: {', '.join(by_role[role])}")
    summary = " | ".join(parts)

    write_json(os.path.join(ws, "analysis", "schema_snapshot.json"), {
        "roles": roles, "by_role": by_role, "summary": summary,
    })
    return summary


@tool
def set_modeling_goal(
    target_column: str,
    task_type: Literal["classification", "regression", "clustering"],
    reasoning: str,
    runtime: ToolRuntime,
) -> str:
    """Set the target column and ML task type."""
    ws = runtime.state["workspace_dir"]
    df = read_csv(active_dataset(runtime))

    if task_type != "clustering" and target_column not in df.columns:
        raise ValueError(f"Column '{target_column}' not found in dataset.")

    goal = {
        "target_column": target_column if task_type != "clustering" else None,
        "task_type": task_type,
        "reasoning": reasoning,
    }
    write_json(os.path.join(ws, "analysis", "modeling_goal.json"), goal)
    return f"Goal set: task_type={task_type}, target={goal['target_column']}"


@tool
def analyze_distributions(runtime: ToolRuntime) -> str:
    """Analyze value distributions for all columns."""
    ws = runtime.state["workspace_dir"]
    df = read_csv(active_dataset(runtime))

    report = {}
    for col in df.columns:
        s = df[col]
        counts = s.value_counts(dropna=False).head(20)
        total = max(len(s), 1)
        top = [{"value": str(v), "count": int(c), "share": round(float(c / total), 4)} for v, c in counts.items()]
        report[col] = {"dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique()), "top_values": top}

    write_json(os.path.join(ws, "analysis", "distribution_report.json"), {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": report,
    })
    return f"Distribution report saved for {len(df.columns)} columns."


@tool
def clean_dataset(
    drop_columns: str = "",
    drop_rows: str = "",
    numeric_imputation: Literal["median", "mean", "none"] = "median",
    categorical_imputation: Literal["mode", "constant", "none"] = "mode",
    outlier_strategy: Literal["iqr_clip", "none"] = "iqr_clip",
    cleaning_reasoning: str = "",
    runtime: ToolRuntime = None,
) -> str:
    """Clean the raw dataset: drop columns/rows, fill nulls, clip outliers."""
    ws = runtime.state["workspace_dir"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")
    df = read_csv(runtime.state["dataset_path"]).copy()
    rows_before = len(df)


    if drop_columns:
        dc = drop_columns.strip().replace("'", '"')
        if dc.startswith("["):
            cols_to_drop = json.loads(dc)
        else:
            cols_to_drop = [c.strip() for c in drop_columns.split(",") if c.strip()]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    if drop_rows:
        dr = drop_rows.strip()
        if dr.startswith("["):
            rules = json.loads(dr.replace("'", '"'))
            for rule in (rules if isinstance(rules, list) else []):
                col, val = rule.get("column"), rule.get("value")
                if col and col in df.columns:
                    df = df[df[col].astype(str) != str(val)].copy()

    df = df.drop_duplicates()
    if target and target in df.columns:
        df = df[df[target].notna()].copy()

    roles = infer_roles(df, target)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    cat_cols = [c for c, r in roles.items() if r == "categorical" and c != target]

    if numeric_imputation == "median":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].median())
    elif numeric_imputation == "mean":
        for c in num_cols:
            df[c] = df[c].fillna(df[c].mean())

    if categorical_imputation == "mode":
        for c in cat_cols:
            mode = df[c].mode(dropna=True)
            if not mode.empty:
                df[c] = df[c].fillna(mode.iloc[0])

    if outlier_strategy == "iqr_clip":
        for c in num_cols:
            q1, q3 = df[c].quantile(0.25), df[c].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                df[c] = df[c].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)

    out = os.path.join(ws, "data", "cleaned", "cleaned_dataset.csv")
    df.to_csv(out, index=False)
    write_json(os.path.join(ws, "analysis", "cleaning_report.json"), {
        "rows_before": rows_before, "rows_after": len(df), "cleaning_reasoning": cleaning_reasoning,
    })
    return f"Cleaned: {rows_before} -> {len(df)} rows. Saved to {out}."


@tool
def run_eda(runtime: ToolRuntime) -> str:
    """Run exploratory data analysis on the current dataset."""
    ws = runtime.state["workspace_dir"]
    goal = get_goal(runtime)
    target = runtime.state.get("target_column") or goal.get("target_column")
    task_type = runtime.state.get("task_type") or goal.get("task_type")
    df = read_csv(active_dataset(runtime))

    corr = {}
    if target and target in df.select_dtypes("number").columns:
        corr = (
            df.select_dtypes("number").corr()[target]
            .drop(labels=[target])
            .sort_values(key=abs, ascending=False)
            .head(10)
            .to_dict()
        )

    class_balance = {}
    if task_type == "classification" and target:
        class_balance = df[target].value_counts(dropna=False).to_dict()

    outliers = {}
    for c in df.select_dtypes("number").columns:
        if c == target:
            continue
        s = df[c].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers[c] = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0

    leakage = []
    if target and pd.api.types.is_numeric_dtype(df.get(target, pd.Series(dtype=float))):
        num = df.select_dtypes("number")
        if target in num.columns:
            c = num.corr()[target].drop(labels=[target])
            leakage = c[abs(c) > 0.98].index.tolist()

    eda = {
        "shape": {"rows": len(df), "cols": len(df.columns)},
        "missing_by_column": df.isna().sum().to_dict(),
        "outliers_by_column": outliers,
        "target_distribution": class_balance,
        "top_correlations_with_target": corr,
        "leakage_candidates": leakage,
    }
    write_json(os.path.join(ws, "analysis", "eda_report.json"), eda)
    with open(os.path.join(ws, "analysis", "eda_report.md"), "w", encoding="utf-8") as _f:
        _f.write(
            f"# EDA\n- rows: {len(df)}\n- target: {target}\n- leakage: {leakage or 'none'}\n",
        )
    return f"EDA done. Leakage candidates: {leakage or 'none'}."


@tool
def engineer_features(feature_specs: List[FeatureSpec], note: str, runtime: ToolRuntime) -> str:
    """Create new features on top of the cleaned dataset."""
    ws = runtime.state["workspace_dir"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")

    cleaned = os.path.join(ws, "data", "cleaned", "cleaned_dataset.csv")
    src = cleaned if os.path.isfile(cleaned) else runtime.state["dataset_path"]
    df = read_csv(src).copy()

    created = []
    for spec in feature_specs:
        if spec.name in df.columns:
            continue
        if any(c not in df.columns for c in spec.source_columns):
            continue
        if any(c == target for c in spec.source_columns):
            continue
        if spec.kind == "ratio":
            a, b = spec.source_columns[:2]
            df[spec.name] = df[a] / (df[b].replace(0, np.nan) + spec.denominator_guard)
        elif spec.kind == "difference":
            a, b = spec.source_columns[:2]
            df[spec.name] = df[a] - df[b]
        elif spec.kind == "product":
            a, b = spec.source_columns[:2]
            df[spec.name] = df[a] * df[b]
        elif spec.kind == "sum":
            a, b = spec.source_columns[:2]
            df[spec.name] = df[a] + df[b]
        elif spec.kind == "text_length":
            df[spec.name] = df[spec.source_columns[0]].fillna("").astype(str).str.len()
        elif spec.kind == "text_word_count":
            df[spec.name] = df[spec.source_columns[0]].fillna("").astype(str).str.split().str.len()
        elif spec.kind == "date_part":
            parsed = pd.to_datetime(df[spec.source_columns[0]], errors="coerce", format="mixed")
            df[spec.name] = getattr(parsed.dt, spec.date_part)
        elif spec.kind == "is_missing":
            df[spec.name] = df[spec.source_columns[0]].isna().astype(int)
        elif spec.kind == "category_frequency":
            freqs = df[spec.source_columns[0]].value_counts(normalize=True, dropna=False)
            df[spec.name] = df[spec.source_columns[0]].map(freqs)
        else:
            continue
        created.append(spec.name)

    if not created:
        raise ValueError("Could not create any features. Check that source_columns exist in the dataset.")

    out = os.path.join(ws, "data", "featured", "featured_dataset.csv")
    df.to_csv(out, index=False)

    roles = infer_roles(df, target)
    by_role = {}
    for col, role in roles.items():
        by_role.setdefault(role, []).append(col)
    summary = f"rows={len(df)}, cols={len(df.columns)}, target={target}"

    write_json(os.path.join(ws, "analysis", "feature_report.json"), {"note": note, "created": created})
    write_json(os.path.join(ws, "analysis", "schema_snapshot.json"), {
        "roles": roles, "by_role": by_role, "summary": summary,
    })
    return f"Created {len(created)} features: {created}. Saved to {out}."


@tool
def select_candidate_models(model_names: List[str], reasoning: str, runtime: ToolRuntime) -> str:
    """Register selected ML models for the current task."""
    ws = runtime.state["workspace_dir"]
    task_type = runtime.state.get("task_type") or get_goal(runtime).get("task_type")
    if not task_type:
        raise ValueError("Task type not set. Run set_modeling_goal first.")

    allowed = SUPPORTED_MODELS[task_type]
    aliases = {
        "ridge": "ridge_regression", "linear_regression": "ridge_regression",
        "logisticregression": "logistic_regression",
        "randomforestclassifier": "random_forest_classifier",
        "randomforestregressor": "random_forest_regressor",
        "gradientboostingclassifier": "gradient_boosting_classifier",
        "gradientboostingregressor": "gradient_boosting_regressor",
        "kneighborsclassifier": "k_neighbors_classifier",
        "kneighborsregressor": "k_neighbors_regressor",
    }

    selected = []
    for name in model_names:
        normalized = name if name in allowed else aliases.get(name.lower().replace("_", ""))
        if normalized and normalized not in selected:
            selected.append(normalized)

    fallbacks = {
        "classification": ["logistic_regression", "random_forest_classifier"],
        "regression": ["ridge_regression", "random_forest_regressor"],
        "clustering": ["kmeans", "agglomerative_clustering"],
    }
    for fb in fallbacks[task_type]:
        if len(selected) >= 2:
            break
        if fb not in selected:
            selected.append(fb)

    write_json(os.path.join(ws, "modeling", "model_plan.json"), {
        "task_type": task_type, "selected_models": selected, "reasoning": reasoning,
    })
    return f"Selected models: {selected}"


@tool
def prepare_splits(
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    runtime: ToolRuntime = None,
) -> str:
    """Split dataset into train/val/test."""
    ws = runtime.state["workspace_dir"]
    goal = get_goal(runtime)
    task_type = runtime.state.get("task_type") or goal.get("task_type")
    target = runtime.state.get("target_column") or goal.get("target_column")
    df = read_csv(active_dataset(runtime))

    strat = None
    if task_type == "classification" and stratify and target and target in df.columns:
        if df[target].value_counts().min() >= 2:
            strat = df[target]

    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=strat)
    val_relative = val_size / (1 - test_size)

    strat2 = None
    if strat is not None and train[target].value_counts().min() >= 2:
        strat2 = train[target]
    train, val = train_test_split(train, test_size=val_relative, random_state=42, stratify=strat2)

    split_dir = os.path.join(ws, "modeling", "splits")
    os.makedirs(split_dir, exist_ok=True)
    train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(split_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(split_dir, "test.csv"), index=False)
    write_json(os.path.join(split_dir, "split_manifest.json"), {
        "train": len(train), "val": len(val), "test": len(test),
    })
    return f"Splits ready: train={len(train)}, val={len(val)}, test={len(test)}."


@tool
def tune_models(
    n_iter: int = 4,
    model_spaces: str = "",
    reasoning: str = "",
    runtime: ToolRuntime = None,
) -> str:
    """Tune hyperparameters for selected models."""
    ws = runtime.state["workspace_dir"]
    plan = read_json(os.path.join(ws, "modeling", "model_plan.json"))
    task_type = plan["task_type"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")
    selected = plan["selected_models"]

    split_dir = os.path.join(ws, "modeling", "splits")
    train_df = read_csv(os.path.join(split_dir, "train.csv"))
    val_df = read_csv(os.path.join(split_dir, "val.csv"))

    spaces = {}
    if model_spaces:
        text = model_spaces
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1:
            text = text[start : end + 1]
        try:
            spaces = json.loads(text)
        except json.JSONDecodeError:
            spaces = {}

    n_iter = min(max(int(n_iter), 1), 8)
    best_params = {}
    leaderboard = []

    _ml_log(f"Подбор гиперпараметров: модели {selected}, до {n_iter} комбинаций на модель (если сетка большая — случайная подвыборка).")

    for model_name in selected:
        space = spaces.get(model_name) or DEFAULT_SPACES.get(model_name, {})

        if space:
            keys = sorted(space.keys())
            combos = [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]
            if len(combos) > n_iter:
                combos = random.Random(42).sample(combos, n_iter)
        else:
            combos = [{}]

        results = []
        for params in combos:
            try:
                if task_type == "clustering":
                    r = fit_clustering(model_name, train_df, params, run_label="Подбор гиперпараметров")
                else:
                    r = fit_supervised(
                        model_name, task_type, train_df, val_df, target, params,
                        run_label="Подбор гиперпараметров",
                    )
                results.append({"model": model_name, "params": params, "metrics": r["metrics"], "rank_score": r["rank_score"]})
            except Exception as e:
                results.append({"model": model_name, "params": params, "error": str(e), "rank_score": -math.inf})

        results.sort(key=_rank_value, reverse=True)
        best_params[model_name] = results[0].get("params", {}) if _rank_value(results[0]) > -math.inf else {}
        leaderboard.extend(results)

    leaderboard.sort(key=_rank_value, reverse=True)
    write_json(os.path.join(ws, "modeling", "hyperparameter_tuning.json"), {
        "task_type": task_type, "reasoning": reasoning,
        "best_params_by_model": best_params, "leaderboard": leaderboard,
    })
    lines = [
        f"Подбор завершён для {len(selected)} модель(ей). Лучшие параметры по валидации:",
    ]
    for name in selected:
        lines.append(f"  • {name}: {_fmt_ml_params(best_params.get(name, {}))}")
    lines.append("Детали всех прогонов записаны в modeling/hyperparameter_tuning.json.")
    return "\n".join(lines)


@tool
def train_models(runtime: ToolRuntime) -> str:
    """Train all selected models and rank them on the validation split."""
    ws = runtime.state["workspace_dir"]
    plan = read_json(os.path.join(ws, "modeling", "model_plan.json"))
    task_type = plan["task_type"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")
    selected = plan["selected_models"]

    tuning = read_json(os.path.join(ws, "modeling", "hyperparameter_tuning.json"), default={})
    best_params = tuning.get("best_params_by_model", {})

    split_dir = os.path.join(ws, "modeling", "splits")
    train_df = read_csv(os.path.join(split_dir, "train.csv"))
    val_df = read_csv(os.path.join(split_dir, "val.csv"))

    models_dir = os.path.join(ws, "models")
    os.makedirs(models_dir, exist_ok=True)

    _ml_log(
        f"Финальное обучение и сохранение: {len(selected)} модель(ей) с лучшими параметрами из подбора. "
        f"Таргет: {target}.",
    )

    leaderboard = []
    for model_name in selected:
        params = best_params.get(model_name, {})
        if task_type == "clustering":
            r = fit_clustering(model_name, train_df, params, run_label="Финальное обучение")
            bundle = {"preprocessor": r["preprocessor"], "model": r["model"], "params": params}
        else:
            r = fit_supervised(
                model_name, task_type, train_df, val_df, target, params,
                run_label="Финальное обучение",
            )
            bundle = {"pipeline": r["pipeline"], "label_encoder": r["label_encoder"], "params": params}

        model_path = os.path.join(models_dir, f"{model_name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(bundle, f)

        leaderboard.append({
            "model_name": model_name,
            "metrics": r["metrics"],
            "rank_score": r["rank_score"],
            "model_path": model_path,
            "params": params,
        })

    leaderboard.sort(key=_rank_value, reverse=True)
    best = leaderboard[0]["model_name"] if leaderboard else None
    write_json(os.path.join(ws, "modeling", "leaderboard.json"), {
        "task_type": task_type, "leaderboard": leaderboard, "best_model_name": best,
    })
    lines = [
        f"Обучено моделей: {len(leaderboard)}. Лидер по валидации: «{best}».",
        "По каждой модели: имя, использованные гиперпараметры, метрики на val, файл:",
    ]
    for row in leaderboard:
        lines.append(
            f"  • {row['model_name']}: параметры {_fmt_ml_params(row.get('params'))}; "
            f"метрики: {json.dumps(row['metrics'], ensure_ascii=False, default=str)}; "
            f"файл: {row['model_path']}",
        )
    return "\n".join(lines)


@tool
def evaluate_models(runtime: ToolRuntime) -> str:
    """Evaluate the best model on the hold-out test split."""
    ws = runtime.state["workspace_dir"]
    lb = read_json(os.path.join(ws, "modeling", "leaderboard.json"))
    task_type = lb["task_type"]
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")

    best = lb["leaderboard"][0]
    model_name = best["model_name"]
    params = best.get("params", {})

    split_dir = os.path.join(ws, "modeling", "splits")
    combined = pd.concat([read_csv(os.path.join(split_dir, "train.csv")), read_csv(os.path.join(split_dir, "val.csv"))], ignore_index=True)
    test_df = read_csv(os.path.join(split_dir, "test.csv"))

    if task_type == "clustering":
        r = fit_clustering(model_name, combined, params, run_label="Оценка (переобучение на train+val)")
        bundle = {"preprocessor": r["preprocessor"], "model": r["model"]}
    else:
        r = fit_supervised(
            model_name, task_type, combined, test_df, target, params,
            run_label="Оценка (переобучение на train+val, метрики на test)",
        )
        bundle = {"pipeline": r["pipeline"], "label_encoder": r["label_encoder"]}

    model_path = os.path.join(ws, "models", "best_current_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    write_json(os.path.join(ws, "modeling", "evaluation.json"), {
        "best_model_name": model_name,
        "task_type": task_type,
        "target_column": target,
        "test_metrics": r["metrics"],
        "validation_metrics": best["metrics"],
        "selection_rank_score": r["rank_score"],
        "current_best_model_path": model_path,
        "best_params": params,
    })
    return f"Evaluated {model_name}. Test metrics: {r['metrics']}"


@tool
def load_long_term_memory(runtime: ToolRuntime) -> str:
    """Load best historical result from long-term memory."""
    ws = runtime.state["workspace_dir"]
    history = read_json(os.path.join(os.path.dirname(ws), "memory", "best_registry.json"), default={"best_run": None})
    best = history.get("best_run")
    if not best:
        return "Memory is empty — this is the first run."
    rs = best.get("selection_rank_score", best.get("selection_score"))
    return f"Historical best: {best.get('best_model_name')}, rank_score={rs}."


@tool
def load_best_model_from_memory(runtime: ToolRuntime) -> str:
    """Check if the historical best model is compatible with the current run."""
    ws = runtime.state["workspace_dir"]
    memory_dir = os.path.join(os.path.dirname(ws), "memory")
    history = read_json(os.path.join(memory_dir, "best_registry.json"), default={"best_run": None})
    best = history.get("best_run")

    if not best:
        write_json(os.path.join(ws, "modeling", "memory_baseline.json"), {"status": "empty"})
        return "No historical model to compare with."

    task_type = runtime.state.get("task_type") or get_goal(runtime).get("task_type")
    target = runtime.state.get("target_column") or get_goal(runtime).get("target_column")
    compatible = best.get("task_type") == task_type and best.get("target_column") == target

    write_json(os.path.join(ws, "modeling", "memory_baseline.json"), {
        "status": "loaded" if compatible else "incompatible",
        "historical_best": best,
        "compatible": compatible,
    })
    if compatible:
        rs = best.get("selection_rank_score", best.get("selection_score"))
        return f"Historical model is compatible. Rank score to beat: {rs}."
    return "Historical model exists but is for a different task/target."


@tool
def save_best_model(runtime: ToolRuntime) -> str:
    """Save current model to long-term memory if it beats the historical best."""
    ws = runtime.state["workspace_dir"]
    evaluation = read_json(os.path.join(ws, "modeling", "evaluation.json"))
    memory_dir = os.path.join(os.path.dirname(ws), "memory")
    history = read_json(os.path.join(memory_dir, "best_registry.json"), default={"best_run": None})
    prev = history.get("best_run")

    current_rank = float(
        evaluation.get("selection_rank_score", evaluation.get("selection_score", float("-inf"))),
    )
    prev_rank = None if prev is None else float(
        prev.get("selection_rank_score", prev.get("selection_score", float("-inf"))),
    )
    if prev is None or current_rank > prev_rank:
        shutil.copy2(os.path.join(ws, "models", "best_current_model.pkl"), os.path.join(memory_dir, "best_model.pkl"))
        write_json(os.path.join(memory_dir, "best_registry.json"), {"best_run": {
            "best_model_name": evaluation["best_model_name"],
            "selection_rank_score": current_rank,
            "task_type": evaluation["task_type"],
            "target_column": evaluation["target_column"],
        }})
        return "New best model saved to memory!"
    return (
        f"Current rank_score ({current_rank:.4f}) didn't beat historical ({float(prev_rank):.4f})."
        if prev_rank is not None
        else "Could not compare with historical best."
    )


def _json_clip(obj: Any, limit: int = 4000) -> str:
    s = json.dumps(obj, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return s
    return s[: max(0, limit - 3)] + "..."


@tool
def collect_pipeline_highlights(runtime: ToolRuntime) -> str:
    """Собрать сжатые факты из артефактов прогона для бизнес-отчёта (без повторного чтения файлов вручную)."""
    ws = runtime.state["workspace_dir"]

    parts: list[str] = []
    parts.append("=== Бизнес-задача (из запуска) ===")
    parts.append(str(runtime.state.get("business_task", "")).strip() or "(не задана)")
    parts.append("")
    parts.append(f"=== Датасет ===\n{runtime.state.get('dataset_path', '')}")

    goal = read_json(os.path.join(ws, "analysis", "modeling_goal.json"), default={})
    parts.append("\n=== Цель моделирования ===")
    parts.append(_json_clip(goal, 2500))

    cleaning = read_json(os.path.join(ws, "analysis", "cleaning_report.json"), default={})
    parts.append("\n=== Очистка данных ===")
    parts.append(_json_clip(cleaning, 2000))

    eda = read_json(os.path.join(ws, "analysis", "eda_report.json"), default={})
    eda_lite = {
        "shape": eda.get("shape"),
        "leakage_candidates": eda.get("leakage_candidates"),
        "top_correlations_with_target": eda.get("top_correlations_with_target"),
        "missing_by_column": dict(list((eda.get("missing_by_column") or {}).items())[:25]),
    }
    parts.append("\n=== EDA (фрагмент) ===")
    parts.append(_json_clip(eda_lite, 3500))

    features = read_json(os.path.join(ws, "analysis", "feature_report.json"), default={})
    parts.append("\n=== Признаки ===")
    parts.append(_json_clip(features, 2000))

    schema = read_json(os.path.join(ws, "analysis", "schema_snapshot.json"), default={})
    parts.append("\n=== Схема (сводка) ===")
    parts.append((schema.get("summary") or _json_clip(schema, 2000))[:2500])

    plan = read_json(os.path.join(ws, "modeling", "model_plan.json"), default={})
    parts.append("\n=== План моделей ===")
    parts.append(_json_clip(plan, 2000))

    tuning = read_json(os.path.join(ws, "modeling", "hyperparameter_tuning.json"), default={})
    tuning_lite = {
        "task_type": tuning.get("task_type"),
        "reasoning": tuning.get("reasoning"),
        "best_params_by_model": tuning.get("best_params_by_model"),
        "leaderboard_head": (tuning.get("leaderboard") or [])[:12],
    }
    parts.append("\n=== Подбор гиперпараметров (фрагмент) ===")
    parts.append(_json_clip(tuning_lite, 3200))

    lb = read_json(os.path.join(ws, "modeling", "leaderboard.json"), default={})
    parts.append("\n=== Лидерборд ===")
    parts.append(_json_clip(lb, 3200))

    ev = read_json(os.path.join(ws, "modeling", "evaluation.json"), default={})
    parts.append("\n=== Оценка на test (best) ===")
    parts.append(_json_clip(ev, 2800))

    split_m = read_json(os.path.join(ws, "modeling", "splits", "split_manifest.json"), default={})
    parts.append("\n=== Сплиты ===")
    parts.append(_json_clip(split_m, 800))

    mem = read_json(os.path.join(os.path.dirname(ws), "memory", "best_registry.json"), default={})
    parts.append("\n=== Долговременная память (если есть) ===")
    parts.append(_json_clip(mem, 1500))

    text = "\n".join(parts)
    max_out = 10000
    if len(text) > max_out:
        text = text[: max_out - 80] + "\n\n[… вывод обрезан по длине; опирайся на имеющееся …]"
    return text


@tool
def write_business_report(markdown_report: str, runtime: ToolRuntime) -> str:
    """Сохранить бизнес-интерпретацию результатов пайплайна (markdown для ЛПР)."""
    ws = runtime.state["workspace_dir"]
    body = (markdown_report or "").strip()
    if len(body) < 120:
        raise ValueError("Отчёт слишком короткий: опиши выводы для бизнеса подробнее (минимум ~120 символов).")

    out_md = os.path.join(ws, "reports", "business_interpretation.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(body)

    out_json = os.path.join(ws, "reports", "business_interpretation.json")
    write_json(out_json, {
        "chars": len(body),
        "business_task": runtime.state.get("business_task"),
        "markdown_path": out_md,
    })
    return f"Бизнес-отчёт сохранён: {out_md}"


@tool
def write_report(runtime: ToolRuntime) -> str:
    """Write a final markdown report for the current run."""
    ws = runtime.state["workspace_dir"]

    goal = read_json(os.path.join(ws, "analysis", "modeling_goal.json"), default={})
    profile = read_json(os.path.join(ws, "analysis", "dataset_profile.json"), default={})
    cleaning = read_json(os.path.join(ws, "analysis", "cleaning_report.json"), default={})
    eda = read_json(os.path.join(ws, "analysis", "eda_report.json"), default={})
    features = read_json(os.path.join(ws, "analysis", "feature_report.json"), default={})
    model_plan = read_json(os.path.join(ws, "modeling", "model_plan.json"), default={})
    evaluation = read_json(os.path.join(ws, "modeling", "evaluation.json"), default={})
    history = read_json(os.path.join(os.path.dirname(ws), "memory", "best_registry.json"), default={"best_run": None})

    report = "\n".join([
        "# Итоговый отчет по запуску агента",
        "",
        f"**Бизнес-задача:** {runtime.state['business_task']}",
        f"**Датасет:** {runtime.state['dataset_path']}",
        f"**Тип задачи:** {goal.get('task_type')}",
        f"**Целевая переменная:** {goal.get('target_column')}",
        "",
        "## Анализ датасета",
        f"- Размер: {profile.get('shape', {}).get('rows')} строк, {profile.get('shape', {}).get('columns')} колонок",
        f"- После очистки: {cleaning.get('rows_after')} строк",
        f"- Обоснование очистки: {cleaning.get('cleaning_reasoning')}",
        "",
        "## EDA",
        f"- Потенциальные утечки: {', '.join(eda.get('leakage_candidates', [])) or 'не обнаружены'}",
        "",
        "## Feature Engineering",
        f"- Новые признаки: {', '.join(features.get('created', [])) or 'не создавались'}",
        "",
        "## Результаты моделирования",
        f"- Модели: {', '.join(model_plan.get('selected_models', []))}",
        f"- Лучшая модель: {evaluation.get('best_model_name')}",
        f"- Test метрики: {evaluation.get('test_metrics')}",
        f"- Validation метрики: {evaluation.get('validation_metrics')}",
        "",
        "## Долговременная память",
        f"- Исторический лучший: {history.get('best_run')}",
    ])

    report_path = os.path.join(ws, "reports", "run_report.md")
    with open(report_path, "w", encoding="utf-8") as _rf:
        _rf.write(report)
    write_json(os.path.join(ws, "reports", "run_report.json"), {
        "business_task": runtime.state["business_task"],
        "goal": goal, "evaluation": evaluation,
    })
    return f"Report saved to {report_path}."


def get_all_tools():
    return [
        profile_dataset, get_dataset_schema, set_modeling_goal, analyze_distributions,
        clean_dataset, run_eda, engineer_features, select_candidate_models,
        prepare_splits, tune_models, train_models, evaluate_models,
        load_long_term_memory, load_best_model_from_memory, save_best_model, write_report,
        collect_pipeline_highlights, write_business_report,
    ]
