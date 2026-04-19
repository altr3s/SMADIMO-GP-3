from __future__ import annotations

import math
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

from smadimo_agent.io_utils import read_json, read_table, write_json, write_table, write_text

try:
    from langchain.tools import ToolRuntime, tool
except ImportError:
    from langchain_core.tools import tool

    ToolRuntime = Any  # type: ignore


TASK_TYPES = {"classification", "regression", "clustering"}

SUPPORTED_MODELS = {
    "classification": {
        "logistic_regression",
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "linear_svc",
        "k_neighbors_classifier",
    },
    "regression": {
        "ridge_regression",
        "sgd_regressor",
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "k_neighbors_regressor",
    },
    "clustering": {
        "kmeans",
        "agglomerative_clustering",
        "dbscan",
    },
}

MODEL_NAME_ALIASES = {
    "classification": {
        "logisticregression": "logistic_regression",
        "logistic_regression": "logistic_regression",
        "randomforestclassifier": "random_forest_classifier",
        "random_forest_classifier": "random_forest_classifier",
        "gradientboostingclassifier": "gradient_boosting_classifier",
        "gradient_boosting_classifier": "gradient_boosting_classifier",
        "xgbclassifier": "gradient_boosting_classifier",
        "xgboostclassifier": "gradient_boosting_classifier",
        "linearsvc": "linear_svc",
        "linear_svc": "linear_svc",
        "svc": "linear_svc",
        "kneighborsclassifier": "k_neighbors_classifier",
        "kneighborsclassifier": "k_neighbors_classifier",
        "k_neighbors_classifier": "k_neighbors_classifier",
        "knnclassifier": "k_neighbors_classifier",
    },
    "regression": {
        "ridge": "ridge_regression",
        "ridgeregression": "ridge_regression",
        "ridge_regression": "ridge_regression",
        "linearregression": "ridge_regression",
        "linear_regression": "ridge_regression",
        "sgdregressor": "sgd_regressor",
        "sgd_regressor": "sgd_regressor",
        "randomforestregressor": "random_forest_regressor",
        "random_forest_regressor": "random_forest_regressor",
        "gradientboostingregressor": "gradient_boosting_regressor",
        "gradient_boosting_regressor": "gradient_boosting_regressor",
        "xgbregressor": "gradient_boosting_regressor",
        "xgboostregressor": "gradient_boosting_regressor",
        "kneighborsregressor": "k_neighbors_regressor",
        "kneighborsregressor": "k_neighbors_regressor",
        "k_neighbors_regressor": "k_neighbors_regressor",
        "knnregressor": "k_neighbors_regressor",
    },
    "clustering": {
        "kmeans": "kmeans",
        "k_means": "kmeans",
        "agglomerativeclustering": "agglomerative_clustering",
        "agglomerative_clustering": "agglomerative_clustering",
        "hierarchicalclustering": "agglomerative_clustering",
        "dbscan": "dbscan",
    },
}


class CleaningPlan(BaseModel):
    drop_duplicates: bool = True
    trim_whitespace: bool = True
    auto_cast_numeric: bool = True
    numeric_coercion_threshold: float = 0.85
    auto_parse_dates: bool = True
    date_columns: List[str] = Field(default_factory=list)
    drop_columns: List[str] = Field(default_factory=list)
    numeric_imputation: Literal["median", "mean", "none"] = "median"
    categorical_imputation: Literal["mode", "constant", "none"] = "mode"
    text_imputation: Literal["empty", "constant", "none"] = "empty"
    fill_constant: str = "missing"
    outlier_strategy: Literal["iqr_clip", "none"] = "iqr_clip"


class FeatureSpec(BaseModel):
    name: str
    kind: Literal[
        "ratio",
        "difference",
        "product",
        "sum",
        "text_length",
        "text_word_count",
        "date_part",
        "is_missing",
        "category_frequency",
    ]
    source_columns: List[str]
    date_part: Optional[Literal["year", "month", "day", "dayofweek", "quarter"]] = None
    denominator_guard: float = 1e-6


class ModelSelectionPlan(BaseModel):
    model_names: List[str]
    reasoning: str


class SplitPlan(BaseModel):
    test_size: float = 0.2
    val_size: float = 0.1
    stratify: bool = True


def build_ml_tools() -> List[Any]:
    return [
        profile_dataset,
        get_dataset_schema,
        set_modeling_goal,
        clean_dataset,
        run_eda,
        engineer_features,
        select_candidate_models,
        prepare_splits,
        train_models,
        evaluate_models,
        load_long_term_memory,
        save_best_model,
        write_report,
    ]


def _workspace_paths(workspace_dir: Path) -> Dict[str, Path]:
    return {
        "workflow_spec": workspace_dir / "workflow_spec.json",
        "analysis_dir": workspace_dir / "analysis",
        "data_dir": workspace_dir / "data",
        "cleaned_dataset": workspace_dir / "data" / "cleaned" / "cleaned_dataset.csv",
        "featured_dataset": workspace_dir / "data" / "featured" / "featured_dataset.csv",
        "profile_json": workspace_dir / "analysis" / "dataset_profile.json",
        "schema_json": workspace_dir / "analysis" / "schema_snapshot.json",
        "goal_json": workspace_dir / "analysis" / "modeling_goal.json",
        "cleaning_json": workspace_dir / "analysis" / "cleaning_report.json",
        "eda_json": workspace_dir / "analysis" / "eda_report.json",
        "eda_md": workspace_dir / "analysis" / "eda_report.md",
        "feature_json": workspace_dir / "analysis" / "feature_report.json",
        "model_plan_json": workspace_dir / "modeling" / "model_plan.json",
        "split_dir": workspace_dir / "modeling" / "splits",
        "split_manifest": workspace_dir / "modeling" / "splits" / "split_manifest.json",
        "leaderboard_json": workspace_dir / "modeling" / "leaderboard.json",
        "evaluation_json": workspace_dir / "modeling" / "evaluation.json",
        "models_dir": workspace_dir / "models",
        "current_best_model": workspace_dir / "models" / "best_current_model.joblib",
        "memory_dir": workspace_dir.parent / "memory",
        "history_json": workspace_dir.parent / "memory" / "best_registry.json",
        "memory_model": workspace_dir.parent / "memory" / "best_model.joblib",
        "memory_meta": workspace_dir.parent / "memory" / "best_model_meta.json",
        "report_md": workspace_dir / "reports" / "run_report.md",
        "report_json": workspace_dir / "reports" / "run_report.json",
    }


def collect_artifacts(workspace_dir: Path) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for name, path in _workspace_paths(workspace_dir).items():
        if path.exists() and path.is_file():
            artifacts[name] = str(path)
    return artifacts


def _get_workspace_dir(runtime: ToolRuntime) -> Path:
    return Path(runtime.state["workspace_dir"]).resolve()


def _get_raw_dataset_path(runtime: ToolRuntime) -> Path:
    return Path(runtime.state["dataset_path"]).resolve()


def _get_goal(runtime: ToolRuntime) -> Dict[str, Any]:
    workspace_dir = _get_workspace_dir(runtime)
    return read_json(_workspace_paths(workspace_dir)["goal_json"])


def _get_task_type(runtime: ToolRuntime) -> Optional[str]:
    return runtime.state.get("task_type") or _get_goal(runtime).get("task_type")


def _get_target_column(runtime: ToolRuntime) -> Optional[str]:
    return runtime.state.get("target_column") or _get_goal(runtime).get("target_column")


def _active_dataset_path(runtime: ToolRuntime) -> Path:
    paths = _workspace_paths(_get_workspace_dir(runtime))
    if paths["featured_dataset"].exists():
        return paths["featured_dataset"]
    if paths["cleaned_dataset"].exists():
        return paths["cleaned_dataset"]
    return _get_raw_dataset_path(runtime)


def _cleaned_base_dataset_path(runtime: ToolRuntime) -> Path:
    cleaned = _workspace_paths(_get_workspace_dir(runtime))["cleaned_dataset"]
    return cleaned if cleaned.exists() else _get_raw_dataset_path(runtime)


def _normalize_model_token(value: str) -> str:
    return "".join(char.lower() for char in value if char.isalnum() or char == "_")


def _normalize_model_names(
    task_type: str,
    requested_names: List[str],
) -> Tuple[List[str], List[Dict[str, str]], List[str]]:
    allowed = SUPPORTED_MODELS[task_type]
    aliases = MODEL_NAME_ALIASES[task_type]
    normalized_names: List[str] = []
    substitutions: List[Dict[str, str]] = []
    invalid_names: List[str] = []

    for requested in requested_names:
        candidate = requested if requested in allowed else aliases.get(_normalize_model_token(requested))
        if not candidate:
            invalid_names.append(requested)
            continue
        if candidate not in normalized_names:
            normalized_names.append(candidate)
        if candidate != requested:
            substitutions.append({"requested": requested, "resolved": candidate})

    return normalized_names, substitutions, invalid_names


def _profile_candidates(df: pd.DataFrame, roles: Dict[str, str]) -> Dict[str, List[str]]:
    classification_candidates: List[str] = []
    regression_candidates: List[str] = []
    for column in df.columns:
        series = df[column]
        if roles[column] == "identifier":
            continue
        non_null = series.dropna()
        unique_values = int(non_null.nunique())
        unique_ratio = float(non_null.nunique() / max(len(non_null), 1))
        if pd.api.types.is_numeric_dtype(series):
            if unique_values > 10 and unique_ratio > 0.03:
                regression_candidates.append(column)
            if 2 <= unique_values <= 20:
                classification_candidates.append(column)
        elif roles[column] == "categorical":
            if 2 <= unique_values <= 25:
                classification_candidates.append(column)
    return {
        "classification": classification_candidates[:10],
        "regression": regression_candidates[:10],
    }


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(series, errors="coerce")


def _columns_by_role(roles: Dict[str, str]) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for column, role in roles.items():
        grouped.setdefault(role, []).append(column)
    for columns in grouped.values():
        columns.sort()
    return grouped


def _schema_summary_text(
    df: pd.DataFrame,
    roles: Dict[str, str],
    target_column: Optional[str] = None,
) -> str:
    columns_by_role = _columns_by_role(roles)
    lines = [
        f"rows={df.shape[0]}, columns={df.shape[1]}",
        f"target={target_column or 'not set'}",
    ]
    for role in [
        "numeric",
        "categorical",
        "text",
        "datetime",
        "datetime_candidate",
        "identifier",
        "target",
    ]:
        role_columns = columns_by_role.get(role, [])
        if role_columns:
            lines.append(f"{role}: {', '.join(role_columns)}")
    return " | ".join(lines)


def _feature_kind_is_compatible(
    spec: FeatureSpec,
    df: pd.DataFrame,
    roles: Dict[str, str],
    target_column: Optional[str],
) -> bool:
    if any(column == target_column for column in spec.source_columns):
        return False

    if spec.kind in {"ratio", "difference", "product", "sum"}:
        return (
            len(spec.source_columns) >= 2
            and all(roles.get(column) == "numeric" for column in spec.source_columns[:2])
        )
    if spec.kind in {"text_length", "text_word_count"}:
        return len(spec.source_columns) >= 1 and roles.get(spec.source_columns[0]) == "text"
    if spec.kind == "date_part":
        return len(spec.source_columns) >= 1 and roles.get(spec.source_columns[0]) in {
            "datetime",
            "datetime_candidate",
        }
    if spec.kind == "is_missing":
        return len(spec.source_columns) >= 1 and spec.source_columns[0] in df.columns
    if spec.kind == "category_frequency":
        return len(spec.source_columns) >= 1 and roles.get(spec.source_columns[0]) == "categorical"
    return False


def _auto_feature_candidates(
    df: pd.DataFrame,
    roles: Dict[str, str],
    target_column: Optional[str],
) -> List[FeatureSpec]:
    candidates: List[FeatureSpec] = []
    numeric_columns = [
        column
        for column, role in roles.items()
        if role == "numeric" and column != target_column
    ]
    categorical_columns = [
        column
        for column, role in roles.items()
        if role == "categorical" and column != target_column
    ]
    text_columns = [
        column
        for column, role in roles.items()
        if role == "text" and column != target_column
    ]
    datetime_columns = [
        column
        for column, role in roles.items()
        if role in {"datetime", "datetime_candidate"} and column != target_column
    ]

    if len(numeric_columns) >= 2:
        first, second = numeric_columns[:2]
        candidates.append(
            FeatureSpec(
                name=f"{first}_to_{second}_ratio",
                kind="ratio",
                source_columns=[first, second],
            )
        )
        candidates.append(
            FeatureSpec(
                name=f"{first}_minus_{second}",
                kind="difference",
                source_columns=[first, second],
            )
        )

    if text_columns:
        text_column = text_columns[0]
        candidates.append(
            FeatureSpec(
                name=f"{text_column}_length",
                kind="text_length",
                source_columns=[text_column],
            )
        )
        candidates.append(
            FeatureSpec(
                name=f"{text_column}_word_count",
                kind="text_word_count",
                source_columns=[text_column],
            )
        )

    if datetime_columns:
        date_column = datetime_columns[0]
        candidates.append(
            FeatureSpec(
                name=f"{date_column}_month",
                kind="date_part",
                source_columns=[date_column],
                date_part="month",
            )
        )
        candidates.append(
            FeatureSpec(
                name=f"{date_column}_dayofweek",
                kind="date_part",
                source_columns=[date_column],
                date_part="dayofweek",
            )
        )

    if categorical_columns:
        category_column = categorical_columns[0]
        candidates.append(
            FeatureSpec(
                name=f"{category_column}_frequency",
                kind="category_frequency",
                source_columns=[category_column],
            )
        )

    missing_columns = [
        column
        for column in df.columns
        if column != target_column and int(df[column].isna().sum()) > 0
    ]
    if missing_columns:
        column = missing_columns[0]
        candidates.append(
            FeatureSpec(
                name=f"{column}_is_missing",
                kind="is_missing",
                source_columns=[column],
            )
        )

    unique_candidates: List[FeatureSpec] = []
    seen_names = set()
    for spec in candidates:
        if spec.name in seen_names or spec.name in df.columns:
            continue
        seen_names.add(spec.name)
        unique_candidates.append(spec)
    return unique_candidates


def _resolve_feature_specs(
    df: pd.DataFrame,
    requested_specs: List[FeatureSpec],
    target_column: Optional[str],
) -> Tuple[List[FeatureSpec], List[Dict[str, str]]]:
    roles = _infer_column_roles(df, target_column=target_column)
    valid_specs: List[FeatureSpec] = []
    skipped_specs: List[Dict[str, str]] = []
    seen_names = set(df.columns.tolist())

    for spec in requested_specs:
        if spec.name in seen_names:
            skipped_specs.append(
                {"name": spec.name, "reason": "feature name already exists in dataset"}
            )
            continue
        missing_columns = [column for column in spec.source_columns if column not in df.columns]
        if missing_columns:
            skipped_specs.append(
                {
                    "name": spec.name,
                    "reason": f"missing columns: {', '.join(missing_columns)}",
                }
            )
            continue
        if not _feature_kind_is_compatible(spec, df, roles, target_column):
            skipped_specs.append(
                {"name": spec.name, "reason": "incompatible source column types"}
            )
            continue
        valid_specs.append(spec)
        seen_names.add(spec.name)

    for spec in _auto_feature_candidates(df, roles, target_column):
        if len(valid_specs) >= 2:
            break
        if spec.name in seen_names:
            continue
        valid_specs.append(spec)
        seen_names.add(spec.name)

    if len(valid_specs) < 2:
        schema_summary = _schema_summary_text(df, roles, target_column=target_column)
        raise ValueError(
            "Unable to build at least 2 valid features from the actual dataset schema. "
            f"Available schema: {schema_summary}"
        )

    return valid_specs, skipped_specs


def _infer_column_roles(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict[str, str]:
    roles: Dict[str, str] = {}
    for column in df.columns:
        if target_column and column == target_column:
            roles[column] = "target"
            continue
        series = df[column]
        non_null = series.dropna()
        unique_ratio = float(non_null.nunique() / max(len(non_null), 1))
        name = column.lower()

        if pd.api.types.is_bool_dtype(series):
            roles[column] = "categorical"
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            roles[column] = "datetime"
            continue

        if pd.api.types.is_numeric_dtype(series):
            if unique_ratio > 0.98 and any(token in name for token in ("id", "uuid", "code", "number")):
                roles[column] = "identifier"
            else:
                roles[column] = "numeric"
            continue

        as_string = non_null.astype(str)
        avg_len = float(as_string.str.len().mean()) if not as_string.empty else 0.0
        avg_words = (
            float(as_string.str.split().str.len().mean()) if not as_string.empty else 0.0
        )
        parse_ratio = 0.0
        if not as_string.empty:
            parsed = _safe_to_datetime(as_string)
            parse_ratio = float(parsed.notna().mean())

        if unique_ratio > 0.98 and any(token in name for token in ("id", "uuid", "code", "number")):
            roles[column] = "identifier"
        elif parse_ratio >= 0.8 and avg_len >= 6:
            roles[column] = "datetime_candidate"
        elif avg_len >= 35 or avg_words >= 4 or unique_ratio >= 0.5:
            roles[column] = "text"
        else:
            roles[column] = "categorical"
    return roles


def _describe_columns(df: pd.DataFrame, roles: Dict[str, str]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for column in df.columns:
        series = df[column]
        sample_values = [value for value in series.dropna().astype(str).head(3).tolist()]
        result.append(
            {
                "name": column,
                "dtype": str(series.dtype),
                "role": roles[column],
                "missing_count": int(series.isna().sum()),
                "missing_ratio": round(float(series.isna().mean()), 4),
                "unique_values": int(series.nunique(dropna=True)),
                "sample_values": sample_values,
            }
        )
    return result


def _auto_cast_numeric(df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    converted: List[str] = []
    for column in df.columns:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            continue
        candidate = pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")
        ratio = float(candidate.notna().mean())
        if ratio >= threshold:
            df[column] = candidate
            converted.append(column)
    return df, converted


def _auto_parse_dates(df: pd.DataFrame, explicit_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    parsed_columns: List[str] = []
    columns_to_check = list(explicit_columns)
    if not columns_to_check:
        for column in df.columns:
            if pd.api.types.is_numeric_dtype(df[column]):
                continue
            sample = df[column].dropna().astype(str)
            if sample.empty:
                continue
            parsed = _safe_to_datetime(sample)
            if float(parsed.notna().mean()) >= 0.8:
                columns_to_check.append(column)
    for column in columns_to_check:
        if column not in df.columns:
            continue
        converted = _safe_to_datetime(df[column])
        if converted.notna().any():
            df[column] = converted
            parsed_columns.append(column)
    return df, parsed_columns


def _apply_cleaning_plan(
    df: pd.DataFrame,
    plan: CleaningPlan,
    target_column: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report: Dict[str, Any] = {
        "rows_before": int(len(df)),
        "columns_before": int(len(df.columns)),
        "dropped_duplicates": 0,
        "dropped_columns": [],
        "numeric_cast_columns": [],
        "parsed_date_columns": [],
        "target_rows_removed": 0,
    }

    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    if plan.trim_whitespace:
        for column in df.select_dtypes(include=["object", "string"]).columns:
            df[column] = df[column].map(
                lambda value: value.strip() if isinstance(value, str) else value
            )

    if plan.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        report["dropped_duplicates"] = int(before - len(df))

    if plan.drop_columns:
        existing = [column for column in plan.drop_columns if column in df.columns]
        df = df.drop(columns=existing)
        report["dropped_columns"] = existing

    if plan.auto_cast_numeric:
        df, converted = _auto_cast_numeric(df, plan.numeric_coercion_threshold)
        report["numeric_cast_columns"] = converted

    if plan.auto_parse_dates or plan.date_columns:
        df, parsed_dates = _auto_parse_dates(df, plan.date_columns)
        report["parsed_date_columns"] = parsed_dates

    roles = _infer_column_roles(df, target_column=target_column)

    if target_column and target_column in df.columns:
        before = len(df)
        df = df[df[target_column].notna()].copy()
        report["target_rows_removed"] = int(before - len(df))

    numeric_columns = [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column]) and column != target_column
    ]
    categorical_columns = [column for column, role in roles.items() if role == "categorical"]
    text_columns = [column for column, role in roles.items() if role == "text"]

    if plan.numeric_imputation != "none":
        strategy = "median" if plan.numeric_imputation == "median" else "mean"
        for column in numeric_columns:
            fill_value = df[column].median() if strategy == "median" else df[column].mean()
            df[column] = df[column].fillna(fill_value)

    if plan.categorical_imputation != "none":
        for column in categorical_columns:
            if plan.categorical_imputation == "mode" and not df[column].mode(dropna=True).empty:
                fill_value = df[column].mode(dropna=True).iloc[0]
            else:
                fill_value = plan.fill_constant
            df[column] = df[column].fillna(fill_value)

    if plan.text_imputation != "none":
        fill_value = "" if plan.text_imputation == "empty" else plan.fill_constant
        for column in text_columns:
            df[column] = df[column].fillna(fill_value)

    if plan.outlier_strategy == "iqr_clip":
        for column in numeric_columns:
            series = df[column].dropna()
            if series.empty:
                continue
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            df[column] = df[column].clip(lower=lower, upper=upper)

    report["rows_after"] = int(len(df))
    report["columns_after"] = int(len(df.columns))
    report["missing_after"] = df.isna().sum().to_dict()
    return df, report


def _outlier_counts(df: pd.DataFrame, target_column: Optional[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for column in df.columns:
        if column == target_column or not pd.api.types.is_numeric_dtype(df[column]):
            continue
        series = df[column].dropna()
        if series.empty:
            counts[column] = 0
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            counts[column] = 0
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        counts[column] = int(mask.sum())
    return counts


def _detect_leakage_candidates(df: pd.DataFrame, target_column: Optional[str]) -> List[str]:
    if not target_column or target_column not in df.columns:
        return []
    leakage: List[str] = []
    target = df[target_column]
    if pd.api.types.is_numeric_dtype(target):
        numeric_df = df.select_dtypes(include=["number"])
        if target_column in numeric_df.columns:
            corr = numeric_df.corr(numeric_only=True)[target_column].drop(labels=[target_column])
            leakage.extend(corr[abs(corr) > 0.98].index.tolist())
    for column in df.columns:
        if column == target_column:
            continue
        if column.lower() == target_column.lower():
            leakage.append(column)
        if df[column].nunique(dropna=True) == len(df) and "id" in column.lower():
            leakage.append(column)
    return sorted(set(leakage))


def _validate_target_and_task(df: pd.DataFrame, target_column: Optional[str], task_type: str) -> None:
    if task_type not in TASK_TYPES:
        raise ValueError(f"Unsupported task type: {task_type}")
    if task_type != "clustering":
        if not target_column:
            raise ValueError("Target column must be set for supervised tasks.")
        if target_column not in df.columns:
            raise ValueError(f"Target column `{target_column}` not found in dataset.")


def _prepare_feature_table(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    roles = _infer_column_roles(df, target_column=target_column)
    drop_columns = [
        column
        for column, role in roles.items()
        if role in {"identifier", "text", "datetime", "datetime_candidate"}
        and column != target_column
    ]
    feature_df = df.drop(columns=drop_columns, errors="ignore").copy()
    metadata = {
        "dropped_from_modeling": drop_columns,
        "numeric_columns": [
            column
            for column in feature_df.columns
            if column != target_column and pd.api.types.is_numeric_dtype(feature_df[column])
        ],
        "categorical_columns": [
            column
            for column in feature_df.columns
            if column != target_column
            and not pd.api.types.is_numeric_dtype(feature_df[column])
            and not pd.api.types.is_datetime64_any_dtype(feature_df[column])
        ],
    }
    return feature_df, metadata


def _build_supervised_preprocessor(
    feature_df: pd.DataFrame,
    target_column: str,
) -> ColumnTransformer:
    numeric_columns = [
        column
        for column in feature_df.columns
        if column != target_column and pd.api.types.is_numeric_dtype(feature_df[column])
    ]
    categorical_columns = [
        column
        for column in feature_df.columns
        if column != target_column
        and not pd.api.types.is_numeric_dtype(feature_df[column])
        and not pd.api.types.is_datetime64_any_dtype(feature_df[column])
    ]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("No supported columns left for modeling after preprocessing.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_model(task_type: str, model_name: str) -> Any:
    if task_type == "classification":
        registry = {
            "logistic_regression": LogisticRegression(max_iter=2000, n_jobs=None),
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=400, random_state=42, n_jobs=-1
            ),
            "gradient_boosting_classifier": GradientBoostingClassifier(random_state=42),
            "linear_svc": LinearSVC(),
            "k_neighbors_classifier": KNeighborsClassifier(n_neighbors=11),
        }
    elif task_type == "regression":
        registry = {
            "ridge_regression": Ridge(alpha=1.0),
            "sgd_regressor": SGDRegressor(
                random_state=42,
                max_iter=2000,
                penalty="elasticnet",
                early_stopping=True,
            ),
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=400, random_state=42, n_jobs=-1
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(random_state=42),
            "k_neighbors_regressor": KNeighborsRegressor(n_neighbors=11),
        }
    else:
        registry = {
            "kmeans": KMeans(n_clusters=3, random_state=42, n_init=20),
            "agglomerative_clustering": AgglomerativeClustering(n_clusters=3),
            "dbscan": DBSCAN(eps=0.7, min_samples=10),
        }
    if model_name not in registry:
        raise ValueError(f"Model `{model_name}` is not registered for task `{task_type}`.")
    return registry[model_name]


def _binary_average(y_true: np.ndarray) -> str:
    return "binary" if len(np.unique(y_true)) == 2 else "weighted"


def _classification_metrics(
    estimator: Pipeline,
    X: pd.DataFrame,
    y_true: np.ndarray,
) -> Dict[str, Optional[float]]:
    predictions = estimator.predict(X)
    average = _binary_average(y_true)
    metrics: Dict[str, Optional[float]] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, average=average, zero_division=0)
        ),
        "recall": float(recall_score(y_true, predictions, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, average=average, zero_division=0)),
        "roc_auc": None,
    }

    if len(np.unique(y_true)) == 2:
        if hasattr(estimator, "predict_proba"):
            probabilities = estimator.predict_proba(X)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        elif hasattr(estimator, "decision_function"):
            scores = estimator.decision_function(X)
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    return metrics


def _regression_metrics(estimator: Pipeline, X: pd.DataFrame, y_true: np.ndarray) -> Dict[str, float]:
    predictions = estimator.predict(X)
    mse = float(mean_squared_error(y_true, predictions))
    metrics = {
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, predictions)),
        "r2": float(r2_score(y_true, predictions)),
    }
    try:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true, predictions))
    except ValueError:
        metrics["mape"] = math.nan
    return metrics


def _safe_cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    unique_labels = set(labels.tolist())
    if len(unique_labels) <= 1:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
        }
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def _selection_score(task_type: str, metrics: Dict[str, Any]) -> float:
    if task_type == "classification":
        return float(metrics.get("roc_auc") or metrics.get("f1") or -math.inf)
    if task_type == "regression":
        return -float(metrics["rmse"])
    return float(metrics.get("silhouette") or -math.inf)


def _fit_supervised_model(
    model_name: str,
    task_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_column: str,
) -> Dict[str, Any]:
    feature_train, prep_meta = _prepare_feature_table(train_df, target_column=target_column)
    feature_val, _ = _prepare_feature_table(val_df, target_column=target_column)

    preprocessor = _build_supervised_preprocessor(feature_train, target_column=target_column)
    estimator = _build_model(task_type, model_name)
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    y_train_raw = feature_train[target_column].to_numpy()
    y_val_raw = feature_val[target_column].to_numpy()
    label_encoder: Optional[LabelEncoder] = None

    if task_type == "classification":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_val = label_encoder.transform(y_val_raw)
    else:
        y_train = y_train_raw
        y_val = y_val_raw

    X_train = feature_train.drop(columns=[target_column])
    X_val = feature_val.drop(columns=[target_column])

    pipeline.fit(X_train, y_train)

    if task_type == "classification":
        metrics = _classification_metrics(pipeline, X_val, y_val)
    else:
        metrics = _regression_metrics(pipeline, X_val, y_val)

    return {
        "pipeline": pipeline,
        "label_encoder": label_encoder,
        "metrics": metrics,
        "selection_score": _selection_score(task_type, metrics),
        "metadata": prep_meta,
    }


def _fit_clustering_model(
    model_name: str,
    train_df: pd.DataFrame,
) -> Dict[str, Any]:
    feature_train, prep_meta = _prepare_feature_table(train_df)
    preprocessor = _build_supervised_preprocessor(feature_train.assign(_dummy_target=0), "_dummy_target")
    X_train = feature_train.copy()
    X_train["_dummy_target"] = 0
    transformed = preprocessor.fit_transform(X_train)
    transformed = np.asarray(transformed)

    estimator = _build_model("clustering", model_name)
    if hasattr(estimator, "fit_predict"):
        labels = estimator.fit_predict(transformed)
    else:
        estimator.fit(transformed)
        labels = estimator.predict(transformed)

    metrics = _safe_cluster_metrics(transformed, np.asarray(labels))
    return {
        "preprocessor": preprocessor,
        "model": estimator,
        "metrics": metrics,
        "selection_score": _selection_score("clustering", metrics),
        "metadata": prep_meta,
    }


@tool
def profile_dataset(runtime: ToolRuntime) -> str:
    """Profile the current dataset and save a structured dataset description."""
    workspace_dir = _get_workspace_dir(runtime)
    df = read_table(_active_dataset_path(runtime))
    roles = _infer_column_roles(df)
    payload = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": _describe_columns(df, roles),
        "roles": roles,
        "columns_by_role": _columns_by_role(roles),
        "summary": _schema_summary_text(df, roles),
        "missing_by_column": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "candidate_targets": _profile_candidates(df, roles),
    }
    write_json(_workspace_paths(workspace_dir)["profile_json"], payload)
    return (
        f"Dataset profiled: {df.shape[0]} rows, {df.shape[1]} columns. "
        f"Profile saved to {_workspace_paths(workspace_dir)['profile_json']}."
    )


@tool
def get_dataset_schema(runtime: ToolRuntime) -> str:
    """Inspect the current dataset schema and persist actual columns grouped by inferred role."""
    workspace_dir = _get_workspace_dir(runtime)
    target_column = _get_target_column(runtime)
    df = read_table(_active_dataset_path(runtime))
    roles = _infer_column_roles(df, target_column=target_column)
    payload = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "roles": roles,
        "columns_by_role": _columns_by_role(roles),
        "summary": _schema_summary_text(df, roles, target_column=target_column),
    }
    write_json(_workspace_paths(workspace_dir)["schema_json"], payload)
    return payload["summary"]


@tool
def set_modeling_goal(
    target_column: str,
    task_type: Literal["classification", "regression", "clustering"],
    reasoning: str,
    runtime: ToolRuntime,
) -> str:
    """Set the target column and task type for the ML problem."""
    workspace_dir = _get_workspace_dir(runtime)
    df = read_table(_active_dataset_path(runtime))
    if task_type != "clustering" and target_column not in df.columns:
        raise ValueError(f"Target column `{target_column}` not found in dataset.")
    payload = {
        "target_column": target_column if task_type != "clustering" else None,
        "task_type": task_type,
        "reasoning": reasoning,
    }
    write_json(_workspace_paths(workspace_dir)["goal_json"], payload)
    return f"Modeling goal stored. task_type={task_type}, target={payload['target_column']}."


@tool
def clean_dataset(plan: CleaningPlan, runtime: ToolRuntime) -> str:
    """Clean the raw dataset according to an explicit cleaning plan."""
    workspace_dir = _get_workspace_dir(runtime)
    target_column = _get_target_column(runtime)
    df = read_table(_get_raw_dataset_path(runtime))
    cleaned_df, report = _apply_cleaning_plan(df, plan, target_column)
    paths = _workspace_paths(workspace_dir)
    write_table(cleaned_df, paths["cleaned_dataset"])
    write_json(paths["cleaning_json"], report)
    return (
        f"Dataset cleaned. Rows: {report['rows_before']} -> {report['rows_after']}. "
        f"Saved to {paths['cleaned_dataset']}."
    )


@tool
def run_eda(runtime: ToolRuntime) -> str:
    """Run exploratory data analysis for the current cleaned dataset."""
    workspace_dir = _get_workspace_dir(runtime)
    target_column = _get_target_column(runtime)
    task_type = _get_task_type(runtime)
    df = read_table(_active_dataset_path(runtime))
    roles = _infer_column_roles(df, target_column=target_column)

    numeric_df = df.select_dtypes(include=["number"])
    corr_with_target: Dict[str, float] = {}
    if target_column and target_column in numeric_df.columns:
        corr_series = numeric_df.corr(numeric_only=True)[target_column].drop(labels=[target_column])
        corr_with_target = corr_series.sort_values(key=lambda item: item.abs(), ascending=False).head(10).to_dict()

    class_balance: Dict[str, int] = {}
    if task_type == "classification" and target_column:
        class_balance = df[target_column].value_counts(dropna=False).to_dict()

    payload = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "roles": roles,
        "missing_by_column": df.isna().sum().to_dict(),
        "outliers_by_numeric_column": _outlier_counts(df, target_column),
        "target_distribution": class_balance,
        "top_target_correlations": corr_with_target,
        "leakage_candidates": _detect_leakage_candidates(df, target_column),
    }

    report_lines = [
        "# EDA report",
        f"- rows: {payload['shape']['rows']}",
        f"- columns: {payload['shape']['columns']}",
        f"- target: {target_column}",
        f"- task_type: {task_type}",
        f"- leakage_candidates: {', '.join(payload['leakage_candidates']) or 'none'}",
    ]
    write_json(_workspace_paths(workspace_dir)["eda_json"], payload)
    write_text(_workspace_paths(workspace_dir)["eda_md"], "\n".join(report_lines))
    return f"EDA report saved to {_workspace_paths(workspace_dir)['eda_json']}."


@tool
def engineer_features(
    feature_specs: List[FeatureSpec],
    note: str,
    runtime: ToolRuntime,
) -> str:
    """Create new features on top of the cleaned dataset."""
    workspace_dir = _get_workspace_dir(runtime)
    target_column = _get_target_column(runtime)
    df = read_table(_cleaned_base_dataset_path(runtime)).copy()
    resolved_specs, skipped_specs = _resolve_feature_specs(
        df=df,
        requested_specs=feature_specs,
        target_column=target_column,
    )
    created_features: List[str] = []

    for spec in resolved_specs:
        if spec.kind == "ratio":
            numerator, denominator = spec.source_columns[:2]
            df[spec.name] = df[numerator] / (df[denominator].replace(0, np.nan) + spec.denominator_guard)
        elif spec.kind == "difference":
            left, right = spec.source_columns[:2]
            df[spec.name] = df[left] - df[right]
        elif spec.kind == "product":
            left, right = spec.source_columns[:2]
            df[spec.name] = df[left] * df[right]
        elif spec.kind == "sum":
            left, right = spec.source_columns[:2]
            df[spec.name] = df[left] + df[right]
        elif spec.kind == "text_length":
            source = spec.source_columns[0]
            df[spec.name] = df[source].fillna("").astype(str).str.len()
        elif spec.kind == "text_word_count":
            source = spec.source_columns[0]
            df[spec.name] = df[source].fillna("").astype(str).str.split().str.len()
        elif spec.kind == "date_part":
            source = spec.source_columns[0]
            parsed = pd.to_datetime(df[source], errors="coerce")
            if spec.date_part == "year":
                df[spec.name] = parsed.dt.year
            elif spec.date_part == "month":
                df[spec.name] = parsed.dt.month
            elif spec.date_part == "day":
                df[spec.name] = parsed.dt.day
            elif spec.date_part == "dayofweek":
                df[spec.name] = parsed.dt.dayofweek
            elif spec.date_part == "quarter":
                df[spec.name] = parsed.dt.quarter
            else:
                raise ValueError("date_part must be set for date_part features.")
        elif spec.kind == "is_missing":
            source = spec.source_columns[0]
            df[spec.name] = df[source].isna().astype(int)
        elif spec.kind == "category_frequency":
            source = spec.source_columns[0]
            frequencies = df[source].value_counts(normalize=True, dropna=False)
            df[spec.name] = df[source].map(frequencies)
        else:
            raise ValueError(f"Unsupported feature kind `{spec.kind}`.")

        created_features.append(spec.name)

    paths = _workspace_paths(workspace_dir)
    write_table(df, paths["featured_dataset"])
    write_json(
        paths["feature_json"],
        {
            "note": note,
            "created_features": created_features,
            "requested_feature_specs": [spec.model_dump() for spec in feature_specs],
            "applied_feature_specs": [spec.model_dump() for spec in resolved_specs],
            "skipped_feature_specs": skipped_specs,
        },
    )
    skipped_count = len(skipped_specs)
    return (
        f"Created {len(created_features)} features from actual dataset columns and saved dataset to "
        f"{paths['featured_dataset']}. Skipped invalid requested features: {skipped_count}."
    )


@tool
def select_candidate_models(plan: ModelSelectionPlan, runtime: ToolRuntime) -> str:
    """Register a shortlist of ML models for the current task."""
    workspace_dir = _get_workspace_dir(runtime)
    task_type = _get_task_type(runtime)
    if not task_type:
        raise ValueError("Task type must be defined before model selection.")
    requested_model_names = plan.model_names
    normalized_models, substitutions, invalid = _normalize_model_names(
        task_type=task_type,
        requested_names=requested_model_names,
    )

    fallback_by_task = {
        "classification": ["logistic_regression", "random_forest_classifier"],
        "regression": ["ridge_regression", "random_forest_regressor"],
        "clustering": ["kmeans", "agglomerative_clustering"],
    }
    for candidate in fallback_by_task[task_type]:
        if len(normalized_models) >= 2:
            break
        if candidate not in normalized_models:
            normalized_models.append(candidate)

    if len(normalized_models) < 2:
        raise ValueError(
            f"Unable to build a valid model pool for task `{task_type}` from requested models: "
            f"{requested_model_names}"
        )

    payload = {
        "task_type": task_type,
        "requested_models": requested_model_names,
        "selected_models": normalized_models,
        "reasoning": plan.reasoning,
        "model_alias_resolutions": substitutions,
        "unsupported_requested_models": invalid,
    }
    write_json(_workspace_paths(workspace_dir)["model_plan_json"], payload)
    message = f"Selected models saved: {', '.join(payload['selected_models'])}."
    if substitutions:
        message += f" Resolved aliases: {substitutions}."
    if invalid:
        message += f" Ignored unsupported requested models: {invalid}."
    return message


@tool
def prepare_splits(plan: SplitPlan, runtime: ToolRuntime) -> str:
    """Create train/validation/test splits for the current dataset."""
    workspace_dir = _get_workspace_dir(runtime)
    df = read_table(_active_dataset_path(runtime))
    task_type = _get_task_type(runtime)
    target_column = _get_target_column(runtime)
    _validate_target_and_task(df, target_column, task_type)

    if not 0 < plan.test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5.")
    if not 0 < plan.val_size < 0.4:
        raise ValueError("val_size must be between 0 and 0.4.")

    paths = _workspace_paths(workspace_dir)
    split_dir = paths["split_dir"]
    split_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "clustering":
        train_df, test_df = train_test_split(df, test_size=plan.test_size, random_state=42)
        relative_val = plan.val_size / (1 - plan.test_size)
        train_df, val_df = train_test_split(train_df, test_size=relative_val, random_state=42)
    else:
        stratify_values = None
        if task_type == "classification" and plan.stratify and target_column:
            class_counts = df[target_column].value_counts(dropna=False)
            if not class_counts.empty and int(class_counts.min()) >= 2:
                stratify_values = df[target_column]
        train_df, test_df = train_test_split(
            df,
            test_size=plan.test_size,
            random_state=42,
            stratify=stratify_values,
        )
        relative_val = plan.val_size / (1 - plan.test_size)
        train_stratify = None
        if task_type == "classification" and plan.stratify and target_column:
            class_counts = train_df[target_column].value_counts(dropna=False)
            if not class_counts.empty and int(class_counts.min()) >= 2:
                train_stratify = train_df[target_column]
        train_df, val_df = train_test_split(
            train_df,
            test_size=relative_val,
            random_state=42,
            stratify=train_stratify,
        )

    write_table(train_df, split_dir / "train.csv")
    write_table(val_df, split_dir / "val.csv")
    write_table(test_df, split_dir / "test.csv")
    write_json(
        split_dir / "split_manifest.json",
        {
            "task_type": task_type,
            "target_column": target_column,
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "test_size": plan.test_size,
            "val_size": plan.val_size,
        },
    )
    return f"Splits prepared in {split_dir}."


@tool
def train_models(runtime: ToolRuntime) -> str:
    """Train the shortlisted models and rank them on the validation split."""
    workspace_dir = _get_workspace_dir(runtime)
    paths = _workspace_paths(workspace_dir)
    task_type = _get_task_type(runtime)
    target_column = _get_target_column(runtime)
    model_plan = read_json(paths["model_plan_json"])
    selected_models = model_plan.get("selected_models", [])
    if not selected_models:
        raise ValueError("No selected models found. Run select_candidate_models first.")

    split_dir = paths["split_dir"]
    train_df = read_table(split_dir / "train.csv")
    val_df = read_table(split_dir / "val.csv")
    models_dir = paths["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)

    leaderboard: List[Dict[str, Any]] = []
    for model_name in selected_models:
        if task_type == "clustering":
            result = _fit_clustering_model(model_name, train_df)
            bundle = {
                "preprocessor": result["preprocessor"],
                "model": result["model"],
                "metadata": result["metadata"],
            }
        else:
            result = _fit_supervised_model(
                model_name=model_name,
                task_type=task_type,
                train_df=train_df,
                val_df=val_df,
                target_column=target_column,
            )
            bundle = {
                "pipeline": result["pipeline"],
                "label_encoder": result["label_encoder"],
                "metadata": result["metadata"],
            }

        model_path = models_dir / f"{model_name}_validation.joblib"
        joblib.dump(bundle, model_path)

        leaderboard.append(
            {
                "model_name": model_name,
                "validation_metrics": result["metrics"],
                "selection_score": result["selection_score"],
                "model_path": str(model_path),
            }
        )

    leaderboard.sort(key=lambda item: item["selection_score"], reverse=True)
    payload = {
        "task_type": task_type,
        "target_column": target_column,
        "leaderboard": leaderboard,
        "best_model_name": leaderboard[0]["model_name"] if leaderboard else None,
    }
    write_json(paths["leaderboard_json"], payload)
    best_name = payload["best_model_name"]
    return f"Trained {len(leaderboard)} models. Current leader: {best_name}."


@tool
def evaluate_models(runtime: ToolRuntime) -> str:
    """Evaluate the best model from the current run on the hold-out test split."""
    workspace_dir = _get_workspace_dir(runtime)
    paths = _workspace_paths(workspace_dir)
    task_type = _get_task_type(runtime)
    target_column = _get_target_column(runtime)
    leaderboard = read_json(paths["leaderboard_json"]).get("leaderboard", [])
    if not leaderboard:
        raise ValueError("Leaderboard is empty. Run train_models first.")

    best_record = leaderboard[0]
    best_model_name = best_record["model_name"]
    split_dir = paths["split_dir"]
    train_df = read_table(split_dir / "train.csv")
    val_df = read_table(split_dir / "val.csv")
    test_df = read_table(split_dir / "test.csv")

    if task_type == "clustering":
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        result = _fit_clustering_model(best_model_name, combined_df)
        bundle = {
            "preprocessor": result["preprocessor"],
            "model": result["model"],
            "metadata": result["metadata"],
        }
        test_metrics = result["metrics"]
    else:
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        result = _fit_supervised_model(
            model_name=best_model_name,
            task_type=task_type,
            train_df=combined_df,
            val_df=test_df,
            target_column=target_column,
        )
        bundle = {
            "pipeline": result["pipeline"],
            "label_encoder": result["label_encoder"],
            "metadata": result["metadata"],
        }
        test_metrics = result["metrics"]

    joblib.dump(bundle, paths["current_best_model"])
    payload = {
        "task_type": task_type,
        "target_column": target_column,
        "best_model_name": best_model_name,
        "validation_metrics": best_record["validation_metrics"],
        "test_metrics": test_metrics,
        "current_best_model_path": str(paths["current_best_model"]),
        "selection_score": _selection_score(task_type, test_metrics),
    }
    write_json(paths["evaluation_json"], payload)
    return f"Best model `{best_model_name}` evaluated. Test metrics saved to {paths['evaluation_json']}."


@tool
def load_long_term_memory(runtime: ToolRuntime) -> str:
    """Read the persisted long-term memory with the best historical result."""
    workspace_dir = _get_workspace_dir(runtime)
    history_path = _workspace_paths(workspace_dir)["history_json"]
    history = read_json(history_path, default={"best_run": None})
    best_run = history.get("best_run")
    if not best_run:
        return "Long-term memory is empty."
    return (
        "Loaded long-term memory. "
        f"Best historical model: {best_run.get('best_model_name')} with score {best_run.get('selection_score')}."
    )


@tool
def save_best_model(runtime: ToolRuntime) -> str:
    """Compare the current run with history and persist the best model across runs."""
    workspace_dir = _get_workspace_dir(runtime)
    paths = _workspace_paths(workspace_dir)
    evaluation = read_json(paths["evaluation_json"])
    if not evaluation:
        raise ValueError("No evaluation found. Run evaluate_models first.")

    history = read_json(paths["history_json"], default={"best_run": None})
    previous = history.get("best_run")
    current_score = float(evaluation["selection_score"])
    should_update = previous is None or current_score > float(previous["selection_score"])

    payload = {
        "best_run": {
            "best_model_name": evaluation["best_model_name"],
            "selection_score": current_score,
            "task_type": evaluation["task_type"],
            "target_column": evaluation["target_column"],
            "model_path": str(paths["memory_model"]),
            "source_run_model_path": evaluation["current_best_model_path"],
        }
    }

    if should_update:
        paths["memory_dir"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths["current_best_model"], paths["memory_model"])
        write_json(paths["history_json"], payload)
        write_json(paths["memory_meta"], payload["best_run"])
        return "Current run improved the historical result. Best model updated in long-term memory."

    return "Current run did not beat the historical best model. Long-term memory kept unchanged."


@tool
def write_report(runtime: ToolRuntime) -> str:
    """Write a final markdown and JSON report for the current run."""
    workspace_dir = _get_workspace_dir(runtime)
    paths = _workspace_paths(workspace_dir)
    profile = read_json(paths["profile_json"])
    goal = read_json(paths["goal_json"])
    cleaning = read_json(paths["cleaning_json"])
    eda = read_json(paths["eda_json"])
    features = read_json(paths["feature_json"])
    model_plan = read_json(paths["model_plan_json"])
    evaluation = read_json(paths["evaluation_json"])
    history = read_json(paths["history_json"], default={"best_run": None})

    report_json = {
        "business_task": runtime.state["business_task"],
        "dataset_path": runtime.state["dataset_path"],
        "target_column": goal.get("target_column"),
        "task_type": goal.get("task_type"),
        "dataset_shape": profile.get("shape"),
        "cleaning": cleaning,
        "eda": eda,
        "features": features,
        "selected_models": model_plan.get("selected_models", []),
        "evaluation": evaluation,
        "historical_best": history.get("best_run"),
        "prompting_techniques": [
            "role prompting",
            "contrastive prompting against leakage and wrong model choice",
            "phase-specific self-check prompting",
            "tool routing by workflow phase",
        ],
    }

    markdown_lines = [
        "# Итоговый отчет по запуску агента",
        "",
        f"**Бизнес-задача:** {runtime.state['business_task']}",
        f"**Исходный датасет:** {runtime.state['dataset_path']}",
        f"**Тип задачи:** {goal.get('task_type')}",
        f"**Целевая переменная:** {goal.get('target_column')}",
        "",
        "## Анализ датасета",
        f"- Размер: {profile.get('shape', {}).get('rows')} строк, {profile.get('shape', {}).get('columns')} колонок",
        f"- Удалено дублей: {cleaning.get('dropped_duplicates')}",
        f"- Удалено строк без target: {cleaning.get('target_rows_removed')}",
        "",
        "## EDA",
        f"- Потенциальные утечки: {', '.join(eda.get('leakage_candidates', [])) or 'не обнаружены'}",
        "",
        "## Feature Engineering",
        f"- Новые признаки: {', '.join(features.get('created_features', [])) or 'не создавались'}",
        "",
        "## Модели",
        f"- Выбранные алгоритмы: {', '.join(model_plan.get('selected_models', []))}",
        f"- Лучшая модель текущего запуска: {evaluation.get('best_model_name')}",
        f"- Validation metrics: {evaluation.get('validation_metrics')}",
        f"- Test metrics: {evaluation.get('test_metrics')}",
        "",
        "## Долговременная память",
        f"- Исторический лучший результат: {history.get('best_run')}",
        "",
        "## Использованные техники промптинга",
        "- role prompting",
        "- contrastive prompting",
        "- self-check prompting",
        "- phase-specific tool routing",
    ]

    write_json(paths["report_json"], report_json)
    write_text(paths["report_md"], "\n".join(markdown_lines))
    return f"Final report written to {paths['report_md']}."
