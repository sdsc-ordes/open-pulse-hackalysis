"""Compute derived features from raw GitHub repo metadata.

Extracted from repo_analysis.ipynb cells 8, 42, 46, 58.
These functions take a DataFrame of raw repo metadata and add
the derived columns needed by the prediction methods.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# String helpers (from notebook cell 8)
# ---------------------------------------------------------------------------


def _clean_string_series(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _non_empty_string_mask(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name not in df.columns:
        return pd.Series(False, index=df.index)
    return _clean_string_series(df[column_name]).ne("")


# ---------------------------------------------------------------------------
# Core feature builder (from notebook cell 8: build_repo_analysis_features)
# ---------------------------------------------------------------------------


def build_repo_analysis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived boolean and numeric features to a repo metadata DataFrame.

    This is the central feature engineering function. It takes raw GitHub
    metadata columns and computes everything the predictors need:
    - has_error, has_valid_metadata, is_linked_project (status flags)
    - repo_age_days, active_days_default_branch (temporal)
    - commit_log, stars_log, contributors_count_log (log-scaled)
    - contributors_total (sum of count + top list length)
    """
    df = df.copy()

    # Status flags
    df["has_error"] = _non_empty_string_mask(df, "error")
    df["has_valid_metadata"] = ~df["has_error"] & _non_empty_string_mask(df, "url")
    df["is_linked_project"] = _non_empty_string_mask(df, "project_foreign_key")

    # Temporal features
    if {"pushed_at", "created_at"}.issubset(df.columns):
        pushed = pd.to_datetime(df["pushed_at"], errors="coerce")
        created = pd.to_datetime(df["created_at"], errors="coerce")
        df["repo_age_days"] = (pushed - created).dt.days
    else:
        df["repo_age_days"] = pd.NA

    if {
        "last_commit_date_default_branch",
        "first_commit_date_default_branch",
    }.issubset(df.columns):
        last_commit = pd.to_datetime(
            df["last_commit_date_default_branch"], errors="coerce"
        )
        first_commit = pd.to_datetime(
            df["first_commit_date_default_branch"], errors="coerce"
        )
        df["active_days_default_branch"] = (last_commit - first_commit).dt.days
    else:
        df["active_days_default_branch"] = pd.NA

    # Log-scaled features
    df["commit_log"] = np.log1p(
        pd.to_numeric(df.get("commit_count_default_branch", 0), errors="coerce").fillna(0)
    )
    df["stars_log"] = np.log1p(
        pd.to_numeric(df.get("stars", 0), errors="coerce").fillna(0)
    )
    df["contributors_count_log"] = np.log1p(
        pd.to_numeric(df.get("contributors_count", 0), errors="coerce").fillna(0)
    )

    # Contributors total (count + length of top contributors list)
    contrib_count = pd.to_numeric(
        df.get("contributors_count", 0), errors="coerce"
    ).fillna(0)
    contrib_top_len = pd.Series(0, index=df.index)
    if "contributors_top" in df.columns:
        contrib_top_len = df["contributors_top"].apply(_count_list_items)
    df["contributors_total"] = (contrib_count + contrib_top_len).astype(int)

    return df


def _count_list_items(val) -> int:
    """Count items in a value that might be a list, JSON string, or scalar."""
    if pd.isna(val) or val == "":
        return 0
    if isinstance(val, list):
        return len(val)
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return len(parsed)
        except (json.JSONDecodeError, ValueError):
            pass
    return 0


# ---------------------------------------------------------------------------
# Feature selection (from notebook cell 58)
# ---------------------------------------------------------------------------

# Features excluded from ML models because they leak the label,
# are biased toward the training hackathon, or are metadata artifacts.
EXCLUDED_FEATURES = {
    # Label leakers
    "is_linked_project",
    "repo_predicted_score",
    "repo_predicted_flag",
    "corr_weighted_score",
    "corr_weighted_flag",
    "ml_predicted_flag",
    "ml_predicted_proba",
    # Metadata artifacts
    "has_error",
    "has_valid_metadata",
    # Biased concept features (too specific to one hackathon)
    "has_top_hackathon_concept",
    "concept_hackathon_score",
    "min_concept_projects",
    "max_concept_projects",
}


def select_features(
    df: pd.DataFrame,
    target: str = "true_hackathon_repos",
    collinearity_threshold: float = 0.85,
) -> list[str]:
    """Select clean numeric/boolean features for ML models.

    Steps:
    1. Exclude label leakers, metadata artifacts, biased concept features
    2. Exclude concept one-hot columns (concept_is_*)
    3. Exclude non-numeric/boolean columns
    4. Drop constant columns
    5. Remove multicollinear pairs (keep the one more correlated with target)

    Returns the list of selected feature names.
    """
    exclude = set(EXCLUDED_FEATURES)
    exclude.update(col for col in df.columns if col.startswith("concept_is_"))

    bool_cols = df.select_dtypes(include=["bool", "boolean"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    candidates = sorted(set(numeric_cols + bool_cols) - {target} - exclude)

    # Build feature matrix
    feature_matrix = df[candidates].copy()
    for col in bool_cols:
        if col in feature_matrix.columns:
            feature_matrix[col] = feature_matrix[col].astype(float)
    feature_matrix = feature_matrix.fillna(0)

    # Drop constant columns
    feature_matrix = feature_matrix.loc[:, feature_matrix.nunique(dropna=False) > 1]

    # Remove multicollinear features
    target_series = df[target].fillna(False).astype(float)
    target_corr = feature_matrix.corrwith(target_series).abs()

    corr_matrix = feature_matrix.corr().abs()
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop: set[str] = set()
    for col in upper.columns:
        correlated_with = upper.index[upper[col] > collinearity_threshold].tolist()
        for partner in correlated_with:
            if col in to_drop or partner in to_drop:
                continue
            if target_corr.get(col, 0) >= target_corr.get(partner, 0):
                to_drop.add(partner)
            else:
                to_drop.add(col)

    return sorted(set(feature_matrix.columns) - to_drop)
