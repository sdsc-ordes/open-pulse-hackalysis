"""Tests for prediction.feature_engineering module."""

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.feature_engineering import (
    build_repo_analysis_features,
    select_features,
    _count_list_items,
    _non_empty_string_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_repo_df():
    """A small DataFrame mimicking raw GitHub repo metadata."""
    return pd.DataFrame({
        "repo_url": [
            "https://github.com/alice/hackathon-app",
            "https://github.com/bob/my-library",
            "https://github.com/carol/broken-repo",
        ],
        "url": [
            "https://github.com/alice/hackathon-app",
            "https://github.com/bob/my-library",
            "",
        ],
        "error": ["", "", "404 Not Found"],
        "project_foreign_key": ["lauzhack:2024:proj1", "", ""],
        "created_at": ["2024-11-20T10:00:00Z", "2020-01-15T08:00:00Z", None],
        "pushed_at": ["2024-11-22T18:00:00Z", "2024-06-01T12:00:00Z", None],
        "first_commit_date_default_branch": [
            "2024-11-20T10:00:00Z",
            "2020-01-15T08:00:00Z",
            None,
        ],
        "last_commit_date_default_branch": [
            "2024-11-22T18:00:00Z",
            "2024-06-01T12:00:00Z",
            None,
        ],
        "commit_count_default_branch": [25, 500, 0],
        "stars": [2, 150, 0],
        "contributors_count": [3, 12, 0],
        "contributors_top": ['[{"login":"a"},{"login":"b"}]', "[]", ""],
    })


@pytest.fixture
def feature_df_with_target():
    """DataFrame with features + target for select_features testing."""
    rng = np.random.RandomState(42)
    n = 100
    target = rng.choice([True, False], size=n, p=[0.4, 0.6])
    return pd.DataFrame({
        "true_hackathon_repos": target,
        "stars": rng.randint(0, 100, n),
        "forks": rng.randint(0, 50, n),
        "commit_count_default_branch": rng.randint(1, 200, n),
        "has_tests": rng.choice([True, False], n),
        "has_docs": rng.choice([True, False], n),
        # Constant column — should be dropped
        "always_zero": np.zeros(n, dtype=int),
        # Label leaker — should be excluded
        "is_linked_project": target,
        # Concept one-hot — should be excluded
        "concept_is_python": rng.choice([True, False], n),
    })


# ---------------------------------------------------------------------------
# Tests: _count_list_items
# ---------------------------------------------------------------------------


def test_count_list_items_with_list():
    assert _count_list_items([1, 2, 3]) == 3


def test_count_list_items_with_json_string():
    assert _count_list_items('[{"login":"a"},{"login":"b"}]') == 2


def test_count_list_items_with_empty():
    assert _count_list_items("") == 0
    assert _count_list_items(None) == 0
    assert _count_list_items(float("nan")) == 0


def test_count_list_items_with_plain_string():
    assert _count_list_items("not a list") == 0


# ---------------------------------------------------------------------------
# Tests: _non_empty_string_mask
# ---------------------------------------------------------------------------


def test_non_empty_string_mask():
    df = pd.DataFrame({"col": ["hello", "", None, "  ", "world"]})
    mask = _non_empty_string_mask(df, "col")
    assert mask.tolist() == [True, False, False, False, True]


def test_non_empty_string_mask_missing_column():
    df = pd.DataFrame({"other": [1, 2]})
    mask = _non_empty_string_mask(df, "missing_col")
    assert mask.tolist() == [False, False]


# ---------------------------------------------------------------------------
# Tests: build_repo_analysis_features
# ---------------------------------------------------------------------------


def test_build_adds_status_flags(minimal_repo_df):
    result = build_repo_analysis_features(minimal_repo_df)

    assert result["has_error"].tolist() == [False, False, True]
    assert result["has_valid_metadata"].tolist() == [True, True, False]
    assert result["is_linked_project"].tolist() == [True, False, False]


def test_build_computes_temporal_features(minimal_repo_df):
    result = build_repo_analysis_features(minimal_repo_df)

    # alice: 2 days between created and pushed
    assert result.loc[0, "repo_age_days"] == 2
    # alice: 2 days between first and last commit
    assert result.loc[0, "active_days_default_branch"] == 2
    # bob: ~1598 days
    assert result.loc[1, "repo_age_days"] > 1500
    # carol: no dates → NaN
    assert pd.isna(result.loc[2, "repo_age_days"])


def test_build_computes_log_features(minimal_repo_df):
    result = build_repo_analysis_features(minimal_repo_df)

    assert result.loc[0, "commit_log"] == pytest.approx(np.log1p(25))
    assert result.loc[1, "stars_log"] == pytest.approx(np.log1p(150))
    assert result.loc[2, "commit_log"] == pytest.approx(0.0)


def test_build_computes_contributors_total(minimal_repo_df):
    result = build_repo_analysis_features(minimal_repo_df)

    # alice: 3 (count) + 2 (top list) = 5
    assert result.loc[0, "contributors_total"] == 5
    # bob: 12 (count) + 0 (empty list) = 12
    assert result.loc[1, "contributors_total"] == 12


def test_build_does_not_modify_input(minimal_repo_df):
    original_cols = set(minimal_repo_df.columns)
    build_repo_analysis_features(minimal_repo_df)
    assert set(minimal_repo_df.columns) == original_cols


# ---------------------------------------------------------------------------
# Tests: select_features
# ---------------------------------------------------------------------------


def test_select_excludes_leakers(feature_df_with_target):
    features = select_features(feature_df_with_target)
    assert "is_linked_project" not in features
    assert "true_hackathon_repos" not in features


def test_select_excludes_concept_onehot(feature_df_with_target):
    features = select_features(feature_df_with_target)
    assert "concept_is_python" not in features


def test_select_drops_constant_columns(feature_df_with_target):
    features = select_features(feature_df_with_target)
    assert "always_zero" not in features


def test_select_returns_valid_features(feature_df_with_target):
    features = select_features(feature_df_with_target)
    # Should keep the real features
    assert "stars" in features
    assert "has_tests" in features
    assert len(features) > 0


def test_select_removes_collinear_features():
    """Two perfectly correlated features — one should be dropped."""
    rng = np.random.RandomState(42)
    n = 100
    target = rng.choice([True, False], size=n, p=[0.4, 0.6])
    vals = rng.randint(0, 100, n)
    df = pd.DataFrame({
        "true_hackathon_repos": target,
        "feature_a": vals,
        "feature_b": vals * 2,  # perfectly correlated with feature_a
        "feature_c": rng.randint(0, 50, n),  # independent
    })
    features = select_features(df, collinearity_threshold=0.85)
    # One of feature_a/feature_b should be dropped, not both
    has_a = "feature_a" in features
    has_b = "feature_b" in features
    assert has_a != has_b, "Exactly one of the collinear pair should survive"
    assert "feature_c" in features
