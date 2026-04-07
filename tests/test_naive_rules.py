"""Tests for prediction.naive_rules module."""

import pandas as pd
import pytest

from hackathon_analysis.prediction.naive_rules import predict_naive


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_repo(**overrides) -> dict:
    """Build a single repo row with sensible defaults."""
    defaults = {
        "repo_url": "https://github.com/test/repo",
        "repo_name": "repo",
        "primary_language": "Python",
        "readme_title": "",
        "readme_text": "",
        "topics": "",
        "stars": 0,
        "forks": 0,
        "contributors_count": 0,
        "commit_count_default_branch": 0,
        "active_days_default_branch": 0,
        "error": "",
    }
    defaults.update(overrides)
    return defaults


def _predict_single(**overrides) -> pd.Series:
    """Run predict_naive on a single repo and return the result row."""
    df = pd.DataFrame([_make_repo(**overrides)])
    result = predict_naive(df)
    return result.iloc[0]


# ---------------------------------------------------------------------------
# Tests: text signal detection
# ---------------------------------------------------------------------------


def test_strong_text_in_readme():
    row = _predict_single(readme_text="Built at LauzHack 2024!")
    assert row["repo_predicted_score"] >= 3
    assert "strong_repo_text" in row["repo_predicted_signals"]


def test_strong_text_in_repo_name():
    row = _predict_single(repo_name="my-hackathon-project")
    assert "strong_repo_text" in row["repo_predicted_signals"]


def test_strong_text_in_topics():
    row = _predict_single(topics='["hackathon-project", "python"]')
    assert "strong_repo_text" in row["repo_predicted_signals"]


def test_strong_text_in_topics_list():
    row = _predict_single(topics=["devpost", "challenge"])
    assert "strong_repo_text" in row["repo_predicted_signals"]


def test_weak_text_match():
    row = _predict_single(readme_text="This is a prototype for the demo")
    assert "weak_repo_text" in row["repo_predicted_signals"]
    assert row["repo_predicted_score"] >= 1


def test_strong_overrides_weak():
    """Strong match should fire, not weak, even if both keywords present."""
    row = _predict_single(readme_text="hackathon prototype demo")
    assert "strong_repo_text" in row["repo_predicted_signals"]
    assert "weak_repo_text" not in row["repo_predicted_signals"]


def test_no_text_match():
    row = _predict_single(readme_text="A library for parsing JSON")
    assert "strong_repo_text" not in row["repo_predicted_signals"]
    assert "weak_repo_text" not in row["repo_predicted_signals"]


# ---------------------------------------------------------------------------
# Tests: GitHub stats signals
# ---------------------------------------------------------------------------


def test_popularity_signal_stars():
    row = _predict_single(stars=5)
    assert "repo_popularity_signal" in row["repo_predicted_signals"]


def test_popularity_signal_forks():
    row = _predict_single(forks=1)
    assert "repo_popularity_signal" in row["repo_predicted_signals"]


def test_no_popularity_signal():
    row = _predict_single(stars=0, forks=0)
    assert "repo_popularity_signal" not in row["repo_predicted_signals"]


def test_contributors_signal():
    row = _predict_single(contributors_count=3)
    assert "contributors_signal" in row["repo_predicted_signals"]


def test_no_contributors_signal():
    row = _predict_single(contributors_count=1)
    assert "contributors_signal" not in row["repo_predicted_signals"]


def test_commit_activity_signal():
    row = _predict_single(commit_count_default_branch=10)
    assert "commit_activity_signal" in row["repo_predicted_signals"]


def test_no_commit_activity_signal():
    row = _predict_single(commit_count_default_branch=3)
    assert "commit_activity_signal" not in row["repo_predicted_signals"]


# ---------------------------------------------------------------------------
# Tests: burst commit window
# ---------------------------------------------------------------------------


def test_strong_burst_window():
    row = _predict_single(active_days_default_branch=2)
    assert "burst_commit_window_strong" in row["repo_predicted_signals"]
    assert row["repo_predicted_score"] >= 2


def test_weak_burst_window():
    row = _predict_single(active_days_default_branch=5)
    assert "burst_commit_window_weak" in row["repo_predicted_signals"]


def test_no_burst_window():
    row = _predict_single(active_days_default_branch=30)
    assert "burst_commit_window_strong" not in row["repo_predicted_signals"]
    assert "burst_commit_window_weak" not in row["repo_predicted_signals"]


def test_zero_active_days_no_burst():
    """active_days=0 should NOT trigger burst (could mean missing data)."""
    row = _predict_single(active_days_default_branch=0)
    assert "burst_commit_window_strong" not in row["repo_predicted_signals"]
    assert "burst_commit_window_weak" not in row["repo_predicted_signals"]


# ---------------------------------------------------------------------------
# Tests: metadata error
# ---------------------------------------------------------------------------


def test_metadata_error_penalty():
    row = _predict_single(error="404 Not Found")
    assert row["repo_predicted_score"] == -1
    assert "metadata_error" in row["repo_predicted_signals"]


def test_metadata_error_no_penalty_with_other_signals():
    """Error penalty only applies when score is 0 (no other signals)."""
    row = _predict_single(error="404 Not Found", stars=5)
    assert "metadata_error" not in row["repo_predicted_signals"]


def test_nan_error_no_penalty():
    """NaN in error column should NOT trigger penalty."""
    row = _predict_single(error=float("nan"))
    assert "metadata_error" not in row["repo_predicted_signals"]


# ---------------------------------------------------------------------------
# Tests: flag threshold
# ---------------------------------------------------------------------------


def test_flag_true_at_threshold():
    """Score >= 4 should produce flag=True."""
    row = _predict_single(
        readme_text="hackathon project",  # +3
        stars=1,                           # +1
    )
    assert row["repo_predicted_score"] >= 4
    assert row["repo_predicted_flag"] == True  # noqa: E712 (numpy bool)


def test_flag_false_below_threshold():
    """Score < 4 should produce flag=False."""
    row = _predict_single(stars=1, forks=1)  # +1 only
    assert row["repo_predicted_score"] < 4
    assert row["repo_predicted_flag"] == False  # noqa: E712 (numpy bool)


# ---------------------------------------------------------------------------
# Tests: output columns
# ---------------------------------------------------------------------------


def test_output_columns_present():
    df = pd.DataFrame([_make_repo()])
    result = predict_naive(df)
    assert "repo_predicted_score" in result.columns
    assert "repo_predicted_flag" in result.columns
    assert "repo_predicted_signals" in result.columns


def test_does_not_modify_input():
    df = pd.DataFrame([_make_repo()])
    original_cols = set(df.columns)
    predict_naive(df)
    assert set(df.columns) == original_cols


def test_no_signal_label():
    """Repo with no matching signals should get 'no_signal'."""
    row = _predict_single()
    assert row["repo_predicted_signals"] == "no_signal"


# ---------------------------------------------------------------------------
# Tests: column name variants
# ---------------------------------------------------------------------------


def test_repo_topics_column_variant():
    """Should also read from 'repo_topics' if 'topics' is missing."""
    repo = _make_repo()
    del repo["topics"]
    repo["repo_topics"] = '["hackathon"]'
    df = pd.DataFrame([repo])
    result = predict_naive(df)
    assert "strong_repo_text" in result.iloc[0]["repo_predicted_signals"]


def test_repo_error_column_variant():
    """Should also read from 'repo_error' if 'error' is missing."""
    repo = _make_repo()
    del repo["error"]
    repo["repo_error"] = "404 Not Found"
    df = pd.DataFrame([repo])
    result = predict_naive(df)
    assert "metadata_error" in result.iloc[0]["repo_predicted_signals"]
