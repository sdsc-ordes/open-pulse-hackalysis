"""Tests for prediction.pipeline module."""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.pipeline import (
    predict_repo,
    predict_repo_pretty,
    fetch_single_repo_metadata,
)
from hackathon_analysis.prediction.train_model import train_and_save


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


FAKE_GITHUB_RESPONSE = {
    "https://github.com/test/hackathon-app": {
        "owner": "test",
        "repo": "hackathon-app",
        "repo_name": "hackathon-app",
        "name_with_owner": "test/hackathon-app",
        "url": "https://github.com/test/hackathon-app",
        "description": "Built at a hackathon in 24 hours",
        "primary_language": "Python",
        "stars": 3,
        "forks": 1,
        "watchers": 2,
        "is_fork": False,
        "is_archived": False,
        "created_at": "2024-11-20T10:00:00Z",
        "updated_at": "2024-11-22T18:00:00Z",
        "pushed_at": "2024-11-22T18:00:00Z",
        "default_branch": "main",
        "commit_count_default_branch": 12,
        "first_commit_date_default_branch": "2024-11-20T10:00:00Z",
        "last_commit_date_default_branch": "2024-11-21T22:00:00Z",
        "contributors_count": 3,
        "contributors_top": [],
        "readme_title": "Hackathon App",
        "readme_text": "Built at LauzHack 2024. A demo prototype.",
        "readme_length": 45,
        "topics": ["hackathon", "python", "demo"],
        "files_total_count": 15,
        "dirs_total_count": 4,
        "issues_total": 2,
        "issues_open": 0,
        "issues_closed": 2,
        "pull_requests_total": 1,
        "pull_requests_open": 0,
        "pull_requests_closed": 1,
        "pull_requests_merged": 1,
        "releases_count": 0,
        "has_ci": False,
        "has_tests": False,
        "has_docker": False,
        "has_docs": False,
        "has_readme_file": True,
        "has_license_file": False,
        "has_contributing": False,
        "has_notebooks": True,
        "n_concepts": 0,
    }
}


@pytest.fixture
def trained_models_dir(tmp_path):
    """Train models on synthetic data and return the models directory."""
    rng = np.random.RandomState(42)
    n = 80
    target = rng.choice([True, False], size=n, p=[0.4, 0.6])

    df = pd.DataFrame({
        "repo_url": [f"https://github.com/user/repo-{i}" for i in range(n)],
        "url": [f"https://github.com/user/repo-{i}" for i in range(n)],
        "error": [""] * n,
        "project_foreign_key": [""] * n,
        "true_hackathon_repos": target,
        "created_at": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "pushed_at": pd.date_range("2023-06-01", periods=n, freq="D").astype(str),
        "first_commit_date_default_branch": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "last_commit_date_default_branch": pd.date_range("2023-06-01", periods=n, freq="D").astype(str),
        "commit_count_default_branch": rng.randint(1, 200, n),
        "stars": rng.randint(0, 50, n),
        "forks": rng.randint(0, 20, n),
        "watchers": rng.randint(0, 10, n),
        "contributors_count": rng.randint(1, 10, n),
        "contributors_top": ["[]"] * n,
        "files_total_count": rng.randint(1, 100, n),
        "dirs_total_count": rng.randint(1, 30, n),
        "readme_length": rng.randint(0, 5000, n),
        "issues_total": rng.randint(0, 20, n),
        "issues_open": rng.randint(0, 5, n),
        "issues_closed": rng.randint(0, 15, n),
        "pull_requests_total": rng.randint(0, 10, n),
        "pull_requests_open": rng.randint(0, 3, n),
        "pull_requests_closed": rng.randint(0, 5, n),
        "pull_requests_merged": rng.randint(0, 5, n),
        "releases_count": rng.randint(0, 5, n),
        "n_concepts": rng.randint(0, 50, n),
        "has_ci": rng.choice([True, False], n),
        "has_tests": rng.choice([True, False], n),
        "has_docker": rng.choice([True, False], n),
        "has_docs": rng.choice([True, False], n),
        "has_readme_file": rng.choice([True, False], n),
        "has_license_file": rng.choice([True, False], n),
        "has_contributing": rng.choice([True, False], n),
        "has_notebooks": rng.choice([True, False], n),
        "is_fork": rng.choice([True, False], n),
        "is_archived": rng.choice([True, False], n),
    })

    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    models_dir = tmp_path / "models"
    train_and_save(csv_path, models_dir=models_dir)
    return models_dir


# ---------------------------------------------------------------------------
# Tests: predict_repo (with mocked GitHub API)
# ---------------------------------------------------------------------------


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_returns_all_methods(mock_fetch, trained_models_dir):
    """Full pipeline should return predictions from all 3 methods."""
    # Clear cached models so it loads from our test dir
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = FAKE_GITHUB_RESPONSE
    url = "https://github.com/test/hackathon-app"

    result = predict_repo(url, github_token="fake-token", models_dir=trained_models_dir)

    assert result["repo_url"] == url
    assert "naive_rule_based" in result
    assert "correlation_weighted" in result
    assert "random_forest" in result
    assert "metadata" in result


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_naive_detects_hackathon(mock_fetch, trained_models_dir):
    """Repo with 'hackathon' in readme should trigger naive rule detection."""
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = FAKE_GITHUB_RESPONSE
    url = "https://github.com/test/hackathon-app"

    result = predict_repo(url, github_token="fake-token", models_dir=trained_models_dir)

    assert result["naive_rule_based"]["score"] >= 3
    assert "strong_repo_text" in result["naive_rule_based"]["signals"]


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_metadata_populated(mock_fetch, trained_models_dir):
    """Metadata section should contain repo info from GitHub."""
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = FAKE_GITHUB_RESPONSE
    url = "https://github.com/test/hackathon-app"

    result = predict_repo(url, github_token="fake-token", models_dir=trained_models_dir)
    meta = result["metadata"]

    assert meta["owner"] == "test"
    assert meta["language"] == "Python"
    assert meta["stars"] == 3
    assert meta["commits"] == 12


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_scores_are_valid(mock_fetch, trained_models_dir):
    """Correlation and RF scores should be in valid ranges."""
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = FAKE_GITHUB_RESPONSE
    url = "https://github.com/test/hackathon-app"

    result = predict_repo(url, github_token="fake-token", models_dir=trained_models_dir)

    assert 0 <= result["correlation_weighted"]["score"] <= 1
    assert 0 <= result["random_forest"]["probability"] <= 1
    assert isinstance(result["naive_rule_based"]["flag"], bool)
    assert isinstance(result["correlation_weighted"]["flag"], bool)
    assert isinstance(result["random_forest"]["flag"], bool)


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_pretty_output(mock_fetch, trained_models_dir):
    """Pretty output should be a non-empty string with key sections."""
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = FAKE_GITHUB_RESPONSE
    url = "https://github.com/test/hackathon-app"

    output = predict_repo_pretty(url, github_token="fake-token", models_dir=trained_models_dir)

    assert isinstance(output, str)
    assert "hackathon-app" in output
    assert "Naive Rules" in output
    assert "Correlation-Weighted" in output
    assert "Random Forest" in output
    assert "Consensus" in output


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


@patch("hackathon_analysis.prediction.pipeline.fetch_repo_metadata")
def test_predict_repo_github_error(mock_fetch, trained_models_dir):
    """Should raise when GitHub returns no data."""
    import hackathon_analysis.prediction.pipeline as pipeline_mod
    pipeline_mod._cached_corr_weights = None
    pipeline_mod._cached_rf_model = None

    mock_fetch.return_value = {}  # empty response
    url = "https://github.com/nonexistent/repo"

    with pytest.raises(RuntimeError, match="no data"):
        predict_repo(url, github_token="fake-token", models_dir=trained_models_dir)
