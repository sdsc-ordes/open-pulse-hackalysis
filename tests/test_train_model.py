"""Tests for prediction.train_model module."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.train_model import train_and_save


@pytest.fixture
def synthetic_csv(tmp_path):
    """Create a minimal CSV that mimics repo_metadata_with_predictions.csv."""
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

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_train_and_save_creates_all_files(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    train_and_save(synthetic_csv, models_dir=models_dir)

    assert (models_dir / "features.json").exists()
    assert (models_dir / "correlation_weights.json").exists()
    assert (models_dir / "rf_model.pkl").exists()
    assert (models_dir / "model_metadata.json").exists()


def test_train_and_save_features_json(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    train_and_save(synthetic_csv, models_dir=models_dir)

    with open(models_dir / "features.json") as f:
        features = json.load(f)

    assert isinstance(features, list)
    assert len(features) > 0
    assert "is_linked_project" not in features
    assert "true_hackathon_repos" not in features


def test_train_and_save_correlation_weights(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    train_and_save(synthetic_csv, models_dir=models_dir)

    with open(models_dir / "correlation_weights.json") as f:
        data = json.load(f)

    assert "weights" in data
    assert "feat_min" in data
    assert "feat_max" in data
    assert "threshold" in data
    assert len(data["weights"]) > 0


def test_train_and_save_rf_model_loadable(synthetic_csv, tmp_path):
    from hackathon_analysis.prediction.random_forest import TrainedRandomForest

    models_dir = tmp_path / "models"
    train_and_save(synthetic_csv, models_dir=models_dir)

    loaded = TrainedRandomForest.load(models_dir / "rf_model.pkl")
    assert len(loaded.features) > 0
    assert len(loaded.importances) > 0


def test_train_and_save_metadata(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    metadata = train_and_save(synthetic_csv, models_dir=models_dir)

    assert metadata["total_repos"] == 80
    assert metadata["valid_repos"] == 80
    assert metadata["n_features"] > 0
    assert 0 < metadata["cv_accuracy"] <= 1.0

    with open(models_dir / "model_metadata.json") as f:
        saved = json.load(f)
    assert saved["n_features"] == metadata["n_features"]


def test_train_and_save_missing_target(synthetic_csv, tmp_path):
    models_dir = tmp_path / "models"
    with pytest.raises(ValueError, match="Target column"):
        train_and_save(synthetic_csv, models_dir=models_dir, target="nonexistent")


def test_train_and_save_missing_file(tmp_path):
    with pytest.raises((FileNotFoundError, Exception)):
        train_and_save(tmp_path / "nonexistent.csv")
