"""Tests for prediction.pipeline module."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.pipeline import (
    predict_repo,
    predict_repo_pretty,
    fetch_single_repo_metadata,
    _ensure_model_file_from_hf,
    _load_correlation_weights,
    _load_rf_model,
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


# ---------------------------------------------------------------------------
# Tests: Model loading with HF Hub fallback (local-first pattern)
# ---------------------------------------------------------------------------


class TestEnsureModelFileFromHF:
    """Tests for the _ensure_model_file_from_hf() function (local-first pattern)."""

    def test_local_file_exists_returns_immediately(self, trained_models_dir):
        """If model file exists locally, return path immediately (fast path)."""
        # The correlation_weights.json file was created by train_and_save()
        result = _ensure_model_file_from_hf(trained_models_dir, "correlation_weights.json")

        assert result.exists()
        assert result.name == "correlation_weights.json"
        assert result.parent == trained_models_dir

    def test_creates_models_dir_if_missing(self, tmp_path):
        """Should create models_dir if it doesn't exist."""
        models_dir = tmp_path / "models" / "subdir"
        assert not models_dir.exists()

        # Create a dummy file to prevent actual HF download
        dummy_file = models_dir / "test_model.pkl"
        models_dir.mkdir(parents=True, exist_ok=True)
        dummy_file.write_text("dummy")

        result = _ensure_model_file_from_hf(models_dir, "test_model.pkl")

        assert result.exists()
        assert result.parent == models_dir

    @patch("hackathon_analysis.prediction.pipeline.hf_hub_download")
    @patch("hackathon_analysis.prediction.pipeline.get_hf_repo_from_env")
    def test_downloads_from_hf_if_local_missing(self, mock_get_hf, mock_download, tmp_path):
        """If local file missing, should download from HF Hub."""
        from hackathon_analysis.data_extraction.dataset_resolver import HFRepo

        # Setup mocks
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        mock_get_hf.return_value = HFRepo(
            repo_id="test/repo",
            repo_type="dataset"
        )

        # Setup mock to create file when called (simulating HF download)
        def create_and_return_file(*args, **kwargs):
            downloaded_file = models_dir / "rf_model.pkl"
            downloaded_file.write_text("model data")
            return str(downloaded_file)

        mock_download.side_effect = create_and_return_file

        result = _ensure_model_file_from_hf(models_dir, "rf_model.pkl")

        # Verify download was called with correct parameters
        mock_download.assert_called_once()
        call_args = mock_download.call_args
        assert call_args[1]["repo_id"] == "test/repo"
        assert call_args[1]["filename"] == "models/rf_model.pkl"
        assert call_args[1]["repo_type"] == "dataset"

        # Verify result points to the downloaded file
        assert result.exists()

    @patch("hackathon_analysis.prediction.pipeline.hf_hub_download")
    @patch("hackathon_analysis.prediction.pipeline.get_hf_repo_from_env")
    def test_raises_error_when_download_fails(self, mock_get_hf, mock_download, tmp_path):
        """If local missing and HF download fails, should raise FileNotFoundError."""
        from hackathon_analysis.data_extraction.dataset_resolver import HFRepo

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        mock_get_hf.return_value = HFRepo(
            repo_id="test/repo",
            repo_type="dataset"
        )
        mock_download.side_effect = Exception("Network error")

        with pytest.raises(FileNotFoundError, match="could not be downloaded"):
            _ensure_model_file_from_hf(models_dir, "missing_model.pkl")


class TestLoadCorrelationWeightsWithHF:
    """Tests for _load_correlation_weights() with HF fallback."""

    def test_loads_from_local_file(self, trained_models_dir):
        """Should load correlation weights from local file."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        # Reset cache to force reload
        pipeline_mod._cached_corr_weights = None

        weights = _load_correlation_weights(trained_models_dir)

        assert weights is not None
        assert hasattr(weights, "to_dict")  # CorrelationWeights has to_dict()

    def test_caches_after_first_load(self, trained_models_dir):
        """Should cache weights in memory after first load."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_corr_weights = None

        # First load
        weights1 = _load_correlation_weights(trained_models_dir)
        # Second load should return cached version
        weights2 = _load_correlation_weights(trained_models_dir)

        assert weights1 is weights2  # Same object (cached)

    @patch("hackathon_analysis.prediction.pipeline._ensure_model_file_from_hf")
    def test_calls_ensure_before_loading(self, mock_ensure, trained_models_dir):
        """Should call _ensure_model_file_from_hf before loading."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_corr_weights = None

        # Setup mock to return actual file
        corr_file = trained_models_dir / "correlation_weights.json"
        mock_ensure.return_value = corr_file

        weights = _load_correlation_weights(trained_models_dir)

        # Verify ensure was called with correct parameters
        mock_ensure.assert_called_once_with(trained_models_dir, "correlation_weights.json")
        assert weights is not None


class TestLoadRandomForestModelWithHF:
    """Tests for _load_rf_model() with HF fallback."""

    def test_loads_from_local_file(self, trained_models_dir):
        """Should load RF model from local pickle file."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_rf_model = None

        model = _load_rf_model(trained_models_dir)

        assert model is not None
        assert hasattr(model, "model")  # TrainedRandomForest has model attribute
        assert hasattr(model, "features")  # TrainedRandomForest has features
        assert hasattr(model, "importances")  # TrainedRandomForest has importances

    def test_caches_after_first_load(self, trained_models_dir):
        """Should cache model in memory after first load."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_rf_model = None

        # First load
        model1 = _load_rf_model(trained_models_dir)
        # Second load should return cached version
        model2 = _load_rf_model(trained_models_dir)

        assert model1 is model2  # Same object (cached)

    @patch("hackathon_analysis.prediction.pipeline._ensure_model_file_from_hf")
    def test_calls_ensure_before_loading(self, mock_ensure, trained_models_dir):
        """Should call _ensure_model_file_from_hf before loading."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_rf_model = None

        # Setup mock to return actual file
        rf_file = trained_models_dir / "rf_model.pkl"
        mock_ensure.return_value = rf_file

        model = _load_rf_model(trained_models_dir)

        # Verify ensure was called with correct parameters
        mock_ensure.assert_called_once_with(trained_models_dir, "rf_model.pkl")
        assert model is not None


class TestLocalFirstPattern:
    """Integration tests for the local-first with HF fallback pattern."""

    def test_local_models_used_when_available(self, trained_models_dir):
        """When models exist locally, they should be used (fast path)."""
        import hackathon_analysis.prediction.pipeline as pipeline_mod

        pipeline_mod._cached_corr_weights = None
        pipeline_mod._cached_rf_model = None

        # Both files should exist (created by train_and_save)
        assert (trained_models_dir / "correlation_weights.json").exists()
        assert (trained_models_dir / "rf_model.pkl").exists()

        # Loading should work without any HF calls
        corr_weights = _load_correlation_weights(trained_models_dir)
        rf_model = _load_rf_model(trained_models_dir)

        assert corr_weights is not None
        assert rf_model is not None

    @patch("hackathon_analysis.prediction.pipeline.hf_hub_download")
    @patch("hackathon_analysis.prediction.pipeline.get_hf_repo_from_env")
    def test_hf_download_only_on_first_missing_file(
        self, mock_get_hf, mock_download, tmp_path
    ):
        """HF should only be accessed when local file is missing."""
        from hackathon_analysis.data_extraction.dataset_resolver import HFRepo

        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Create one model file locally
        corr_file = models_dir / "correlation_weights.json"
        corr_file.write_text('{"feature": 0.5}')

        # Setup mock for rf_model.pkl download
        mock_get_hf.return_value = HFRepo(repo_id="test/repo", repo_type="dataset")

        def create_rf_file(*args, **kwargs):
            """Create file when mock is called (simulating download)."""
            rf_file = models_dir / "rf_model.pkl"
            rf_file.write_text("model_data")
            return str(rf_file)

        mock_download.side_effect = create_rf_file

        # First call - should use local file (no HF call)
        result1 = _ensure_model_file_from_hf(models_dir, "correlation_weights.json")
        assert result1.exists()
        mock_download.assert_not_called()

        # Second call - should download (file missing)
        result2 = _ensure_model_file_from_hf(models_dir, "rf_model.pkl")
        assert result2.exists()
        mock_download.assert_called_once()
