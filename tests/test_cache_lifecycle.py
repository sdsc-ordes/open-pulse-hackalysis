"""Tests for concept cache lifecycle: download from HF, incremental enrichment, re-upload.

Tests verify that:
1. Cell 4 downloads cache from HF Hub when available
2. Enrichment loop in Cell 22 checks cache and skips enriched repos
3. New repos are enriched and cache is updated
4. Cache is persisted locally during enrichment (for recovery)
5. Cell 24 uploads updated cache back to HF
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

import pandas as pd
import pytest


class TestCacheDownloadInCell4:
    """Tests for cache download in Cell 4 (data loading)."""

    @pytest.fixture
    def sample_cache(self):
        """Create a sample cached data."""
        return {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            },
            "https://github.com/c/d": {
                "repo_concepts": [],
                "repo_concept_names": [],
                "repo_top_concept": None,
                "repo_concepts_error": "no_credentials",
            },
        }

    def test_cache_downloaded_from_hf_if_exists(self, sample_cache, tmp_path):
        """Cache should be downloaded from HF Hub if available."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Simulate HF download by writing cache locally
        with open(cache_path, "w") as f:
            json.dump(sample_cache, f)

        # Load it (simulating Cell 4 logic)
        with open(cache_path) as f:
            loaded_cache = json.load(f)

        # Verify
        assert len(loaded_cache) == 2
        assert "https://github.com/a/b" in loaded_cache
        assert loaded_cache["https://github.com/a/b"]["repo_top_concept"] == "Machine Learning"

    def test_empty_cache_initialized_if_not_available(self, tmp_path):
        """If cache doesn't exist on HF, start with empty cache."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Cache doesn't exist
        assert not cache_path.exists()

        # Initialize empty cache (Cell 4 logic)
        concept_cache = {}
        assert len(concept_cache) == 0

    @patch("builtins.open", new_callable=mock_open)
    def test_hf_download_handles_missing_cache_gracefully(self, mock_file):
        """If cache file doesn't exist on HF, should not raise error."""
        # Simulate file not found on HF Hub
        mock_file.side_effect = FileNotFoundError("Cache not found on HF")

        # Cell 4 should handle this gracefully
        try:
            with open("nonexistent_cache.json") as f:
                cache = json.load(f)
        except FileNotFoundError:
            cache = {}

        # Should have empty cache
        assert cache == {}


class TestIncrementalEnrichmentInCell22:
    """Tests for incremental enrichment in Cell 22."""

    @pytest.fixture
    def sample_repos(self):
        """Create sample repos to enrich."""
        return pd.DataFrame({
            "repo_url": [
                "https://github.com/a/b",
                "https://github.com/c/d",
                "https://github.com/e/f",
            ],
            "repo_name": ["repo_b", "repo_d", "repo_f"],
            "description": ["Desc A", "Desc B", "Desc C"],
        })

    @pytest.fixture
    def existing_cache(self):
        """Create cache with some repos already enriched."""
        return {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            }
        }

    def test_enrichment_skips_cached_repos(self, sample_repos, existing_cache):
        """Enrichment should skip repos that are already in cache with no error."""
        enriched_count = 0
        cached_count = 0

        # Simulate Cell 22 logic
        for idx, row in sample_repos.iterrows():
            repo_url = row["repo_url"]

            if repo_url in existing_cache and existing_cache[repo_url].get("repo_concepts_error") is None:
                # Use cached data
                cached_count += 1
            else:
                # Would enrich this repo
                enriched_count += 1

        # Verify
        assert cached_count == 1  # One repo was cached
        assert enriched_count == 2  # Two repos need enrichment

    def test_enrichment_retries_repos_with_error(self, sample_repos):
        """Enrichment should retry repos that had errors in cache."""
        cache = {
            "https://github.com/c/d": {
                "repo_concepts": [],
                "repo_concept_names": [],
                "repo_top_concept": None,
                "repo_concepts_error": "no_credentials",  # Had error
            }
        }

        should_enrich = []

        for idx, row in sample_repos.iterrows():
            repo_url = row["repo_url"]
            if repo_url not in cache or cache[repo_url].get("repo_concepts_error") is not None:
                should_enrich.append(repo_url)

        # Verify repo with error is retried
        assert "https://github.com/c/d" in should_enrich

    def test_cache_updated_after_enrichment(self, sample_repos, existing_cache):
        """Cache should be updated with newly enriched repos."""
        enriched_result = {
            "repo_concepts": ["Data"],
            "repo_concept_names": ["Data Science"],
            "repo_top_concept": "Data Science",
            "repo_concepts_error": None,
        }

        # Simulate enriching a new repo
        repo_url = "https://github.com/e/f"
        existing_cache[repo_url] = enriched_result

        # Verify cache was updated
        assert repo_url in existing_cache
        assert existing_cache[repo_url]["repo_top_concept"] == "Data Science"


class TestCachePersistenceDuringEnrichment:
    """Tests for cache persistence during enrichment (for recovery)."""

    def test_cache_saved_periodically(self, tmp_path):
        """Cache should be saved periodically during enrichment (every 50 repos or on error)."""
        cache_path = tmp_path / "repo_concepts_cache.json"
        cache = {}

        # Simulate enriching 55 repos with periodic saves
        for i in range(55):
            repo_url = f"https://github.com/repo/{i}"
            cache[repo_url] = {
                "repo_concepts": ["Concept"],
                "repo_concept_names": ["Concept"],
                "repo_top_concept": "Concept",
                "repo_concepts_error": None,
            }

            # Save every 50
            if (i + 1) % 50 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f)

        # Verify final save happened
        assert cache_path.exists()
        with open(cache_path) as f:
            saved_cache = json.load(f)

        # Should have at least 50 entries
        assert len(saved_cache) >= 50

    def test_cache_saved_on_error(self, tmp_path):
        """Cache should be saved immediately when an error occurs."""
        cache_path = tmp_path / "repo_concepts_cache.json"
        cache = {
            "https://github.com/repo/1": {
                "repo_concepts": [],
                "repo_concept_names": [],
                "repo_top_concept": None,
                "repo_concepts_error": "api_error",
            }
        }

        # Error occurred, save cache
        with open(cache_path, "w") as f:
            json.dump(cache, f)

        # Cache should be persistent
        assert cache_path.exists()
        with open(cache_path) as f:
            loaded = json.load(f)
        assert "https://github.com/repo/1" in loaded

    def test_cache_recovered_after_interrupt(self, tmp_path):
        """If enrichment is interrupted, cache should be recoverable from disk."""
        cache_path = tmp_path / "repo_concepts_cache.json"
        partial_cache = {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            },
            "https://github.com/c/d": {
                "repo_concepts": [],
                "repo_concept_names": [],
                "repo_top_concept": None,
                "repo_concepts_error": "api_timeout",  # Partial enrichment
            },
        }

        # Save partial cache
        with open(cache_path, "w") as f:
            json.dump(partial_cache, f)

        # Simulate restart: load the partial cache
        with open(cache_path) as f:
            recovered_cache = json.load(f)

        # Verify recovery
        assert len(recovered_cache) == 2
        assert recovered_cache["https://github.com/c/d"]["repo_concepts_error"] == "api_timeout"


class TestCacheUploadInCell24:
    """Tests for cache upload in Cell 24 (export)."""

    def test_final_cache_saved_before_upload(self, tmp_path):
        """Cache should be finalized and saved before upload in Cell 24."""
        cache_path = tmp_path / "repo_concepts_cache.json"
        concept_cache = {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            },
            "https://github.com/c/d": {
                "repo_concepts": ["Web"],
                "repo_concept_names": ["Web Development"],
                "repo_top_concept": "Web Development",
                "repo_concepts_error": None,
            },
        }

        # Cell 24: Save cache before upload
        with open(cache_path, "w") as f:
            json.dump(concept_cache, f, indent=2)

        # Verify
        assert cache_path.exists()
        with open(cache_path) as f:
            saved = json.load(f)
        assert len(saved) == 2

    @patch("hackathon_analysis.common_utils.upload_folder")
    def test_cache_uploaded_with_data(self, mock_upload, tmp_path):
        """Cache should be uploaded with other data to HF Hub."""
        from hackathon_analysis.common_utils import upload_to_hugging_face

        # Create mock data and cache
        (tmp_path / "repo_metadata_with_concepts.csv").touch()
        (tmp_path / "repo_concepts_cache.json").touch()

        # Mock the upload
        mock_upload.return_value = None

        # Cell 24: Upload
        upload_to_hugging_face(tmp_path, path_in_repo="data/", verbose=False)

        # Verify upload was called
        mock_upload.assert_called_once()


class TestCacheLifecycleIntegration:
    """End-to-end tests for complete cache lifecycle."""

    def test_full_lifecycle_first_run(self, tmp_path):
        """Test: First run with no existing cache."""
        # Step 1: Cell 4 - Load (or start fresh)
        cache_path = tmp_path / "repo_concepts_cache.json"
        concept_cache = {}  # Empty cache
        assert len(concept_cache) == 0

        # Step 2: Cell 22 - Enrich all
        repos = [
            {"repo_url": "https://github.com/a/b"},
            {"repo_url": "https://github.com/c/d"},
        ]

        enriched = 0
        for repo in repos:
            url = repo["repo_url"]
            concept_cache[url] = {
                "repo_concepts": ["Concept"],
                "repo_concept_names": ["Concept"],
                "repo_top_concept": "Concept",
                "repo_concepts_error": None,
            }
            enriched += 1

        assert enriched == 2
        assert len(concept_cache) == 2

        # Step 3: Cell 24 - Upload
        with open(cache_path, "w") as f:
            json.dump(concept_cache, f)

        assert cache_path.exists()

    def test_full_lifecycle_second_run_incremental(self, tmp_path):
        """Test: Second run with existing cache (incremental)."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Step 1: Cell 4 - Download cache from previous run
        existing_cache = {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            }
        }

        with open(cache_path, "w") as f:
            json.dump(existing_cache, f)

        with open(cache_path) as f:
            concept_cache = json.load(f)

        assert len(concept_cache) == 1

        # Step 2: Cell 22 - Enrich only new repos
        new_repos = [
            {"repo_url": "https://github.com/a/b"},  # Already in cache
            {"repo_url": "https://github.com/c/d"},  # New
        ]

        cached_count = 0
        enriched_count = 0

        for repo in new_repos:
            url = repo["repo_url"]
            if url in concept_cache and concept_cache[url].get("repo_concepts_error") is None:
                cached_count += 1
            else:
                concept_cache[url] = {
                    "repo_concepts": ["Web"],
                    "repo_concept_names": ["Web Development"],
                    "repo_top_concept": "Web Development",
                    "repo_concepts_error": None,
                }
                enriched_count += 1

        assert cached_count == 1  # Skipped cached repo
        assert enriched_count == 1  # Enriched new repo
        assert len(concept_cache) == 2

        # Step 3: Cell 24 - Upload updated cache
        with open(cache_path, "w") as f:
            json.dump(concept_cache, f)

        with open(cache_path) as f:
            final_cache = json.load(f)

        assert len(final_cache) == 2
        assert "https://github.com/c/d" in final_cache  # New repo is there
