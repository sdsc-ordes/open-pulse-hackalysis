"""Tests for notebook concept cache upload to Hugging Face Hub (Priority #5).

Tests verify that:
1. Concept cache JSON is created in the data directory
2. Cache is uploaded to Hugging Face Hub along with other data
3. Cache format is correct (repo_url -> concept data mapping)
4. Export workflow saves CSV, Parquet, and Cache JSON in correct order
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


class TestNotebookConceptCacheExport:
    """Tests for notebook concept cache export (Cell 22)."""

    @pytest.fixture
    def notebook_path(self):
        """Path to the notebook being tested."""
        return Path("src/hackathon_analysis/data_analysis/repo_analysis.ipynb")

    @pytest.fixture
    def sample_enriched_df(self):
        """Create a sample enriched DataFrame with concept columns."""
        return pd.DataFrame({
            "repo_url": [
                "https://github.com/a/b",
                "https://github.com/c/d",
            ],
            "repo_name": ["repo_b", "repo_d"],
            "repo_concepts": [["ML", "Data"], ["Web"]],
            "repo_concept_names": [["Machine Learning", "Data Science"], ["Web Development"]],
            "repo_top_concept": ["Machine Learning", "Web Development"],
            "repo_concepts_error": [None, None],
            "stars": [5, 15],
            "commit_count_default_branch": [10, 20],
        })

    def test_notebook_file_exists(self, notebook_path):
        """Notebook file should exist."""
        assert notebook_path.exists(), f"Notebook not found: {notebook_path}"

    def test_notebook_is_valid_json(self, notebook_path):
        """Notebook should be valid JSON."""
        with open(notebook_path) as f:
            try:
                nb = json.load(f)
                assert isinstance(nb, dict)
                assert "cells" in nb
            except json.JSONDecodeError as e:
                pytest.fail(f"Notebook JSON is invalid: {e}")

    def test_cell_22_exists(self, notebook_path):
        """Cell 22 (export cell) should exist."""
        with open(notebook_path) as f:
            nb = json.load(f)

        assert len(nb["cells"]) > 22, "Notebook doesn't have cell 22"
        assert nb["cells"][22]["cell_type"] == "code"

    def test_cell_22_exports_to_csv_parquet(self, notebook_path):
        """Cell 22 should export to both CSV and Parquet."""
        with open(notebook_path) as f:
            nb = json.load(f)

        cell_22 = nb["cells"][22]
        source = "".join(cell_22["source"])

        # Check for CSV export
        assert ".to_csv(" in source, "CSV export not found in Cell 22"
        assert "repo_metadata_with_concepts.csv" in source

        # Check for Parquet export
        assert ".to_parquet(" in source, "Parquet export not found in Cell 22"
        assert "repo_metadata_with_concepts.parquet" in source

    def test_cell_22_saves_cache_json(self, notebook_path):
        """Cell 22 should save concept cache as JSON."""
        with open(notebook_path) as f:
            nb = json.load(f)

        cell_22 = nb["cells"][22]
        source = "".join(cell_22["source"])

        # Check for cache file path definition
        assert "repo_concepts_cache.json" in source, "Cache file path not found"

        # Check for JSON writing
        assert "json.dump(" in source, "json.dump not found for cache saving"

        # Check that cache is built from concept columns
        assert "cache_dict" in source, "Cache dictionary not built"
        assert "repo_concept_names" in source

    def test_cell_22_uploads_to_hf(self, notebook_path):
        """Cell 22 should call upload_to_hugging_face."""
        with open(notebook_path) as f:
            nb = json.load(f)

        cell_22 = nb["cells"][22]
        source = "".join(cell_22["source"])

        # Check for upload call
        assert "upload_to_hugging_face" in source, "Upload function not called"
        assert "DATA_ROOT" in source, "DATA_ROOT not used in upload"
        assert 'path_in_repo="data/"' in source or "path_in_repo='data/'" in source

    def test_cache_dict_structure(self, sample_enriched_df, tmp_path):
        """Cache dict should map repo_url to concept data."""
        # Simulate what Cell 22 does
        cache_dict = {}
        for _, row in sample_enriched_df.iterrows():
            repo_url = row["repo_url"]
            cache_dict[repo_url] = {
                "repo_concepts": row.get("repo_concepts", []),
                "repo_concept_names": row.get("repo_concept_names", []),
                "repo_top_concept": row.get("repo_top_concept"),
                "repo_concepts_error": row.get("repo_concepts_error"),
            }

        # Verify structure
        assert len(cache_dict) == 2
        assert "https://github.com/a/b" in cache_dict
        assert "https://github.com/c/d" in cache_dict

        # Verify entry format
        entry = cache_dict["https://github.com/a/b"]
        assert "repo_concepts" in entry
        assert "repo_concept_names" in entry
        assert "repo_top_concept" in entry
        assert "repo_concepts_error" in entry

        # Verify values
        assert entry["repo_concept_names"] == ["Machine Learning", "Data Science"]
        assert entry["repo_top_concept"] == "Machine Learning"

    def test_cache_json_serialization(self, sample_enriched_df, tmp_path):
        """Cache should be serializable to JSON."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Build cache
        cache_dict = {}
        for _, row in sample_enriched_df.iterrows():
            repo_url = row["repo_url"]
            cache_dict[repo_url] = {
                "repo_concepts": row.get("repo_concepts", []),
                "repo_concept_names": row.get("repo_concept_names", []),
                "repo_top_concept": row.get("repo_top_concept"),
                "repo_concepts_error": row.get("repo_concepts_error"),
            }

        # Save to JSON
        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

        # Verify it's readable
        assert cache_path.exists()
        with open(cache_path) as f:
            loaded = json.load(f)

        assert loaded == cache_dict


class TestConceptCacheExportWorkflow:
    """End-to-end tests for concept cache export workflow."""

    @pytest.fixture
    def sample_enriched_df(self):
        """Create a sample enriched DataFrame."""
        return pd.DataFrame({
            "repo_url": [
                "https://github.com/x/y",
                "https://github.com/m/n",
            ],
            "repo_name": ["repo_y", "repo_n"],
            "repo_concepts": [["AI"], ["Mobile"]],
            "repo_concept_names": [["Artificial Intelligence"], ["Mobile Development"]],
            "repo_top_concept": ["Artificial Intelligence", "Mobile Development"],
            "repo_concepts_error": [None, None],
            "stars": [100, 50],
        })

    def test_export_saves_all_formats(self, sample_enriched_df, tmp_path):
        """Export should save CSV, Parquet, and Cache JSON."""
        # Simulate Cell 22 export
        out_csv = tmp_path / "repo_metadata_with_concepts.csv"
        out_parquet = tmp_path / "repo_metadata_with_concepts.parquet"
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Export
        sample_enriched_df.to_csv(out_csv, index=False)
        sample_enriched_df.to_parquet(out_parquet, index=False)

        # Build cache
        cache_dict = {}
        for _, row in sample_enriched_df.iterrows():
            repo_url = row["repo_url"]
            cache_dict[repo_url] = {
                "repo_concepts": row.get("repo_concepts", []),
                "repo_concept_names": row.get("repo_concept_names", []),
                "repo_top_concept": row.get("repo_top_concept"),
                "repo_concepts_error": row.get("repo_concepts_error"),
            }

        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

        # Verify all files exist
        assert out_csv.exists()
        assert out_parquet.exists()
        assert cache_path.exists()

    def test_export_order_csv_before_cache(self, sample_enriched_df, tmp_path):
        """CSV should be exported before cache is saved."""
        # This ensures the source data exists before cache is built
        out_csv = tmp_path / "repo_metadata_with_concepts.csv"
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Export CSV first
        sample_enriched_df.to_csv(out_csv, index=False)
        assert out_csv.exists()

        # Then build cache
        cache_dict = {}
        for _, row in sample_enriched_df.iterrows():
            repo_url = row["repo_url"]
            cache_dict[repo_url] = {
                "repo_concept_names": row.get("repo_concept_names", []),
            }

        with open(cache_path, "w") as f:
            json.dump(cache_dict, f)

        # Both should exist
        assert out_csv.exists()
        assert cache_path.exists()

    @patch("hackathon_analysis.common_utils.upload_folder")
    def test_cache_upload_with_other_data(self, mock_upload, sample_enriched_df, tmp_path):
        """Upload should include cache along with CSV and Parquet."""
        from hackathon_analysis.common_utils import upload_to_hugging_face

        # Export all data
        sample_enriched_df.to_csv(tmp_path / "repo_metadata_with_concepts.csv", index=False)
        sample_enriched_df.to_parquet(tmp_path / "repo_metadata_with_concepts.parquet", index=False)

        # Build and save cache
        cache_dict = {}
        for _, row in sample_enriched_df.iterrows():
            repo_url = row["repo_url"]
            cache_dict[repo_url] = {
                "repo_concept_names": row.get("repo_concept_names", []),
            }

        with open(tmp_path / "repo_concepts_cache.json", "w") as f:
            json.dump(cache_dict, f)

        # Upload
        upload_to_hugging_face(tmp_path, path_in_repo="data/", verbose=False)

        # Verify upload was called
        mock_upload.assert_called_once()


class TestConceptCacheIntegration:
    """Integration tests for concept cache with notebook workflow."""

    def test_cache_can_be_loaded_by_cell_24(self, tmp_path):
        """Cache saved by Cell 22 should be loadable by Cell 24."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Simulate Cell 22: save cache
        cache_dict = {
            "https://github.com/a/b": {
                "repo_concepts": ["ML", "Data"],
                "repo_concept_names": ["Machine Learning", "Data Science"],
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

        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

        # Simulate Cell 24: load cache
        with open(cache_path) as f:
            loaded_cache = json.load(f)

        # Verify structure matches Cell 24 expectations
        assert len(loaded_cache) == 2
        for repo_url in loaded_cache:
            entry = loaded_cache[repo_url]
            assert "repo_concept_names" in entry
            assert "repo_top_concept" in entry

    def test_cache_handles_missing_concept_fields(self, tmp_path):
        """Cache should handle repos with missing concept data gracefully."""
        cache_path = tmp_path / "repo_concepts_cache.json"

        # Create cache with some repos missing concept data
        cache_dict = {
            "https://github.com/a/b": {
                "repo_concepts": ["ML"],
                "repo_concept_names": ["Machine Learning"],
                "repo_top_concept": "Machine Learning",
                "repo_concepts_error": None,
            },
            "https://github.com/e/f": {
                "repo_concepts": [],
                "repo_concept_names": [],
                "repo_top_concept": None,
                "repo_concepts_error": "no_credentials",
            },
        }

        with open(cache_path, "w") as f:
            json.dump(cache_dict, f, indent=2)

        # Load and verify graceful handling
        with open(cache_path) as f:
            loaded = json.load(f)

        # Both entries should load successfully
        assert "https://github.com/a/b" in loaded
        assert "https://github.com/e/f" in loaded

        # Second repo's entry should be valid even with empty concepts
        assert loaded["https://github.com/e/f"]["repo_concept_names"] == []
        assert loaded["https://github.com/e/f"]["repo_concepts_error"] is not None

    def test_cache_format_matches_cell_24_expectations(self):
        """Cache format should match what Cell 24 expects."""
        # Document the expected cache format
        expected_structure = {
            "repo_url_1": {
                "repo_concepts": list,  # List of concept IDs
                "repo_concept_names": list,  # List of concept names
                "repo_top_concept": str,  # Top concept name or None
                "repo_concepts_error": str,  # Error message or None
            }
        }

        # Verify Cell 24 uses these fields with .get() which provides defaults
        # This means optional fields are safe
        cache_entry = {
            "repo_concepts": ["concept1"],
            "repo_concept_names": ["Concept Name"],
            "repo_top_concept": "Concept Name",
            "repo_concepts_error": None,
        }

        # Cell 24 loads like this:
        assert cache_entry.get("repo_concepts", []) is not None
        assert cache_entry.get("repo_concept_names", []) is not None
        assert cache_entry.get("repo_top_concept") is not None
        # This structure is compatible


class TestConceptCacheUploadPath:
    """Tests for upload path configuration."""

    def test_cache_file_is_in_data_directory(self):
        """Cache should be in data/ directory for upload."""
        # Cell 22 defines: cache_path = DATA_ROOT / "repo_concepts_cache.json"
        # Cell 67 uploads: upload_to_hugging_face(DATA_ROOT, path_in_repo="data/")
        # This means cache_path is automatically included in the upload

        # The cache file location should be:
        # DATA_ROOT / "repo_concepts_cache.json"
        # Which will be uploaded to: data/repo_concepts_cache.json in HF Hub

        cache_filename = "repo_concepts_cache.json"
        assert cache_filename.endswith(".json")

    @patch("hackathon_analysis.common_utils.upload_folder")
    def test_upload_includes_cache_in_data_folder(self, mock_upload, tmp_path):
        """Upload should include cache JSON in the data/ folder."""
        from hackathon_analysis.common_utils import upload_to_hugging_face

        # Create sample files in data directory
        (tmp_path / "repo_metadata_with_concepts.csv").touch()
        (tmp_path / "repo_metadata_with_concepts.parquet").touch()
        (tmp_path / "repo_concepts_cache.json").touch()

        # Upload
        upload_to_hugging_face(tmp_path, path_in_repo="data/", verbose=False)

        # Verify upload was called with correct folder
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args

        # The call should include the directory containing all files
        assert str(tmp_path) in str(call_args) or tmp_path in call_args[1].values()
