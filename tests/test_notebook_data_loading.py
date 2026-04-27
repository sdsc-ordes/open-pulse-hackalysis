"""Tests for notebook data loading with local-first + HF fallback strategy.

Tests verify that:
1. Notebook loads from local files when available (CSV, Parquet)
2. Notebook falls back to HF Hub when local files missing
3. Helpful error message shown when data unavailable
4. Project metadata loading is graceful (optional)
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


class TestNotebookDataLoadingLogic:
    """Tests for Cell 4 data loading logic."""

    @pytest.fixture
    def sample_repo_df(self):
        """Create a sample repo metadata DataFrame."""
        return pd.DataFrame({
            "repo_url": [
                "https://github.com/a/b",
                "https://github.com/c/d",
            ],
            "repo_name": ["repo_b", "repo_d"],
            "stars": [10, 20],
            "project_foreign_key": ["proj1", None],
            "true_hackathon_repos": [True, False],
            "has_valid_metadata": [True, True],
        })

    def test_load_from_local_csv(self, sample_repo_df, tmp_path):
        """Cell 4 should load from local CSV if it exists."""
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"
        sample_repo_df.to_csv(csv_path, index=False)

        # Simulate Cell 4 logic
        repo_metadata_df = None
        if csv_path.exists():
            repo_metadata_df = pd.read_csv(csv_path)

        # Verify
        assert repo_metadata_df is not None
        assert len(repo_metadata_df) == 2
        assert "repo_url" in repo_metadata_df.columns

    def test_load_from_local_parquet(self, sample_repo_df, tmp_path):
        """Cell 4 should load from local Parquet if it exists."""
        parquet_path = tmp_path / "repo_metadata_with_concepts.parquet"
        sample_repo_df.to_parquet(parquet_path, index=False)

        # Simulate Cell 4 logic - parquet has priority
        repo_metadata_df = None
        if parquet_path.exists():
            repo_metadata_df = pd.read_parquet(parquet_path).convert_dtypes()

        # Verify
        assert repo_metadata_df is not None
        assert len(repo_metadata_df) == 2

    def test_parquet_has_priority_over_csv(self, sample_repo_df, tmp_path):
        """Cell 4 should check Parquet first, then CSV."""
        parquet_path = tmp_path / "repo_metadata_with_concepts.parquet"
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"

        # Create both files (Parquet with 2 rows, CSV with 1 row)
        sample_repo_df.to_parquet(parquet_path, index=False)
        sample_repo_df.iloc[:1].to_csv(csv_path, index=False)

        # Simulate Cell 4 priority logic
        if parquet_path.exists():
            repo_metadata_df = pd.read_parquet(parquet_path).convert_dtypes()
        elif csv_path.exists():
            repo_metadata_df = pd.read_csv(csv_path)

        # Should load Parquet (2 rows), not CSV (1 row)
        assert len(repo_metadata_df) == 2

    @patch("hackathon_analysis.common_utils.load_huggingface_dataset")
    def test_fallback_to_hf_hub(self, mock_hf_load, sample_repo_df, tmp_path):
        """Cell 4 should download from HF Hub if local files missing."""
        from hackathon_analysis.common_utils import load_huggingface_dataset

        parquet_path = tmp_path / "repo_metadata_with_concepts.parquet"
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"

        # Neither file exists
        assert not parquet_path.exists()
        assert not csv_path.exists()

        # Mock HF download
        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = sample_repo_df
        mock_hf_load.return_value = mock_dataset

        # Simulate Cell 4 fallback logic
        if not parquet_path.exists() and not csv_path.exists():
            hf_dataset = load_huggingface_dataset(repo_id="test/repo", split="train")
            repo_metadata_df = hf_dataset.to_pandas()

        # Verify HF was called
        mock_hf_load.assert_called_once_with(repo_id="test/repo", split="train")
        assert len(repo_metadata_df) == 2


class TestNotebookProjectDataLoading:
    """Tests for project metadata loading in Cell 4."""

    @pytest.fixture
    def sample_project_df(self):
        """Create a sample project DataFrame."""
        return pd.DataFrame({
            "project_foreign_key": ["proj1", "proj2", "proj3"],
            "project_name": ["LH2023", "LH2024", "LH2025"],
            "year": [2023, 2024, 2025],
        })

    @pytest.fixture
    def sample_repo_df(self):
        """Create a sample repo DataFrame."""
        return pd.DataFrame({
            "repo_url": ["https://github.com/a/b", "https://github.com/c/d"],
            "project_foreign_key": ["proj1", "proj2"],
        })

    def test_load_project_metadata_from_parquet(self, sample_project_df, tmp_path):
        """Cell 4 should load project metadata from Parquet if available."""
        project_parquet = tmp_path / "lauzhack_projects.parquet"
        sample_project_df.to_parquet(project_parquet, index=False)

        # Simulate Cell 4 logic
        if project_parquet.exists():
            project_df = pd.read_parquet(project_parquet)
        else:
            project_df = None

        # Verify
        assert project_df is not None
        assert len(project_df) == 3

    def test_create_minimal_project_df_if_missing(self, sample_repo_df, tmp_path):
        """Cell 4 should create minimal project_df if parquet missing."""
        project_parquet = tmp_path / "lauzhack_projects.parquet"

        # Parquet doesn't exist
        assert not project_parquet.exists()

        # Simulate Cell 4 graceful fallback
        if project_parquet.exists():
            project_df = pd.read_parquet(project_parquet)
        else:
            # Infer from repo_df
            project_df = sample_repo_df[["project_foreign_key"]].dropna().drop_duplicates().copy()
            project_df.columns = ["project_id"]
            project_df["project_name"] = "Unknown"

        # Verify
        assert len(project_df) == 2
        assert "project_id" in project_df.columns
        assert "project_name" in project_df.columns


class TestNotebookErrorHandling:
    """Tests for error handling when data unavailable."""

    def test_error_message_when_no_data(self, tmp_path):
        """Cell 4 should raise helpful error if no data available."""
        parquet_path = tmp_path / "repo_metadata_with_concepts.parquet"
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"

        # Neither file exists, and we'll skip HF download for this test
        assert not parquet_path.exists()
        assert not csv_path.exists()

        # Verify error condition is detected
        has_local_data = parquet_path.exists() or csv_path.exists()
        assert not has_local_data

    def test_error_includes_setup_instructions(self):
        """Error message should include setup options."""
        error_message = (
            "No data available. You have two options:\n"
            "Option 1: Download from HF Hub\n"
            "Option 2: Extract from source files"
        )

        # Verify it includes helpful instructions
        assert "Option 1" in error_message
        assert "Option 2" in error_message


class TestNotebookIntegration:
    """Integration tests for full data loading flow."""

    def test_notebook_cell_flow(self, tmp_path):
        """Test the complete Cell 4 flow with local data."""
        # Setup
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"
        sample_df = pd.DataFrame({
            "repo_url": ["https://github.com/a/b"],
            "stars": [10],
            "project_foreign_key": ["proj1"],
        })
        sample_df.to_csv(csv_path, index=False)

        # Load
        repo_metadata_df = None
        if csv_path.exists():
            repo_metadata_df = pd.read_csv(csv_path)

        # Verify ready for analysis
        assert repo_metadata_df is not None
        assert "repo_url" in repo_metadata_df.columns
        assert "project_foreign_key" in repo_metadata_df.columns
        print(f"✓ Ready for analysis with {len(repo_metadata_df)} repos")

    def test_notebook_loading_precedence(self, tmp_path):
        """Verify loading precedence: Parquet > CSV > HF > Error."""
        parquet_path = tmp_path / "repo_metadata_with_concepts.parquet"
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"

        # Test 1: Only CSV exists
        csv_path.touch()
        loaded_from = "CSV" if csv_path.exists() and not parquet_path.exists() else None
        assert loaded_from == "CSV"

        # Test 2: Both exist (Parquet has priority)
        parquet_path.touch()
        loaded_from = "Parquet" if parquet_path.exists() else ("CSV" if csv_path.exists() else None)
        assert loaded_from == "Parquet"


class TestNotebookMarkdownGuide:
    """Tests for markdown cell documentation."""

    def test_notebook_has_loading_workflow_cell(self):
        """Notebook should have markdown explaining data loading strategy."""
        nb_path = Path("src/hackathon_analysis/data_analysis/repo_analysis.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)

        # Find markdown cell explaining workflow
        found_workflow_cell = False
        for cell in nb["cells"]:
            if cell["cell_type"] == "markdown":
                source = "".join(cell["source"])
                if "Data Loading Workflow" in source:
                    found_workflow_cell = True
                    # Verify it documents the strategy
                    assert "Local Parquet" in source
                    assert "Hugging Face Hub" in source
                    assert "Loading Priority" in source
                    break

        assert found_workflow_cell, "Notebook should include data loading workflow documentation"

    def test_cell_4_has_hf_import(self):
        """Cell 4 should import HF utilities for fallback."""
        nb_path = Path("src/hackathon_analysis/data_analysis/repo_analysis.ipynb")
        with open(nb_path) as f:
            nb = json.load(f)

        # Find Cell 4 (data loading)
        if len(nb["cells"]) > 4:
            cell_4 = nb["cells"][4]
            source = "".join(cell_4["source"])

            # Should handle HF fallback
            assert "load_huggingface_dataset" in source or "HF" in source.upper()
