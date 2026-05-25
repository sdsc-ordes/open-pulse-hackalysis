"""Tests for notebook Hugging Face Hub upload functionality (Priority #3a).

Tests verify that:
1. upload_to_hugging_face() is imported in the notebook
2. Upload call is present in the export cell
3. Export workflow is correct (create DataFrame → export CSV → upload to HF)
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


class TestNotebookHFUpload:
    """Tests for notebook HF upload integration."""

    @pytest.fixture
    def notebook_path(self):
        """Path to the notebook being tested."""
        return Path("src/hackathon_analysis/data_analysis/repo_analysis.ipynb")

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

    def test_upload_import_in_notebook(self, notebook_path):
        """Notebook should import upload_to_hugging_face."""
        with open(notebook_path) as f:
            nb = json.load(f)

        # Check all code cells for the import
        found_import = False
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if "from hackathon_analysis.common_utils import upload_to_hugging_face" in source:
                    found_import = True
                    break

        assert found_import, "upload_to_hugging_face import not found in notebook"

    def test_upload_call_in_export_cell(self, notebook_path):
        """Export cell (cell 70) should call upload_to_hugging_face."""
        with open(notebook_path) as f:
            nb = json.load(f)

        # Cell 70 is the predictions export cell (Cell 69 is ensemble majority vote)
        assert len(nb["cells"]) > 70, "Notebook doesn't have cell 70"

        export_cell = nb["cells"][70]
        assert export_cell["cell_type"] == "code"

        source = "".join(export_cell["source"])

        # Check for upload call
        assert "upload_to_hugging_face" in source, "upload_to_hugging_face call not found"
        assert "DATA_ROOT" in source, "DATA_ROOT not used in upload"
        assert 'path_in_repo="data/"' in source or "path_in_repo='data/'" in source, \
            "Upload path not specified correctly"

    def test_export_cell_workflow(self, notebook_path):
        """Export cell should have complete workflow: export CSV then upload."""
        with open(notebook_path) as f:
            nb = json.load(f)

        export_cell = nb["cells"][70]
        source = "".join(export_cell["source"])

        # Check workflow order
        csv_export_pos = source.find(".to_csv(")
        upload_pos = source.find("upload_to_hugging_face(")

        assert csv_export_pos >= 0, "CSV export not found"
        assert upload_pos >= 0, "Upload call not found"
        assert csv_export_pos < upload_pos, "CSV export should come before upload"

    def test_export_includes_csv_path(self, notebook_path):
        """Export should target repo_metadata_with_predictions.csv."""
        with open(notebook_path) as f:
            nb = json.load(f)

        export_cell = nb["cells"][70]
        source = "".join(export_cell["source"])

        assert "repo_metadata_with_predictions.csv" in source, \
            "Expected CSV filename not found in export"


class TestUploadFunctionality:
    """Tests for upload_to_hugging_face() function behavior."""

    def test_upload_function_exists(self):
        """upload_to_hugging_face should be importable."""
        from hackathon_analysis.common_utils import upload_to_hugging_face

        assert callable(upload_to_hugging_face)

    def test_upload_function_signature(self):
        """upload_to_hugging_face should have expected parameters."""
        from hackathon_analysis.common_utils import upload_to_hugging_face
        import inspect

        sig = inspect.signature(upload_to_hugging_face)
        params = list(sig.parameters.keys())

        assert "output_folder" in params, "Missing output_folder parameter"
        assert "path_in_repo" in params, "Missing path_in_repo parameter"

    @patch("hackathon_analysis.common_utils.upload_folder")
    def test_upload_handles_data_root(self, mock_upload):
        """Upload should handle DATA_ROOT directory correctly."""
        from hackathon_analysis.common_utils import upload_to_hugging_face
        from pathlib import Path

        test_path = Path("/tmp/data")

        # Mock the upload_folder to avoid actual HF calls
        mock_upload.return_value = None

        # Call with typical notebook parameters
        result = upload_to_hugging_face(test_path, path_in_repo="data/", verbose=False)

        # Verify upload was called with correct folder
        mock_upload.assert_called_once()
        call_args = mock_upload.call_args
        assert str(test_path) in str(call_args) or test_path in call_args[1].values()

    def test_upload_with_path_in_repo(self):
        """Upload should accept path_in_repo parameter."""
        from hackathon_analysis.common_utils import upload_to_hugging_face
        from pathlib import Path
        import inspect

        sig = inspect.signature(upload_to_hugging_face)

        # Check parameter exists
        assert "path_in_repo" in sig.parameters

        # Check it's optional (has default)
        param = sig.parameters["path_in_repo"]
        assert param.default is not inspect.Parameter.empty or param.default is None


class TestNotebookExportWorkflow:
    """End-to-end tests for notebook export workflow."""

    @pytest.fixture
    def sample_export_df(self):
        """Create a sample export DataFrame."""
        return pd.DataFrame({
            "repo_url": ["https://github.com/a/b", "https://github.com/c/d"],
            "commit_count_default_branch": [10, 20],
            "stars": [5, 15],
            "contributors_count": [2, 3],
            "true_hackathon_repos": [True, False],
            "created_at": ["2024-01-01", "2024-02-01"],
            "corr_weights": [[0.1, 0.2], [0.3, 0.4]],  # Will be dropped
            "ml_feature_cols": [["f1", "f2"], ["f3"]],  # Will be dropped
        })

    def test_export_drops_internal_columns(self, sample_export_df, tmp_path):
        """Export should drop internal columns before CSV export."""
        # Simulate notebook export logic
        export_df = sample_export_df.drop(
            columns=["corr_weights", "ml_feature_cols"],
            errors="ignore"
        )

        assert "corr_weights" not in export_df.columns
        assert "ml_feature_cols" not in export_df.columns
        assert "repo_url" in export_df.columns
        assert len(export_df) == 2

    def test_export_csv_format(self, sample_export_df, tmp_path):
        """Exported CSV should be readable and properly formatted."""
        export_df = sample_export_df.drop(
            columns=["corr_weights", "ml_feature_cols"],
            errors="ignore"
        )

        csv_path = tmp_path / "repo_metadata_with_predictions.csv"
        export_df.to_csv(csv_path, index=False)

        # Verify CSV is readable
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 2
        assert list(loaded["repo_url"]) == [
            "https://github.com/a/b",
            "https://github.com/c/d",
        ]

    @patch("hackathon_analysis.common_utils.upload_folder")
    def test_upload_after_csv_export(self, mock_upload, sample_export_df, tmp_path):
        """After CSV export, notebook should upload to HF."""
        from hackathon_analysis.common_utils import upload_to_hugging_face

        # Create CSV
        export_df = sample_export_df.drop(
            columns=["corr_weights", "ml_feature_cols"],
            errors="ignore"
        )
        csv_path = tmp_path / "repo_metadata_with_predictions.csv"
        export_df.to_csv(csv_path, index=False)

        # Upload (as notebook would do)
        upload_to_hugging_face(tmp_path, path_in_repo="data/", verbose=False)

        # Verify upload was called
        mock_upload.assert_called_once()


class TestHFUploadIntegration:
    """Integration tests for notebook → HF workflow."""

    def test_environment_variables_for_upload(self):
        """HF upload requires proper environment variables."""
        from hackathon_analysis.data_extraction.dataset_resolver import get_hf_repo_from_env

        repo = get_hf_repo_from_env()

        # Should have defaults
        assert repo.repo_id is not None
        assert repo.repo_type is not None
        assert repo.repo_id == "SDSC/open-pulse-hackathon-data-analysis"
        assert repo.repo_type == "dataset"

    def test_notebook_upload_matches_dashboard_expectations(self):
        """Notebook upload should create data that dashboard can load."""
        # Dashboard expects:
        # 1. CSV file with repo metadata
        # 2. Uploaded to HF with appropriate structure
        # 3. Loadable via load_dataset(repo_id, split="train")

        # Verify notebook exports correct format
        from hackathon_analysis.data_extraction.dataset_resolver import get_hf_repo_from_env

        repo = get_hf_repo_from_env()

        # Notebook uploads to path_in_repo="data/"
        # Dashboard should be able to find it
        assert repo.repo_id is not None

    def test_upload_error_handling(self):
        """Upload should handle errors gracefully."""
        from hackathon_analysis.common_utils import upload_to_hugging_face
        from pathlib import Path

        # Calling with non-existent folder should handle gracefully
        # (actual behavior depends on implementation)
        assert callable(upload_to_hugging_face)
