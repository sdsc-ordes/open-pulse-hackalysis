"""Tests for the Hackalysis command line interface."""

import os

from hackathon_analysis import cli
from typer.testing import CliRunner

runner = CliRunner()


def test_extract_help():
    """Check if the 'hackalysis extract --help' command exits successfully."""
    result = runner.invoke(cli.app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "extract" in result.stdout.lower()
    assert "hackathon_provider" in result.stdout or "provider" in result.stdout.lower()


def test_main_help():
    """Check if the 'hackalysis --help' command exits successfully."""
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "extract" in result.stdout.lower()


def test_extract_missing_required_args():
    """Check that extract fails gracefully without required arguments."""
    result = runner.invoke(cli.app, ["extract"])
    # Should fail because required args are missing
    assert result.exit_code != 0


def test_extract_invalid_provider():
    """Check that extract fails with invalid provider."""
    result = runner.invoke(
        cli.app,
        [
            "extract",
            "-o", "/tmp/test",
            "-p", "invalid_provider"
        ]
    )
    assert result.exit_code != 0
    # Error message could be in stdout or stderr
    output = result.stdout.lower() + (result.stderr.lower()
                                      if hasattr(result, 'stderr') else "")
    EXPECTED_EXIT_CODE = 2
    assert "unknown" in output or "error" in output or result.exit_code == EXPECTED_EXIT_CODE


def test_get_github_token_loads_dotenv(monkeypatch):
    """Check that token lookup loads dotenv values."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def _fake_load_dotenv(override=False):
        assert override is False
        os.environ["GITHUB_TOKEN"] = "token-from-dotenv"
        return True

    monkeypatch.setattr(cli, "load_dotenv", _fake_load_dotenv)
    assert cli._get_github_token() == "token-from-dotenv"


def test_github_extract_account_help():
    """Check if the account extraction help command exits successfully."""
    result = runner.invoke(cli.app, ["github-extract-account", "--help"])
    assert result.exit_code == 0
    assert "account" in result.stdout.lower()


def test_github_extract_account_invokes_runner(monkeypatch, tmp_path):
    """Check that the account CLI command calls the account runner."""

    monkeypatch.setattr(cli, "_get_github_token", lambda: "fake-token")

    captured = {}
    uploaded = {}

    def _fake_upload_to_hugging_face(output_folder):
        uploaded["path"] = output_folder
        return True

    monkeypatch.setattr(cli, "upload_to_hugging_face", _fake_upload_to_hugging_face)

    def _fake_run_repo_metadata_from_account(**kwargs):
        captured.update(kwargs)
        out_dir = kwargs["hackathon_folder"]
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "github_account_github_repo_metadata.json"
        parquet_path = out_dir / "github_account_github_repo_metadata.parquet"
        project_json_path = out_dir / "github_account_github_project_metadata.json"
        project_parquet_path = out_dir / "github_account_github_project_metadata.parquet"
        json_path.write_text("{}", encoding="utf-8")
        parquet_path.write_text("", encoding="utf-8")
        project_json_path.write_text("[]", encoding="utf-8")
        project_parquet_path.write_text("", encoding="utf-8")
        return {
            "repos_found": 2,
            "outputs": {
                "json": str(json_path),
                "parquet": str(parquet_path),
                "project_json": str(project_json_path),
                "project_parquet": str(project_parquet_path),
            },
        }

    monkeypatch.setattr(cli, "run_repo_metadata_from_account", _fake_run_repo_metadata_from_account)

    result = runner.invoke(
        cli.app,
        [
            "github-extract-account",
            "-o", str(tmp_path),
            "-a", "openai",
        ],
    )

    assert result.exit_code == 0
    assert captured["account_name"] == "openai"
    assert captured["provider_prefix"] == "github_account"
    assert captured["hackathon_folder"] == tmp_path / "github-account-openai"
    assert uploaded["path"] == tmp_path / "github-account-openai"


def test_github_extract_shows_extract_first_message_when_projects_dataset_missing(monkeypatch, tmp_path):
    """Missing project parquet should produce a clear CLI instruction."""

    monkeypatch.setattr(cli, "_get_github_token", lambda: "fake-token")
    monkeypatch.setattr(cli, "resolve_lauzhack_years", lambda output_folder, years: [2025])
    monkeypatch.setattr(
        cli,
        "get_projects_parquet_path",
        lambda output_folder, provider, year=None, hackathon_name=None: (
            tmp_path / "lauzhack-2025" / "lauzhack_projects.parquet",
            "lauzhack-2025/lauzhack_projects.parquet",
        ),
    )

    def _missing_from_hf(output_folder, subpath):
        raise FileNotFoundError(subpath)

    monkeypatch.setattr(cli, "ensure_local_file_from_hf", _missing_from_hf)

    result = runner.invoke(
        cli.app,
        [
            "github-extract",
            "-o", str(tmp_path),
            "-p", "lauzhack",
            "-y", "2025",
            "--no-upload",
        ],
    )

    output = result.stdout.lower() + (result.stderr.lower() if hasattr(result, "stderr") else "")
    assert result.exit_code == 1
    assert "run the extract command first" in output
