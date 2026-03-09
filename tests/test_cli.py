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
