"""Tests for Pydantic validation in extraction models."""

import pytest
from pydantic import ValidationError

from hackathon_analysis.data_extraction.models import (
    GitHubRepoMetadata,
    LauzHackMetadata,
)


def test_github_repo_metadata_parses_iso_datetimes_and_ints():
    model = GitHubRepoMetadata(
        owner="octocat",
        repo="hello-world",
        has_tests=True,
        has_docs=False,
        has_ci=True,
        has_docker=False,
        has_notebooks=False,
        has_contributing=False,
        has_license_file=True,
        has_readme_file=True,
        created_at="2024-12-01T10:22:33Z",
        stars="123",
    )

    dumped = model.model_dump(mode="json")
    assert dumped["created_at"] == "2024-12-01T10:22:33Z"
    assert dumped["stars"] == 123


def test_github_repo_metadata_rejects_invalid_datetime():
    with pytest.raises(ValidationError):
        GitHubRepoMetadata(
            owner="octocat",
            repo="hello-world",
            has_tests=True,
            has_docs=False,
            has_ci=True,
            has_docker=False,
            has_notebooks=False,
            has_contributing=False,
            has_license_file=True,
            has_readme_file=True,
            created_at="not-a-date",
        )

def test_lauzhack_metadata_parses_date_bounds():
    model = LauzHackMetadata(
        source_url="https://lauzhack.com/2024",
        date_start="2024-11-30",
        date_end="2024-12-01",
        extracted_at="2024-10-01T08:00:00Z",
    )

    dumped = model.model_dump(mode="json")
    assert dumped["date_start"] == "2024-11-30"
    assert dumped["date_end"] == "2024-12-01"
    assert dumped["extracted_at"] == "2024-10-01T08:00:00Z"


def test_lauzhack_metadata_rejects_invalid_date():
    with pytest.raises(ValidationError):
        LauzHackMetadata(date_start="2024-13-01")


def test_models_validate_http_urls():
    gh = GitHubRepoMetadata(
        owner="octocat",
        repo="hello-world",
        has_tests=True,
        has_docs=False,
        has_ci=True,
        has_docker=False,
        has_notebooks=False,
        has_contributing=False,
        has_license_file=True,
        has_readme_file=True,
        url="https://github.com/octocat/hello-world",
        homepage_url="https://example.com",
    )
    assert gh.model_dump(mode="json")["url"] == "https://github.com/octocat/hello-world"

    with pytest.raises(ValidationError):
        GitHubRepoMetadata(
            owner="octocat",
            repo="hello-world",
            has_tests=True,
            has_docs=False,
            has_ci=True,
            has_docker=False,
            has_notebooks=False,
            has_contributing=False,
            has_license_file=True,
            has_readme_file=True,
            url="not-a-url",
        )
