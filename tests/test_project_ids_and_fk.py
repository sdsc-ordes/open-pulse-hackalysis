"""Tests for LauzHack project hard ids and GitHub metadata foreign keys."""

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from hackathon_analysis.data_extraction.github_extractor import (
    run_repo_metadata_from_projects_parquet,
)
from hackathon_analysis.data_extraction.lauzhack_extractor import process_project_data


def test_process_project_data_assigns_deterministic_project_hard_id():
    raw_projects = [
        {
            "id": 1,
            "title": "AI Helper",
            "description": "Does useful stuff",
            "url": "https://github.com/team/ai-helper",
            "team": ["Alice", "Bob"],
        }
    ]

    first = process_project_data(raw_projects)
    second = process_project_data(raw_projects)

    assert first[0]["project_hard_id"].startswith("lhp_")
    assert first[0]["project_hard_id"] == second[0]["project_hard_id"]


def test_github_repo_metadata_includes_project_foreign_key(tmp_path: Path):
    projects_df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "P1",
                "project_hard_id": "lhp_aaa111",
                "url": "https://github.com/org/repo-one",
            },
            {
                "id": 2,
                "title": "P2",
                "project_hard_id": "lhp_ccc333",
                "url": "https://github.com/org/repo-two",
            },
        ]
    )
    projects_parquet = tmp_path / "projects.parquet"
    projects_df.to_parquet(projects_parquet, index=False)

    def _fake_fetch_repo_metadata(urls, **kwargs):  # noqa: ANN001
        return {u: {"owner": "org", "repo": u.rsplit("/", 1)[-1]} for u in urls}

    with patch(
        "hackathon_analysis.data_extraction.github_extractor.fetch_repo_metadata",
        side_effect=_fake_fetch_repo_metadata,
    ):
        summary = run_repo_metadata_from_projects_parquet(
            projects_parquet=projects_parquet,
            hackathon_folder=tmp_path,
            provider_prefix="lauzhack",
            write_project_level_output=False,
        )

    repo_json = Path(summary["outputs"]["json"])
    data = json.loads(repo_json.read_text(encoding="utf-8"))

    repo_one = data["https://github.com/org/repo-one"]
    repo_two = data["https://github.com/org/repo-two"]

    assert repo_one["project_foreign_key"] == "lhp_aaa111"
    assert repo_two["project_foreign_key"] == "lhp_ccc333"


def test_github_repo_metadata_includes_foreign_key_on_error_rows(tmp_path: Path):
    projects_df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "P1",
                "project_hard_id": "lhp_err111",
                "url": "https://github.com/org/repo-error",
            }
        ]
    )
    projects_parquet = tmp_path / "projects.parquet"
    projects_df.to_parquet(projects_parquet, index=False)

    def _fake_fetch_repo_metadata(urls, **kwargs):  # noqa: ANN001
        return {u: {"error": "boom", "owner": "org", "repo": "repo-error"} for u in urls}

    with patch(
        "hackathon_analysis.data_extraction.github_extractor.fetch_repo_metadata",
        side_effect=_fake_fetch_repo_metadata,
    ):
        summary = run_repo_metadata_from_projects_parquet(
            projects_parquet=projects_parquet,
            hackathon_folder=tmp_path,
            provider_prefix="lauzhack",
            write_project_level_output=False,
        )

    repo_json = Path(summary["outputs"]["json"])
    data = json.loads(repo_json.read_text(encoding="utf-8"))
    err_repo = data["https://github.com/org/repo-error"]
    assert err_repo["error"] == "boom"
    assert err_repo["project_foreign_key"] == "lhp_err111"


def test_github_repo_metadata_raises_if_repo_is_linked_to_multiple_projects(tmp_path: Path):
    projects_df = pd.DataFrame(
        [
            {
                "id": 1,
                "title": "P1",
                "project_hard_id": "lhp_aaa111",
                "url": "https://github.com/org/repo-one",
            },
            {
                "id": 2,
                "title": "P2",
                "project_hard_id": "lhp_bbb222",
                "url": "https://github.com/org/repo-one",
            },
        ]
    )
    projects_parquet = tmp_path / "projects.parquet"
    projects_df.to_parquet(projects_parquet, index=False)

    def _fake_fetch_repo_metadata(urls, **kwargs):  # noqa: ANN001
        return {u: {"owner": "org", "repo": u.rsplit("/", 1)[-1]} for u in urls}

    with patch(
        "hackathon_analysis.data_extraction.github_extractor.fetch_repo_metadata",
        side_effect=_fake_fetch_repo_metadata,
    ):
        try:
            run_repo_metadata_from_projects_parquet(
                projects_parquet=projects_parquet,
                hackathon_folder=tmp_path,
                provider_prefix="lauzhack",
                write_project_level_output=False,
            )
        except ValueError as exc:
            assert "linked to multiple projects" in str(exc)
        else:
            raise AssertionError("Expected ValueError for duplicate repo-to-project mapping")
