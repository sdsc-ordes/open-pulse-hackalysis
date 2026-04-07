"""Tests for direct GitHub account metadata extraction."""

import json
from pathlib import Path

from hackathon_analysis.data_extraction import github_extractor as ge


class _FakeClient:
    def __init__(self, account_type="Organization"):
        self.account_type = account_type

    def rest_get(self, path, params=None):  # noqa: ANN001
        if path == "/users/openai":
            return {"type": self.account_type}
        raise AssertionError(f"Unexpected path: {path}")


def test_fetch_account_repos_auto_org(monkeypatch):
    """Auto-detect org accounts and return canonical repo URLs."""

    monkeypatch.setattr(
        ge,
        "fetch_org_repos",
        lambda client, org: [(org, "repo-one"), (org, "repo-two")],
    )

    urls = ge.fetch_account_repos(_FakeClient(), "openai", account_type="auto")

    assert urls == [
        "https://github.com/openai/repo-one",
        "https://github.com/openai/repo-two",
    ]


def test_fetch_account_repos_explicit_user(monkeypatch):
    """Explicit user scans should use the user repo listing helper."""

    monkeypatch.setattr(
        ge,
        "fetch_user_repos",
        lambda client, user: [(user, "repo-a")],
    )

    urls = ge.fetch_account_repos(_FakeClient(), "octocat", account_type="user")

    assert urls == ["https://github.com/octocat/repo-a"]


def test_run_repo_metadata_from_account_writes_outputs(tmp_path: Path, monkeypatch):
    """Account-based orchestration should write repo-level outputs."""

    monkeypatch.setattr(
        ge,
        "fetch_account_repos",
        lambda client, account_name, account_type="auto": [
            "https://github.com/openai/repo-one"
        ],
    )
    monkeypatch.setattr(
        ge,
        "fetch_repo_metadata",
        lambda urls, **kwargs: {
            urls[0]: {
                "owner": "openai",
                "repo": "repo-one",
            }
        },
    )

    summary = ge.run_repo_metadata_from_account(
        account_name="openai",
        hackathon_folder=tmp_path,
        provider_prefix="github_account",
        token="fake-token",
    )

    assert summary["repos_found"] == 1
    repo_json = Path(summary["outputs"]["json"])
    repo_parquet = Path(summary["outputs"]["parquet"])
    project_json = Path(summary["outputs"]["project_json"])
    project_parquet = Path(summary["outputs"]["project_parquet"])
    assert repo_json.exists()
    assert repo_parquet.exists()
    assert project_json.exists()
    assert project_parquet.exists()

    payload = json.loads(repo_json.read_text(encoding="utf-8"))
    assert payload["https://github.com/openai/repo-one"]["repo"] == "repo-one"
    assert payload["https://github.com/openai/repo-one"]["project_foreign_key"] is None

    project_payload = json.loads(project_json.read_text(encoding="utf-8"))
    assert project_payload[0]["project_fk"] is None
    assert project_payload[0]["project_uid"] is None
    assert project_payload[0]["github_repo_urls"] == ["https://github.com/openai/repo-one"]
