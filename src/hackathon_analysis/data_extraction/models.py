"""Pydantic schemas for extraction pipelines."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class GitHubContributor(BaseModel):
    """Top contributor entry for a repository."""

    login: str
    contributions: Optional[int] = None
    html_url: Optional[str] = None


class GitHubLanguageEntry(BaseModel):
    """Language + byte size entry."""

    language: Optional[str] = None
    size: Optional[int] = None


class GitHubRootEntry(BaseModel):
    """Root-level repository tree entry."""

    path: Optional[str] = None
    type: Optional[str] = None
    size: Optional[int] = None


class GitHubRootFlags(BaseModel):
    """Derived boolean flags from root entries."""

    has_tests: bool
    has_docs: bool
    has_ci: bool
    has_docker: bool
    has_notebooks: bool
    has_contributing: bool
    has_license_file: bool
    has_readme_file: bool


class GitHubRepoMetadata(BaseModel):
    """Normalized metadata payload for a single repository."""

    owner: str
    repo_name: Optional[str] = None
    repo: str
    name_with_owner: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    homepage_url: Optional[str] = None
    topics: List[str] = Field(default_factory=list)
    is_private: Optional[bool] = None
    is_archived: Optional[bool] = None
    is_fork: Optional[bool] = None
    parent_repo: Optional[str] = None
    parent_url: Optional[str] = None
    default_branch: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    pushed_at: Optional[str] = None
    stars: Optional[int] = None
    forks: Optional[int] = None
    watchers: Optional[int] = None
    primary_language: Optional[str] = None
    languages_top: List[GitHubLanguageEntry] = Field(default_factory=list)
    license_spdx: Optional[str] = None
    license_name: Optional[str] = None
    commit_count_default_branch: Optional[int] = None
    first_commit_date_default_branch: Optional[str] = None
    last_commit_date_default_branch: Optional[str] = None
    last_commit_oid_default_branch: Optional[str] = None
    pull_requests_total: Optional[int] = None
    pull_requests_open: Optional[int] = None
    pull_requests_closed: Optional[int] = None
    pull_requests_merged: Optional[int] = None
    issues_total: Optional[int] = None
    issues_open: Optional[int] = None
    issues_closed: Optional[int] = None
    releases_count: Optional[int] = None
    latest_release_tag: Optional[str] = None
    latest_release_date: Optional[str] = None
    contributors_count: Optional[int] = None
    contributors_top: List[GitHubContributor] = Field(default_factory=list)
    readme_title: Optional[str] = None
    readme_text: Optional[str] = None
    readme_length: Optional[int] = None
    files_root_entries: List[GitHubRootEntry] = Field(default_factory=list)
    files_total_count: Optional[int] = None
    dirs_total_count: Optional[int] = None
    has_tests: bool
    has_docs: bool
    has_ci: bool
    has_docker: bool
    has_notebooks: bool
    has_contributing: bool
    has_license_file: bool
    has_readme_file: bool


class GitHubRepoMetadataError(BaseModel):
    """Error payload for repository metadata extraction failures."""

    error: str
    owner: str
    repo: str


class ProjectRepoMappingRow(BaseModel):
    """Project row to GitHub repo URL mapping."""

    source_row_index: int | str
    project_uid: str
    project_id: Optional[str] = None
    project_title: Optional[str] = None
    github_repo_urls: List[str] = Field(default_factory=list)
    github_repo_count: int


class LauzHackProject(BaseModel):
    """Normalized LauzHack project record."""

    model_config = ConfigDict(extra="allow")

    id: Optional[int | str] = None
    title: str
    description: str = ""
    url: str = ""
    team: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    image_url: str = ""
    awards: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    hackathon_name: Optional[str] = None
    hackathon_year: Optional[int | str] = None
    hackathon_location: Optional[str] = None


class LauzHackMetadata(BaseModel):
    """Normalized LauzHack metadata record."""

    model_config = ConfigDict(extra="allow")

    source_url: Optional[str] = None
    year: Optional[int] = None
    name: str = "LauzHack"
    description: Optional[str] = None
    date: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
    location: str = "EPFL, Lausanne, Switzerland"
    sponsors: Optional[List[str]] = None
    prizes: Optional[List[str]] = None
    social_links: Optional[Dict[str, str]] = None
    extracted_at: Optional[str] = None
