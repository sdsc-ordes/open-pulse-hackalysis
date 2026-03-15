"""Pydantic schemas for extraction pipelines."""

from __future__ import annotations

from datetime import date as dt_date
from datetime import datetime as dt_datetime
from typing import Any, Dict, List, Optional

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    field_validator,
)


def _strip_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def _parse_datetime(value: Any) -> Any:
    value = _strip_or_none(value)
    if value is None or isinstance(value, dt_datetime):
        return value
    if isinstance(value, dt_date):
        return dt_datetime.combine(value, dt_datetime.min.time())
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        return dt_datetime.fromisoformat(normalized)
    return value


def _parse_date(value: Any) -> Any:
    value = _strip_or_none(value)
    if value is None or isinstance(value, dt_date):
        return value
    if isinstance(value, dt_datetime):
        return value.date()
    if isinstance(value, str):
        return dt_date.fromisoformat(value)
    return value


def _string_list(value: Any) -> Any:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return value


class GitHubContributor(BaseModel):
    """Top contributor entry for a repository."""

    login: str
    contributions: Optional[NonNegativeInt] = None
    html_url: Optional[AnyHttpUrl] = None

    @field_validator("login", mode="before")
    @classmethod
    def _normalize_login(cls, value: Any) -> str:
        return str(_strip_or_none(value) or "anonymous")

    @field_validator("html_url", mode="before")
    @classmethod
    def _normalize_html_url(cls, value: Any) -> Any:
        return _strip_or_none(value)


class GitHubLanguageEntry(BaseModel):
    """Language + byte size entry."""

    language: Optional[str] = None
    size: Optional[NonNegativeInt] = None

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> Any:
        return _strip_or_none(value)


class GitHubRootEntry(BaseModel):
    """Root-level repository tree entry."""

    path: Optional[str] = None
    type: Optional[str] = None
    size: Optional[NonNegativeInt] = None

    @field_validator("path", "type", mode="before")
    @classmethod
    def _normalize_path_like(cls, value: Any) -> Any:
        return _strip_or_none(value)


class GitHubRootFlags(BaseModel):
    """Derived boolean flags from root entries."""

    has_tests: bool = False
    has_docs: bool = False
    has_ci: bool = False
    has_docker: bool = False
    has_notebooks: bool = False
    has_contributing: bool = False
    has_license_file: bool = False
    has_readme_file: bool = False


class GitHubRepoMetadata(BaseModel):
    """Normalized metadata payload for a single repository."""

    owner: str
    repo_name: Optional[str] = None
    repo: str
    name_with_owner: Optional[str] = None
    url: Optional[AnyHttpUrl] = None
    description: Optional[str] = None
    homepage_url: Optional[AnyHttpUrl] = None
    topics: List[str] = Field(default_factory=list)
    is_private: Optional[bool] = False
    is_archived: Optional[bool] = False
    is_fork: Optional[bool] = False
    parent_repo: Optional[str] = None
    parent_url: Optional[AnyHttpUrl] = None
    default_branch: Optional[str] = None
    created_at: Optional[dt_datetime] = None
    updated_at: Optional[dt_datetime] = None
    pushed_at: Optional[dt_datetime] = None
    stars: Optional[NonNegativeInt] = 0
    forks: Optional[NonNegativeInt] = 0
    watchers: Optional[NonNegativeInt] = 0
    primary_language: Optional[str] = None
    languages_top: List[GitHubLanguageEntry] = Field(default_factory=list)
    license_spdx: Optional[str] = None
    license_name: Optional[str] = "Unknown"
    commit_count_default_branch: Optional[NonNegativeInt] = 0
    first_commit_date_default_branch: Optional[dt_datetime] = None
    last_commit_date_default_branch: Optional[dt_datetime] = None
    last_commit_oid_default_branch: Optional[str] = None
    pull_requests_total: Optional[NonNegativeInt] = 0
    pull_requests_open: Optional[NonNegativeInt] = 0
    pull_requests_closed: Optional[NonNegativeInt] = 0
    pull_requests_merged: Optional[NonNegativeInt] = 0
    issues_total: Optional[NonNegativeInt] = 0
    issues_open: Optional[NonNegativeInt] = 0
    issues_closed: Optional[NonNegativeInt] = 0
    releases_count: Optional[NonNegativeInt] = 0
    latest_release_tag: Optional[str] = None
    latest_release_date: Optional[dt_datetime] = None
    contributors_count: Optional[NonNegativeInt] = None
    contributors_top: List[GitHubContributor] = Field(default_factory=list)
    readme_title: Optional[str] = None
    readme_text: Optional[str] = ""
    readme_length: Optional[NonNegativeInt] = 0
    files_root_entries: List[GitHubRootEntry] = Field(default_factory=list)
    files_total_count: Optional[NonNegativeInt] = None
    dirs_total_count: Optional[NonNegativeInt] = None
    project_foreign_key: Optional[str] = None
    has_tests: bool = False
    has_docs: bool = False
    has_ci: bool = False
    has_docker: bool = False
    has_notebooks: bool = False
    has_contributing: bool = False
    has_license_file: bool = False
    has_readme_file: bool = False

    @field_validator(
        "owner",
        "repo",
        "repo_name",
        "name_with_owner",
        "url",
        "description",
        "homepage_url",
        "parent_repo",
        "parent_url",
        "default_branch",
        "primary_language",
        "license_spdx",
        "license_name",
        "last_commit_oid_default_branch",
        "latest_release_tag",
        "readme_title",
        "readme_text",
        "project_foreign_key",
        mode="before",
    )
    @classmethod
    def _normalize_text_fields(cls, value: Any) -> Any:
        return _strip_or_none(value)

    @field_validator(
        "created_at",
        "updated_at",
        "pushed_at",
        "first_commit_date_default_branch",
        "last_commit_date_default_branch",
        "latest_release_date",
        mode="before",
    )
    @classmethod
    def _normalize_datetime_fields(cls, value: Any) -> Any:
        return _parse_datetime(value)

    @field_validator("topics", mode="before")
    @classmethod
    def _normalize_topics(cls, value: Any) -> Any:
        return _string_list(value)


class GitHubRepoMetadataError(BaseModel):
    """Error payload for repository metadata extraction failures."""

    error: str
    owner: str
    repo: str


class ProjectRepoMappingRow(BaseModel):
    """Project row to GitHub repo URL mapping."""

    source_row_index: int | str
    project_fk: str
    project_uid: str
    project_id: Optional[str] = None
    project_title: Optional[str] = None
    github_repo_urls: List[str] = Field(default_factory=list)
    github_repo_count: NonNegativeInt

    @field_validator("project_fk", "project_uid", "project_id", "project_title", mode="before")
    @classmethod
    def _normalize_project_text(cls, value: Any) -> Any:
        return _strip_or_none(value)

    @field_validator("github_repo_urls", mode="before")
    @classmethod
    def _normalize_repo_urls(cls, value: Any) -> Any:
        return _string_list(value)


class GitHubProjectMetadataRow(BaseModel):
    """Project-shaped GitHub metadata output row."""

    model_config = ConfigDict(extra="allow")

    id: Optional[NonNegativeInt | str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    url: Optional[str] = None
    team: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    image_url: Optional[str] = ""
    awards: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    hackathon_name: Optional[str] = None
    hackathon_year: Optional[NonNegativeInt | str] = None
    hackathon_location: Optional[str] = None
    project_fk: Optional[str] = None
    project_uid: Optional[str] = None
    project_id: Optional[str] = None
    project_title: Optional[str] = None
    github_repo_urls: List[str] = Field(default_factory=list)
    github_repo_count: NonNegativeInt = 0
    github_repos_metadata: List[Dict[str, Any]] = Field(default_factory=list)

    @field_validator(
        "title",
        "description",
        "url",
        "image_url",
        "hackathon_name",
        "hackathon_location",
        "project_fk",
        "project_uid",
        "project_id",
        "project_title",
        mode="before",
    )
    @classmethod
    def _normalize_text_like(cls, value: Any) -> Any:
        return _strip_or_none(value)

    @field_validator("team", "tags", "awards", "categories", "github_repo_urls", mode="before")
    @classmethod
    def _normalize_list_like(cls, value: Any) -> Any:
        return _string_list(value)


class LauzHackProject(BaseModel):
    """Normalized LauzHack project record."""

    model_config = ConfigDict(extra="allow")

    id: Optional[NonNegativeInt | str] = None
    project_hard_id: str
    title: str
    description: str = ""
    url: str = ""
    team: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    image_url: str = ""
    awards: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    hackathon_name: Optional[str] = None
    hackathon_year: Optional[NonNegativeInt | str] = None
    hackathon_location: Optional[str] = None

    @field_validator(
        "title",
        "project_hard_id",
        "description",
        "url",
        "image_url",
        "hackathon_name",
        "hackathon_location",
        mode="before",
    )
    @classmethod
    def _normalize_lauzhack_text(cls, value: Any) -> Any:
        if value is None:
            return None
        return str(value).strip()

    @field_validator("team", "tags", "awards", "categories", mode="before")
    @classmethod
    def _normalize_lauzhack_lists(cls, value: Any) -> Any:
        return _string_list(value)


class LauzHackMetadata(BaseModel):
    """Normalized LauzHack metadata record."""

    model_config = ConfigDict(extra="allow")

    source_url: Optional[AnyHttpUrl] = None
    year: Optional[NonNegativeInt] = None
    name: str = "LauzHack"
    description: Optional[str] = None
    date: Optional[str] = None
    date_start: Optional[dt_date] = None
    date_end: Optional[dt_date] = None
    location: str = "EPFL, Lausanne, Switzerland"
    sponsors: Optional[List[str]] = None
    prizes: Optional[List[str]] = None
    social_links: Optional[Dict[str, AnyHttpUrl]] = None
    extracted_at: Optional[dt_datetime] = None

    @field_validator("source_url", "name", "description", "date", "location", mode="before")
    @classmethod
    def _normalize_metadata_text(cls, value: Any) -> Any:
        return _strip_or_none(value)

    @field_validator("date_start", "date_end", mode="before")
    @classmethod
    def _normalize_date_bounds(cls, value: Any) -> Any:
        return _parse_date(value)

    @field_validator("extracted_at", mode="before")
    @classmethod
    def _normalize_extracted_at(cls, value: Any) -> Any:
        return _parse_datetime(value)

    @field_validator("sponsors", "prizes", mode="before")
    @classmethod
    def _normalize_optional_lists(cls, value: Any) -> Any:
        if value is None:
            return None
        return _string_list(value)
