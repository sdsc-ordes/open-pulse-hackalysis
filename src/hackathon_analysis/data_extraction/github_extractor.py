from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from hackathon_analysis.data_extraction.models import (
    GitHubContributor,
    GitHubLanguageEntry,
    GitHubRepoMetadata,
    GitHubRepoMetadataError,
    GitHubRootEntry,
    GitHubRootFlags,
    ProjectRepoMappingRow,
)

logging.basicConfig(level=logging.INFO)


def parse_github_repo_url(url: str) -> Tuple[str, str]:
    url = url.strip()

    ssh = re.match(
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
    if ssh:
        return ssh.group("owner"), ssh.group("repo")

    m = re.match(
        r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+)", url)
    if not m:
        raise ValueError(f"Not a GitHub repo URL: {url}")

    owner = m.group("owner")
    repo = m.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def parse_github_url(url: str) -> Tuple[str, Optional[str]]:
    """Parse a GitHub URL and return (owner, repo).

    Returns (owner, None) for organization URLs like https://github.com/owner/
    Returns (owner, repo) for repository URLs like https://github.com/owner/repo
    """
    url = url.strip().rstrip("/")

    ssh_match = re.match(
        r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
    if ssh_match:
        return ssh_match.group("owner"), ssh_match.group("repo")

    https_match = re.match(
        r"^https?://github\.com/(?P<owner>[^/]+)(?:/(?P<repo>[^/#?]+))?", url)
    if not https_match:
        raise ValueError(f"Not a GitHub URL: {url}")

    owner = https_match.group("owner")
    repo = https_match.group("repo")

    if repo:
        if repo.endswith(".git"):
            repo = repo[:-4]
        return owner, repo

    return owner, None


def fetch_org_repos(client: "GitHubClient", org: str) -> List[Tuple[str, str]]:
    """Fetch all repositories from a GitHub organization.

    Returns list of (owner, repo) tuples.
    """
    repos = []
    page = 1
    per_page = 100

    while True:
        try:
            org_repos = client.rest_get(
                f"/orgs/{org}/repos",
                {"type": "public", "per_page": per_page,
                    "page": page, "sort": "updated"}
            )
            if not org_repos:
                break

            for repo_data in org_repos:
                if not repo_data.get("archived", False):
                    repos.append((org, repo_data["name"]))

            if len(org_repos) < per_page:
                break

            page += 1
        except Exception as e:
            logging.warning(
                f"Error fetching repos for organization '{org}': {e}")
            break

    return repos


def extract_readme_title(md: str, max_lines: int = 50) -> Optional[str]:
    """Extract README title efficiently with single pass.

    Looks for H1 header first, then returns first non-empty line.
    Only scans first max_lines to optimize performance.
    """
    if not md:
        return None

    for i, line in enumerate(md.splitlines()):
        if i >= max_lines:
            break
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()

    # Single pass for first non-empty line
    for i, line in enumerate(md.splitlines()):
        if i >= max_lines:
            break
        s = line.strip()
        if s:
            return s[:200]
    return None


def normalize_readme_text(md: str) -> str:
    if not md:
        return ""
    return md.replace("\r\n", "\n").strip()


def _parse_last_page_from_link(link_header: str) -> Optional[int]:
    if not link_header:
        return None
    parts = [p.strip() for p in link_header.split(",")]
    for p in parts:
        if 'rel="last"' in p:
            m = re.search(r"[?&]page=(\d+)", p)
            if m:
                return int(m.group(1))
    return None


def get_first_and_last_commit_dates_rest(
    client: "GitHubClient", owner: str, repo: str, branch: str
) -> Tuple[Optional[str], Optional[str]]:
    path = f"/repos/{owner}/{repo}/commits"

    newest_json, headers = client.rest_get_with_headers(
        path, {"sha": branch, "per_page": 1, "page": 1})
    last_date = newest_json[0]["commit"]["committer"]["date"] if newest_json else None

    last_page = _parse_last_page_from_link(headers.get("Link", ""))
    if not last_page:
        return None, last_date

    oldest_json = client.rest_get(
        path, {"sha": branch, "per_page": 1, "page": last_page})
    first_date = oldest_json[0]["commit"]["committer"]["date"] if oldest_json else None

    return first_date, last_date


def get_contributors_count_rest(
    client: "GitHubClient",
    owner: str,
    repo: str,
) -> Optional[int]:
    path = f"/repos/{owner}/{repo}/contributors"
    _, headers = client.rest_get_with_headers(
        path,
        {"per_page": 1, "anon": "true"},
    )

    last_page = _parse_last_page_from_link(headers.get("Link", ""))
    if last_page is not None:
        return last_page

    # if there is no Link header, there may be 0 or 1 page only
    try:
        data = client.rest_get(path, {"per_page": 100, "anon": "true"})
        if isinstance(data, list):
            return len(data)
    except Exception:
        return None

    return None


def derive_root_flags(root_entries: List[Dict[str, Any]]) -> Dict[str, bool]:
    paths = {str(x.get("path", "")).lower() for x in root_entries}

    has_tests = any(
        p in {"tests", "test"} or p.startswith("tests/") or p.startswith("test/")
        for p in paths
    )
    has_docs = any(
        p in {"docs", "doc"} or p.startswith("docs/") or p.startswith("doc/")
        for p in paths
    )
    has_ci = any(
        p.startswith(".github") or p in {".gitlab-ci.yml", "azure-pipelines.yml"}
        for p in paths
    )
    has_docker = any(
        "docker" in p or p == "dockerfile" or p.endswith("/dockerfile")
        for p in paths
    )
    has_notebooks = any(p.endswith(".ipynb") for p in paths)
    has_contributing = any("contributing" in p for p in paths)
    has_license_file = any(p == "license" or p.startswith("license.") for p in paths)
    has_readme_file = any(p == "readme.md" or p.startswith("readme.") for p in paths)

    return {
        "has_tests": has_tests,
        "has_docs": has_docs,
        "has_ci": has_ci,
        "has_docker": has_docker,
        "has_notebooks": has_notebooks,
        "has_contributing": has_contributing,
        "has_license_file": has_license_file,
        "has_readme_file": has_readme_file,
    }


def extract_topics(repo_data: Dict[str, Any]) -> List[str]:
    nodes = ((repo_data.get("repositoryTopics") or {}).get("nodes") or [])
    topics = []
    for n in nodes:
        topic = n.get("topic") or {}
        name = topic.get("name")
        if name:
            topics.append(name)
    return topics


def flatten_languages(repo_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    languages_top = []
    for e in ((repo_data.get("languages") or {}).get("edges") or []):
        node = e.get("node") or {}
        languages_top.append(
            {
                "language": node.get("name"),
                "size": e.get("size"),
            }
        )
    return languages_top


def _dump_model(model: Any) -> Dict[str, Any]:
    return model.model_dump(mode="python")


class GitHubClient:
    REST = "https://api.github.com"
    GRAPHQL = "https://api.github.com/graphql"

    def __init__(self, token: Optional[str] = None, timeout_s: int = 30):
        self.token = token
        self.timeout_s = timeout_s

    def headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github+json",
             "User-Agent": "repo-metadata-fetcher"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def rest_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        r = requests.get(
            f"{self.REST}{path}",
            params=params or {},
            headers=self.headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json()

    def rest_get_with_headers(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, str]]:
        r = requests.get(
            f"{self.REST}{path}",
            params=params or {},
            headers=self.headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return r.json(), dict(r.headers)

    def graphql(self, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(
            self.GRAPHQL,
            json={"query": query, "variables": variables},
            headers=self.headers(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        payload = r.json()
        if "errors" in payload:
            raise RuntimeError(payload["errors"])
        return payload["data"]


REPO_QUERY = """
query RepoMeta($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    name
    nameWithOwner
    url
    description
    homepageUrl

    repositoryTopics(first: 20) {
      nodes {
        topic {
          name
        }
      }
    }

    isPrivate
    isArchived
    isFork
    parent {
      nameWithOwner
      url
    }
    createdAt
    updatedAt
    pushedAt
    stargazerCount
    forkCount
    watchers { totalCount }
    primaryLanguage { name }
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      edges { size node { name } }
    }

    licenseInfo {
      spdxId
      name
    }

    defaultBranchRef {
      name
      target {
        __typename
        ... on Commit {
          history(first: 1) {
            totalCount
            edges {
              node {
                committedDate
                oid
              }
            }
          }
        }
      }
    }

    pullRequests { totalCount }
    openPullRequests: pullRequests(states: OPEN) { totalCount }
    mergedPullRequests: pullRequests(states: MERGED) { totalCount}
    closedPullRequests: pullRequests(states: CLOSED) { totalCount}

    issues { totalCount }

    openIssues: issues(states: OPEN) { totalCount }

    closedIssues: issues(states: CLOSED) {
      totalCount
    }

    releases {
      totalCount
    }

    latestRelease {
      tagName
      publishedAt
    }
  }
}
"""


def fetch_repo_metadata(
    repo_urls: List[str],
    token: Optional[str] = None,
    top_contributors: int = 10,
    max_root_entries: int = 200,
    include_readme_text: bool = False,
    fetch_readme: bool = True,
    readme_size_limit_bytes: Optional[int] = None,
    readme_text_max_bytes: int = 100_000,
) -> Dict[str, Dict[str, Any]]:
    client = GitHubClient(token=token)
    out: Dict[str, Dict[str, Any]] = {}

    logging.info(
        f"Fetching metadata for {len(repo_urls)} repositories from GitHub")

    for url in repo_urls:
        owner, repo = parse_github_repo_url(url)

        try:
            data = client.graphql(REPO_QUERY, {"owner": owner, "name": repo})
            repo_data = data.get("repository")
            if not repo_data:
                out[url] = _dump_model(
                    GitHubRepoMetadataError(
                        error="repo not found or no access",
                        owner=owner,
                        repo=repo,
                    )
                )
                continue

            repo_name = repo_data.get("name")
            default_branch = None
            commit_count = None
            first_commit_date = None
            last_commit_date = None
            last_commit_oid = None

            dbr = repo_data.get("defaultBranchRef")
            if dbr and dbr.get("target") and dbr["target"].get("__typename") == "Commit":
                default_branch = dbr.get("name")
                history = (dbr["target"].get("history") or {})
                commit_count = history.get("totalCount")
                newest_edges = history.get("edges") or []

                if newest_edges:
                    node = newest_edges[0].get("node") or {}
                    last_commit_oid = node.get("oid")
                    # graphQL gives last commit too, but keep REST as source of truth for first/last pair
                if default_branch:
                    try:
                        first_commit_date, last_commit_date = get_first_and_last_commit_dates_rest(
                            client,
                            owner,
                            repo,
                            default_branch,
                        )
                    except Exception as e:
                        logging.warning(f"Could not fetch first/last commit dates for {owner}/{repo}: {e}")

            readme_title = None
            readme_text = None
            readme_length = None
            if fetch_readme:
                try:
                    readme = client.rest_get(f"/repos/{owner}/{repo}/readme")
                    b64 = (readme or {}).get("content") or ""
                    if b64:
                        # Skip decoding if size limit is set and exceeded (fast path)
                        if readme_size_limit_bytes is not None and len(b64) > readme_size_limit_bytes:
                            logging.warning(
                                f"README too large for {owner}/{repo} (base64: {len(b64)} bytes), skipping"
                            )
                        else:
                            # Decode and process README
                            md = base64.b64decode(b64).decode("utf-8", errors="replace")
                            md = normalize_readme_text(md)
                            readme_length = len(md)
                            readme_title = extract_readme_title(md)

                            # Truncate for storage if needed
                            if len(md) > readme_text_max_bytes:
                                logging.warning(
                                    f"README for {owner}/{repo} truncated from {len(md)} to {readme_text_max_bytes} bytes"
                                )
                                if include_readme_text:
                                    readme_text = md[:readme_text_max_bytes]
                            else:
                                if include_readme_text:
                                    readme_text = md
                except Exception as e:
                    logging.warning(
                        f"Could not fetch README for {owner}/{repo}: {e}")

            contributors = []
            contributors_count = None
            try:
                contributors_count = get_contributors_count_rest(
                    client, owner, repo)

                lim = max(1, min(top_contributors, 100))
                contribs = client.rest_get(
                    f"/repos/{owner}/{repo}/contributors",
                    {"per_page": lim, "anon": "true"},
                )
                if isinstance(contribs, list):
                    for c in contribs[:top_contributors]:
                        contributor = GitHubContributor(
                            login=c.get("login") or c.get("name") or "anonymous",
                            contributions=c.get("contributions"),
                            html_url=c.get("html_url"),
                        )
                        contributors.append(_dump_model(contributor))
            except Exception as e:
                logging.warning(
                    f"Could not fetch contributors for {owner}/{repo}: {e}")

            root_entries = []
            files_total = None
            dirs_total = None
            if default_branch:
                try:
                    root = client.rest_get(
                        f"/repos/{owner}/{repo}/contents/",
                        {"ref": default_branch},
                    )
                    if isinstance(root, list):
                        for item in root[:max_root_entries]:
                            root_entries.append(
                                _dump_model(
                                    GitHubRootEntry(
                                        path=item.get("path"),
                                        type=item.get("type"),
                                        size=item.get("size"),
                                    )
                                )
                            )

                    branch = client.rest_get(
                        f"/repos/{owner}/{repo}/branches/{default_branch}")
                    tree_sha = (branch.get("commit") or {}).get(
                        "commit", {}).get("tree", {}).get("sha")
                    if tree_sha:
                        tree = client.rest_get(
                            f"/repos/{owner}/{repo}/git/trees/{tree_sha}",
                            {"recursive": "1"},
                        )
                        nodes = tree.get("tree") or []
                        files_total = sum(
                            1 for n in nodes if n.get("type") == "blob")
                        dirs_total = sum(
                            1 for n in nodes if n.get("type") == "tree")
                except Exception as e:
                    logging.warning(
                        f"Could not fetch file tree info for {owner}/{repo}: {e}")

            topics = extract_topics(repo_data)
            languages_top = [
                _dump_model(GitHubLanguageEntry.model_validate(lang))
                for lang in flatten_languages(repo_data)
            ]
            root_flags = _dump_model(
                GitHubRootFlags.model_validate(derive_root_flags(root_entries))
            )

            description = repo_data.get("description")
            homepage_url = repo_data.get("homepageUrl")

            metadata = GitHubRepoMetadata(
                owner=owner,
                repo_name=repo_name,
                repo=repo,
                name_with_owner=repo_data.get("nameWithOwner"),
                url=repo_data.get("url"),
                description=description,
                homepage_url=homepage_url,
                topics=topics,
                is_private=repo_data.get("isPrivate"),
                is_archived=repo_data.get("isArchived"),
                is_fork=repo_data.get("isFork"),
                parent_repo=(repo_data.get("parent") or {}).get("nameWithOwner"),
                parent_url=(repo_data.get("parent") or {}).get("url"),
                default_branch=default_branch,
                created_at=repo_data.get("createdAt"),
                updated_at=repo_data.get("updatedAt"),
                pushed_at=repo_data.get("pushedAt"),
                stars=repo_data.get("stargazerCount"),
                forks=repo_data.get("forkCount"),
                watchers=(repo_data.get("watchers") or {}).get("totalCount"),
                primary_language=(repo_data.get("primaryLanguage") or {}).get("name"),
                languages_top=languages_top,
                license_spdx=(repo_data.get("licenseInfo") or {}).get("spdxId"),
                license_name=(repo_data.get("licenseInfo") or {}).get("name"),
                commit_count_default_branch=commit_count,
                first_commit_date_default_branch=first_commit_date,
                last_commit_date_default_branch=last_commit_date,
                last_commit_oid_default_branch=last_commit_oid,
                pull_requests_total=(repo_data.get("pullRequests") or {}).get("totalCount"),
                pull_requests_open=(repo_data.get("openPullRequests") or {}).get("totalCount"),
                pull_requests_closed=(repo_data.get("closedPullRequests") or {}).get("totalCount"),
                pull_requests_merged=(repo_data.get("mergedPullRequests") or {}).get("totalCount"),
                issues_total=(repo_data.get("issues") or {}).get("totalCount"),
                issues_open=(repo_data.get("openIssues") or {}).get("totalCount"),
                issues_closed=(repo_data.get("closedIssues") or {}).get("totalCount"),
                releases_count=(repo_data.get("releases") or {}).get("totalCount"),
                latest_release_tag=(repo_data.get("latestRelease") or {}).get("tagName"),
                latest_release_date=(repo_data.get("latestRelease") or {}).get("publishedAt"),
                contributors_count=contributors_count,
                contributors_top=contributors,
                readme_title=readme_title,
                readme_text=readme_text,
                readme_length=readme_length,
                files_root_entries=root_entries,
                files_total_count=files_total,
                dirs_total_count=dirs_total,
                **root_flags,
            )
            out[url] = _dump_model(metadata)

        except Exception as e:
            out[url] = _dump_model(
                GitHubRepoMetadataError(
                    error=str(e),
                    owner=owner,
                    repo=repo,
                )
            )

    return out


def extract_github_urls_from_df(df: pd.DataFrame, token: Optional[str] = None) -> List[str]:
    if "url" not in df.columns:
        logging.error("Column 'url' not found in dataframe")
        return []

    # Updated regex to capture organization URLs (single segment) and repo URLs (two segments)
    url_re = re.compile(
        r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?",
        re.IGNORECASE,
    )

    urls: set[str] = set()
    org_urls: set[str] = set()  # Track original org URLs for logging
    total_matches = 0
    client = GitHubClient(token=token) if token else None

    logging.info(f"Scanning {len(df)} project rows from column 'url'")

    for idx, cell in df["url"].dropna().astype(str).items():
        matches = url_re.findall(cell)
        title = df.loc[idx].get("title", "")[:50] if "title" in df.columns else ""
        logging.info(f"row={idx} repo_urls_found={len(matches)} title='{title}'")

        for owner, repo in matches:
            if repo:
                # It's a repository URL
                repo = repo.rstrip(").,;]}>#")
                if repo.lower().endswith(".git"):
                    repo = repo[:-4]
                urls.add(f"https://github.com/{owner}/{repo}")
                total_matches += 1
            else:
                # It's an organization URL - fetch its repositories
                org_url = f"https://github.com/{owner}"
                if org_url not in org_urls and client:
                    org_urls.add(org_url)
                    logging.info(
                        f"Fetching repositories from organization: {owner}")
                    try:
                        org_repos = fetch_org_repos(client, owner)
                        for org, repo_name in org_repos:
                            urls.add(f"https://github.com/{org}/{repo_name}")
                            total_matches += 1
                        logging.info(
                            f"Found {len(org_repos)} repositories in organization '{owner}'")
                    except Exception as e:
                        logging.warning(
                            f"Failed to fetch repositories for organization '{owner}': {e}")

    logging.info(f"Total repo URL matches found: {total_matches}")
    logging.info(f"Unique GitHub repos extracted: {len(urls)}")

    return sorted(urls)


def build_project_repo_mapping(
    df: pd.DataFrame,
    token: Optional[str] = None,
    project_id_col: str = "id",
    project_title_col: str = "title",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Build per-project GitHub repo mapping while preserving project row identity.

    Returns:
    - project_rows: one entry per input project row with derived ids and repo URLs
    - unique_repo_urls: deduplicated repo URLs across all projects
    """
    if "url" not in df.columns:
        logging.error("Column 'url' not found in dataframe")
        return [], []

    url_re = re.compile(
        r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)(?:/([A-Za-z0-9_.-]+))?",
        re.IGNORECASE,
    )

    client = GitHubClient(token=token) if token else None
    org_repo_cache: Dict[str, List[str]] = {}
    project_rows: List[Dict[str, Any]] = []
    all_repo_urls: set[str] = set()
    seen_uids: set[str] = set()

    logging.info(f"Building project-repo mapping from {len(df)} project rows")

    for idx, row in df.iterrows():
        cell = str(row.get("url", "") or "")
        matches = url_re.findall(cell)
        repo_urls: set[str] = set()

        for owner, repo in matches:
            if repo:
                repo = repo.rstrip(").,;]}>#")
                if repo.lower().endswith(".git"):
                    repo = repo[:-4]
                repo_urls.add(f"https://github.com/{owner}/{repo}")
                continue

            if not client:
                continue

            org = owner
            if org not in org_repo_cache:
                logging.info(f"Fetching repositories from organization: {org}")
                try:
                    org_repo_cache[org] = [
                        f"https://github.com/{org_name}/{repo_name}"
                        for org_name, repo_name in fetch_org_repos(client, org)
                    ]
                    logging.info(
                        f"Found {len(org_repo_cache[org])} repositories in organization '{org}'"
                    )
                except Exception as e:
                    logging.warning(
                        f"Failed to fetch repositories for organization '{org}': {e}"
                    )
                    org_repo_cache[org] = []

            repo_urls.update(org_repo_cache[org])

        project_id = row.get(project_id_col) if project_id_col in df.columns else None
        if pd.isna(project_id):
            project_id = None
        project_id = str(project_id).strip() if project_id is not None else None
        if project_id == "":
            project_id = None

        project_title = row.get(project_title_col) if project_title_col in df.columns else None
        if pd.isna(project_title):
            project_title = None
        project_title = str(project_title).strip() if project_title is not None else None
        if project_title == "":
            project_title = None

        base_uid = f"{project_id_col}:{project_id}" if project_id else f"row_index:{idx}"
        project_uid = base_uid
        if project_uid in seen_uids:
            project_uid = f"{base_uid}|row:{idx}"
        seen_uids.add(project_uid)

        repo_urls_list = sorted(repo_urls)
        all_repo_urls.update(repo_urls_list)

        project_rows.append(
            _dump_model(
                ProjectRepoMappingRow(
                    source_row_index=(
                        int(idx) if isinstance(idx, int) else str(idx)
                    ),
                    project_uid=project_uid,
                    project_id=project_id,
                    project_title=project_title,
                    github_repo_urls=repo_urls_list,
                    github_repo_count=len(repo_urls_list),
                )
            )
        )

    logging.info(f"Unique GitHub repos extracted: {len(all_repo_urls)}")
    return project_rows, sorted(all_repo_urls)


def write_project_metadata_outputs(
    hackathon_folder: Path,
    provider_prefix: str,
    projects_df: pd.DataFrame,
    project_rows: List[Dict[str, Any]],
    repo_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Path]:
    """Write project-level GitHub outputs (one row per input project)."""
    hackathon_folder.mkdir(parents=True, exist_ok=True)
    project_json_path = hackathon_folder / f"{provider_prefix}_github_project_metadata.json"
    project_parquet_path = hackathon_folder / f"{provider_prefix}_github_project_metadata.parquet"

    # Keep project-level output schema explicit to avoid non-JSON-native parquet types.
    base_columns = [
        "id",
        "title",
        "description",
        "url",
        "team",
        "tags",
        "image_url",
        "awards",
        "categories",
        "hackathon_name",
        "hackathon_year",
        "hackathon_location",
    ]
    selected_base_columns = [c for c in base_columns if c in projects_df.columns]
    projects_by_index: Dict[str, Dict[str, Any]] = {}
    def _normalize_base_value(v: Any) -> Any:
        if v is None:
            return None

        if isinstance(v, dict):
            return {str(k): _normalize_base_value(val) for k, val in v.items()}

        if isinstance(v, (list, tuple, set)):
            return [_normalize_base_value(val) for val in v]

        # numpy/pandas array-likes
        if hasattr(v, "tolist") and not isinstance(v, (str, bytes, bytearray)):
            return _normalize_base_value(v.tolist())

        # numpy scalars
        if hasattr(v, "item") and not isinstance(v, (str, bytes, bytearray)):
            item_v = v.item()
            if item_v is not v:
                return _normalize_base_value(item_v)

        # datetime-like objects
        if hasattr(v, "isoformat") and not isinstance(v, (str, bytes, bytearray)):
            return v.isoformat()

        na_mask = pd.isna(v)
        # pd.isna can return array-like for non-scalar values; only coerce scalar NA.
        if not hasattr(na_mask, "__len__"):
            return None if bool(na_mask) else v
        return v

    for i, row in projects_df.iterrows():
        idx_key = str(i if not isinstance(i, int) else int(i))
        row_data = {c: row[c] for c in selected_base_columns}
        projects_by_index[idx_key] = {
            c: _normalize_base_value(v) for c, v in row_data.items()
        }

    merged_rows: List[Dict[str, Any]] = []
    for proj in project_rows:
        idx_key = str(proj["source_row_index"])
        base = dict(projects_by_index.get(idx_key, {}))

        repo_urls = proj.get("github_repo_urls") or []
        repo_items = []
        for u in repo_urls:
            m = repo_meta.get(u)
            if isinstance(m, dict):
                repo_items.append({"input_url": u, **m})
            else:
                repo_items.append({"input_url": u, "error": "metadata missing"})

        merged_rows.append(
            {
                **base,
                "project_uid": proj.get("project_uid"),
                "project_id": proj.get("project_id"),
                "project_title": proj.get("project_title"),
                "github_repo_urls": repo_urls,
                "github_repo_count": proj.get("github_repo_count", len(repo_urls)),
                "github_repos_metadata": repo_items,
            }
        )

    def _deep_to_python(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, dict):
            return {str(k): _deep_to_python(val) for k, val in v.items()}
        if isinstance(v, (list, tuple, set)):
            return [_deep_to_python(val) for val in v]
        if hasattr(v, "tolist") and not isinstance(v, (str, bytes, bytearray)):
            return _deep_to_python(v.tolist())
        if hasattr(v, "item") and not isinstance(v, (str, bytes, bytearray)):
            item_v = v.item()
            if item_v is not v:
                return _deep_to_python(item_v)
        return v

    merged_rows = _deep_to_python(merged_rows)

    project_json_path.write_text(
        json.dumps(merged_rows, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    project_df = pd.DataFrame(merged_rows)
    for col in project_df.columns:
        if project_df[col].map(lambda x: isinstance(x, (list, dict))).any():
            project_df[col] = project_df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict))
                else x
            )

    project_df.to_parquet(project_parquet_path, index=False)
    return {"project_json": project_json_path, "project_parquet": project_parquet_path}


def write_repo_metadata_outputs(
    hackathon_folder: Path,
    provider_prefix: str,
    repo_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Path]:
    """
    Writes raw JSON and a flattened parquet table into the hackathon_folder.
    Returns paths.
    """
    hackathon_folder.mkdir(parents=True, exist_ok=True)

    raw_json_path = hackathon_folder / \
        f"{provider_prefix}_github_repo_metadata.json"
    parquet_path = hackathon_folder / \
        f"{provider_prefix}_github_repo_metadata.parquet"

    raw_json_path.write_text(
        json.dumps(repo_meta, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for url, meta in repo_meta.items():
        row = {"input_url": url}
        if isinstance(meta, dict):
            row.update(meta)
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in ["topics", "languages_top", "contributors_top", "files_root_entries"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False)
                if isinstance(x, (list, dict))
                else x
            )

    df.to_parquet(parquet_path, index=False)

    return {"json": raw_json_path, "parquet": parquet_path}


def run_repo_metadata_from_projects_parquet(
    projects_parquet: Path,
    hackathon_folder: Path,
    provider_prefix: str,
    *,
    token: Optional[str] = None,
    top_contributors: int = 8,
    max_root_entries: int = 200,
    include_readme_text: bool = False,
    fetch_readme: bool = True,
    readme_size_limit_bytes: Optional[int] = None,
    readme_text_max_bytes: int = 100_000,
    write_repo_level_output: bool = True,
    write_project_level_output: bool = True,
) -> Dict[str, Any]:
    """
    End to end helper:
    reads projects parquet
    extracts GitHub URLs
    fetches repo metadata
    writes outputs

    Returns a small summary dict.
    """
    df = pd.read_parquet(projects_parquet)
    logging.info(f"projects rows={len(df)} cols={list(df.columns)[:10]}")

    project_rows, urls = build_project_repo_mapping(df, token=token)
    logging.info(f"github urls found={len(urls)}")

    repo_meta = fetch_repo_metadata(
        urls,
        token=token,
        top_contributors=top_contributors,
        max_root_entries=max_root_entries,
        include_readme_text=include_readme_text,
        fetch_readme=fetch_readme,
        readme_size_limit_bytes=readme_size_limit_bytes,
        readme_text_max_bytes=readme_text_max_bytes,
    )

    out_paths: Dict[str, Path] = {}
    if write_repo_level_output:
        out_paths.update(
            write_repo_metadata_outputs(
                hackathon_folder=hackathon_folder,
                provider_prefix=provider_prefix,
                repo_meta=repo_meta,
            )
        )

    if write_project_level_output:
        out_paths.update(
            write_project_metadata_outputs(
                hackathon_folder=hackathon_folder,
                provider_prefix=provider_prefix,
                projects_df=df,
                project_rows=project_rows,
                repo_meta=repo_meta,
            )
        )

    return {
        "projects_parquet": str(projects_parquet),
        "hackathon_folder": str(hackathon_folder),
        "provider_prefix": provider_prefix,
        "projects_rows": len(df),
        "repos_found": len(urls),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }


if __name__ == "__main__":
    urls = ["https://github.com/sdsc-ordes/gimie"]
    token = os.getenv("GITHUB_TOKEN")
    print("token_present", bool(token), "token_len", len(token or ""))
    print("fetching metadata for:", urls)
    meta = fetch_repo_metadata(
        urls,
        token=token,
        top_contributors=8,
        include_readme_text=True,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))
