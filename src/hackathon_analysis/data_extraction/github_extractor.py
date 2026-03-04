from __future__ import annotations

import base64
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests


def parse_github_repo_url(url: str) -> Tuple[str, str]:
    url = url.strip()

    ssh = re.match(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$", url)
    if ssh:
        return ssh.group("owner"), ssh.group("repo")

    m = re.match(r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/#?]+)", url)
    if not m:
        raise ValueError(f"Not a GitHub repo URL: {url}")

    owner = m.group("owner")
    repo = m.group("repo")
    if repo.endswith(".git"):
        repo = repo[:-4]
    return owner, repo


def extract_readme_title(md: str) -> Optional[str]:
    if not md:
        return None
    for line in md.splitlines():
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    for line in md.splitlines():
        s = line.strip()
        if s:
            return s[:200]
    return None


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

    newest_json, headers = client.rest_get_with_headers(path, {"sha": branch, "per_page": 1, "page": 1})
    last_date = newest_json[0]["commit"]["committer"]["date"] if newest_json else None

    last_page = _parse_last_page_from_link(headers.get("Link", ""))
    if not last_page:
        return None, last_date

    oldest_json = client.rest_get(path, {"sha": branch, "per_page": 1, "page": last_page})
    first_date = oldest_json[0]["commit"]["committer"]["date"] if oldest_json else None

    return first_date, last_date


class GitHubClient:
    REST = "https://api.github.com"
    GRAPHQL = "https://api.github.com/graphql"

    def __init__(self, token: Optional[str] = None, timeout_s: int = 30):
        self.token = token
        self.timeout_s = timeout_s

    def headers(self) -> Dict[str, str]:
        h = {"Accept": "application/vnd.github+json", "User-Agent": "repo-metadata-fetcher"}
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
    nameWithOwner
    url
    description
    isPrivate
    isArchived
    isFork
    parent {
      nameWithOwner
      url
    }
    createdAt
    updatedAt
    stargazerCount
    forkCount
    watchers { totalCount }
    primaryLanguage { name }
    languages(first: 10, orderBy: {field: SIZE, direction: DESC}) {
      edges { size node { name } }
    }
    defaultBranchRef {
      name
      target {
        __typename
        ... on Commit {
          newest: history(first: 1) { totalCount edges { node { committedDate oid } } }
        }
      }
    }
    pullRequests { totalCount }
    openPullRequests: pullRequests(states: OPEN) { totalCount }
    issues { totalCount }
    openIssues: issues(states: OPEN) { totalCount }
  }
}
"""


def fetch_repo_metadata(
    repo_urls: List[str],
    token: Optional[str] = None,
    top_contributors: int = 10,
    max_root_entries: int = 200,
) -> Dict[str, Dict[str, Any]]:
    client = GitHubClient(token=token)
    out: Dict[str, Dict[str, Any]] = {}

    for url in repo_urls:
        owner, repo = parse_github_repo_url(url)

        try:
            data = client.graphql(REPO_QUERY, {"owner": owner, "name": repo})
            r = data.get("repository")
            if not r:
                out[url] = {"error": "repo not found or no access", "owner": owner, "repo": repo}
                continue

            default_branch = None
            commit_count = None
            first_commit_date = None
            last_commit_date = None
            first_commit_oid = None
            last_commit_oid = None

            dbr = r.get("defaultBranchRef")
            if dbr and dbr.get("target") and dbr["target"].get("__typename") == "Commit":
                default_branch = dbr.get("name")
                tgt = dbr["target"]

                newest_edges = (tgt.get("newest") or {}).get("edges") or []
                commit_count = (tgt.get("newest") or {}).get("totalCount")

                if newest_edges:
                    last_commit_oid = newest_edges[0]["node"]["oid"]

                if default_branch:
                    try:
                        first_commit_date, last_commit_date = get_first_and_last_commit_dates_rest(
                            client, owner, repo, default_branch
                        )
                    except Exception:
                        pass

            # README title
            readme_title = None
            try:
                readme = client.rest_get(f"/repos/{owner}/{repo}/readme")
                b64 = (readme or {}).get("content") or ""
                if b64:
                    md = base64.b64decode(b64).decode("utf-8", errors="replace")
                    readme_title = extract_readme_title(md)
            except Exception:
                pass

            # Contributors
            contributors = []
            try:
                lim = max(1, min(top_contributors, 100))
                contribs = client.rest_get(
                    f"/repos/{owner}/{repo}/contributors",
                    {"per_page": lim, "anon": "true"},
                )
                if isinstance(contribs, list):
                    for c in contribs[:top_contributors]:
                        contributors.append(
                            {
                                "login": c.get("login") or c.get("name") or "anonymous",
                                "contributions": c.get("contributions"),
                                "html_url": c.get("html_url"),
                            }
                        )
            except Exception:
                pass

            # Files info (root + total counts via tree)
            root_entries = []
            files_total = None
            dirs_total = None
            if default_branch:
                try:
                    root = client.rest_get(f"/repos/{owner}/{repo}/contents/", {"ref": default_branch})
                    if isinstance(root, list):
                        for item in root[:max_root_entries]:
                            root_entries.append(
                                {"path": item.get("path"), "type": item.get("type"), "size": item.get("size")}
                            )

                    branch = client.rest_get(f"/repos/{owner}/{repo}/branches/{default_branch}")
                    tree_sha = (branch.get("commit") or {}).get("commit", {}).get("tree", {}).get("sha")
                    if tree_sha:
                        tree = client.rest_get(f"/repos/{owner}/{repo}/git/trees/{tree_sha}", {"recursive": "1"})
                        nodes = tree.get("tree") or []
                        files_total = sum(1 for n in nodes if n.get("type") == "blob")
                        dirs_total = sum(1 for n in nodes if n.get("type") == "tree")
                except Exception:
                    pass

            languages_top = []
            for e in ((r.get("languages") or {}).get("edges") or []):
                node = e.get("node") or {}
                languages_top.append({"language": node.get("name"), "size": e.get("size")})

            out[url] = {
                "owner": owner,
                "repo": repo,
                "name_with_owner": r.get("nameWithOwner"),
                "url": r.get("url"),
                "is_fork": r.get("isFork"),
                "parent_repo": (r.get("parent") or {}).get("nameWithOwner"),
                "parent_url": (r.get("parent") or {}).get("url"),
                "description": r.get("description"),
                "is_private": r.get("isPrivate"),
                "is_archived": r.get("isArchived"),
                "created_at": r.get("createdAt"),
                "updated_at": r.get("updatedAt"),
                "readme_title": readme_title,
                "stars": r.get("stargazerCount"),
                "forks": r.get("forkCount"),
                "watchers": (r.get("watchers") or {}).get("totalCount"),
                "primary_language": (r.get("primaryLanguage") or {}).get("name"),
                "languages_top": languages_top,
                "default_branch": default_branch,
                "commit_count_default_branch": commit_count,
                "first_commit_date_default_branch": first_commit_date,
                "last_commit_date_default_branch": last_commit_date,
                "first_commit_oid_default_branch": first_commit_oid,
                "last_commit_oid_default_branch": last_commit_oid,
                "pull_requests_total": (r.get("pullRequests") or {}).get("totalCount"),
                "pull_requests_open": (r.get("openPullRequests") or {}).get("totalCount"),
                "issues_total": (r.get("issues") or {}).get("totalCount"),
                "issues_open": (r.get("openIssues") or {}).get("totalCount"),
                "contributors_top": contributors,
                "files_root_entries": root_entries,
                "files_total_count": files_total,
                "dirs_total_count": dirs_total,
            }

        except Exception as e:
            out[url] = {"error": str(e), "owner": owner, "repo": repo}

    return out


if __name__ == "__main__":
    urls = ["https://github.com/sdsc-ordes/gimie"]
    token = os.getenv("GITHUB_TOKEN")
    print("token_present", bool(token), "token_len", len(token or ""))
    print(fetch_repo_metadata(urls, token=token, top_contributors=8))