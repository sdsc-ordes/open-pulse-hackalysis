from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import requests
import logging



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


def fetch_repo_metadata(repo_urls: List[str], token: Optional[str] = None, top_contributors: int = 10, max_root_entries: int = 200,) -> Dict[str, Dict[str, Any]]:
    client = GitHubClient(token=token)
    out: Dict[str, Dict[str, Any]] = {}
    print(f"Fetching metadata for {len(repo_urls)} repositories from GitHub...")
    for url in repo_urls:
        owner, repo = parse_github_repo_url(url)

        try:
            data = client.graphql(REPO_QUERY, {"owner": owner, "name": repo})
            repo_data = data.get("repository")
            if not repo_data:
                out[url] = {"error": "repo not found or no access", "owner": owner, "repo": repo}
                continue

            default_branch = None
            commit_count = None
            first_commit_date = None
            last_commit_date = None
            first_commit_oid = None
            last_commit_oid = None

            dbr = repo_data.get("defaultBranchRef")
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

            readme_title = None
            try:
                readme = client.rest_get(f"/repos/{owner}/{repo}/readme")
                b64 = (readme or {}).get("content") or ""
                if b64:
                    md = base64.b64decode(b64).decode("utf-8", errors="replace")
                    readme_title = extract_readme_title(md)
            except Exception:
                pass

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
            for e in ((repo_data.get("languages") or {}).get("edges") or []):
                node = e.get("node") or {}
                languages_top.append({"language": node.get("name"), "size": e.get("size")})

            out[url] = {
                "owner": owner,
                "repo": repo,
                "name_with_owner": repo_data.get("nameWithOwner"),
                "url": repo_data.get("url"),
                "is_fork": repo_data.get("isFork"),
                "parent_repo": (repo_data.get("parent") or {}).get("nameWithOwner"),
                "parent_url": (repo_data.get("parent") or {}).get("url"),
                "description": repo_data.get("description"),
                "is_private": repo_data.get("isPrivate"),
                "is_archived": repo_data.get("isArchived"),
                "created_at": repo_data.get("createdAt"),
                "updated_at": repo_data.get("updatedAt"),
                "readme_title": readme_title,
                "stars": repo_data.get("stargazerCount"),
                "forks": repo_data.get("forkCount"),
                "watchers": (repo_data.get("watchers") or {}).get("totalCount"),
                "primary_language": (repo_data.get("primaryLanguage") or {}).get("name"),
                "languages_top": languages_top,
                "default_branch": default_branch,
                "commit_count_default_branch": commit_count,
                "first_commit_date_default_branch": first_commit_date,
                "last_commit_date_default_branch": last_commit_date,
                "first_commit_oid_default_branch": first_commit_oid,
                "last_commit_oid_default_branch": last_commit_oid,
                "pull_requests_total": (repo_data.get("pullRequests") or {}).get("totalCount"),
                "pull_requests_open": (repo_data.get("openPullRequests") or {}).get("totalCount"),
                "issues_total": (repo_data.get("issues") or {}).get("totalCount"),
                "issues_open": (repo_data.get("openIssues") or {}).get("totalCount"),
                "contributors_top": contributors,
                "files_root_entries": root_entries,
                "files_total_count": files_total,
                "dirs_total_count": dirs_total,
            }

        except Exception as e:
            out[url] = {"error": str(e), "owner": owner, "repo": repo}

    return out


def extract_github_urls_from_df(df: pd.DataFrame) -> List[str]:
    logging.basicConfig(level=logging.INFO)

    if "url" not in df.columns:
        logging.error("Column 'url' not found in dataframe")
        return []

    url_re = re.compile(
        r"https?://(?:www\.)?github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    )

    urls: set[str] = set()
    total_matches = 0

    logging.info(f"Scanning {len(df)} project rows from column 'url'")

    for idx, cell in df["url"].dropna().astype(str).items():
        matches = url_re.findall(cell)
        logging.info(f"row={idx} repo_urls_found={len(matches)} title='{df.loc[idx].get('title', '')[:50]}'")

        for owner, repo in matches:

            repo = repo.rstrip(").,;]}>#")
            if repo.lower().endswith(".git"):
                repo = repo[:-4]

            urls.add(f"https://github.com/{owner}/{repo}")
            total_matches += 1

    logging.info(f"Total repo URL matches found: {total_matches}")
    logging.info(f"Unique GitHub repos extracted: {len(urls)}")

    return sorted(urls)



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

    raw_json_path = hackathon_folder / f"{provider_prefix}_github_repo_metadata.json"
    parquet_path = hackathon_folder / f"{provider_prefix}_github_repo_metadata.parquet"

    raw_json_path.write_text(json.dumps(repo_meta, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for url, meta in repo_meta.items():
        row = {"input_url": url}
        if isinstance(meta, dict):
            row.update(meta)
        rows.append(row)

    df = pd.DataFrame(rows)

    for col in ["languages_top", "contributors_top", "files_root_entries"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x
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
    print("projects rows:", len(df), "cols:", list(df.columns)[:10])
    urls = extract_github_urls_from_df(df)
    print("github urls found:", len(urls))

    repo_meta = fetch_repo_metadata(
        urls,
        token=token,
        top_contributors=top_contributors,
        max_root_entries=max_root_entries,
    )

    out_paths = write_repo_metadata_outputs(
        hackathon_folder=hackathon_folder,
        provider_prefix=provider_prefix,
        repo_meta=repo_meta,
    )

    return {
        "projects_parquet": str(projects_parquet),
        "hackathon_folder": str(hackathon_folder),
        "provider_prefix": provider_prefix,
        "repos_found": len(urls),
        "outputs": {k: str(v) for k, v in out_paths.items()},
    }


if __name__ == "__main__":
    urls = ["https://github.com/sdsc-ordes/gimie"]
    token = os.getenv("GITHUB_TOKEN")
    print("token_present", bool(token), "token_len", len(token or ""))
    print("fetching metadata for:", urls)
    meta = fetch_repo_metadata(urls, token=token, top_contributors=8)
    print(json.dumps(meta, indent=2))