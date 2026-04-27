"""Optional EPFL concept enrichment for repos.

Extracted from repo_analysis.ipynb cell 20.
Calls the EPFL Graph API to get semantic concepts for a repo's text.
Falls back gracefully if credentials are missing or the API is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 4000
DEFAULT_MAX_TRIES = 2
DEFAULT_DELAY_RETRY = 2


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _build_graph_api_json_from_env() -> Path | None:
    """Build a temporary credentials JSON from env vars."""
    host = os.getenv("EPFL_GRAPH_API_HOST")
    port = os.getenv("EPFL_GRAPH_API_PORT")
    user = os.getenv("EPFL_GRAPH_API_USER")
    password = os.getenv("EPFL_GRAPH_API_PASSWORD")

    if not all([host, port, user, password]):
        return None

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(
        {
            "host": host.rstrip("/"),
            "port": int(port),
            "user": user,
            "password": password,
        },
        tmp,
    )
    tmp.flush()
    tmp.close()
    return Path(tmp.name)


def _get_login_info() -> dict | None:
    """Attempt to authenticate with EPFL Graph API. Returns None on failure."""
    try:
        from graphai_client.client_api.utils import login
    except ImportError:
        log.info("graphai-client not installed — skipping concept enrichment")
        return None

    # Try env var pointing to JSON file first
    json_path = os.getenv("EPFL_GRAPH_API_JSON")
    if json_path and Path(json_path).exists():
        try:
            return login(json_path)
        except Exception as e:
            log.warning("EPFL Graph API login failed with JSON file: %s", e)
            return None

    # Try building from individual env vars
    tmp_path = _build_graph_api_json_from_env()
    if tmp_path is None:
        log.info("EPFL Graph API credentials not configured — skipping concept enrichment")
        return None

    try:
        return login(str(tmp_path))
    except Exception as e:
        log.warning("EPFL Graph API login failed: %s", e)
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Text building
# ---------------------------------------------------------------------------


def _coerce_string_list(value) -> list[str]:
    """Parse a value into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            value = json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return [stripped]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def build_repo_text(row: dict, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Build a text blob from repo metadata for concept extraction."""
    parts = [
        row.get("repo_name") or row.get("repo") or row.get("name_with_owner"),
        row.get("description"),
    ]

    topics = _coerce_string_list(row.get("topics") or row.get("repo_topics"))
    if topics:
        parts.append("Topics: " + ", ".join(topics))

    readme = row.get("readme_text")
    if isinstance(readme, str) and readme.strip():
        parts.append(readme.strip())

    text = "\n\n".join(
        str(p).strip() for p in parts if isinstance(p, str) and p.strip()
    )
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Concept extraction
# ---------------------------------------------------------------------------


def extract_concepts_for_repo(row: dict, login_info: dict) -> dict:
    """Extract concepts for a single repo row.

    Returns dict with:
    - repo_concepts (list[dict]): raw API response
    - repo_concept_names (list[str])
    - repo_top_concept (str | None)
    - repo_concepts_error (str | None)
    """
    from graphai_client.client_api.text import extract_concepts_from_text

    text = build_repo_text(row)
    if not text.strip():
        return {
            "repo_concepts": [],
            "repo_concept_names": [],
            "repo_top_concept": None,
            "repo_concepts_error": "empty_text",
        }

    try:
        concepts = extract_concepts_from_text(
            text,
            login_info,
            restrict_to_ontology=True,
            max_tries=DEFAULT_MAX_TRIES,
            delay_retry=DEFAULT_DELAY_RETRY,
        ) or []
    except Exception as e:
        log.warning("Concept extraction failed: %s", e)
        return {
            "repo_concepts": [],
            "repo_concept_names": [],
            "repo_top_concept": None,
            "repo_concepts_error": str(e),
        }

    concept_names = [
        c.get("concept_name") for c in concepts
        if isinstance(c, dict) and c.get("concept_name")
    ]

    return {
        "repo_concepts": concepts,
        "repo_concept_names": concept_names,
        "repo_top_concept": concept_names[0] if concept_names else None,
        "repo_concepts_error": None,
    }


# ---------------------------------------------------------------------------
# Public API: enrich a single repo (used by pipeline)
# ---------------------------------------------------------------------------


def try_enrich_with_concepts(row: dict) -> dict:
    """Attempt to enrich a repo row with EPFL concepts.

    Returns a dict with concept fields. If anything fails (no credentials,
    API down, import error), returns default values with empty concepts.
    This function NEVER raises — it always falls back gracefully.
    """
    defaults = {
        "repo_concepts": [],
        "repo_concept_names": [],
        "repo_top_concept": None,
        "repo_concepts_error": "not_attempted",
    }

    try:
        login_info = _get_login_info()
        if login_info is None:
            defaults["repo_concepts_error"] = "no_credentials"
            return defaults

        result = extract_concepts_for_repo(row, login_info)
        log.info("Extracted %d concepts for repo", len(result["repo_concept_names"]))
        return result

    except Exception as e:
        log.warning("Concept enrichment failed: %s", e)
        defaults["repo_concepts_error"] = str(e)
        return defaults
