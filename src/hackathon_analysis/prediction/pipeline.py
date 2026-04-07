"""Main prediction pipeline: GitHub URL → hackathon classification.

This is the entry point for production use. Give it a GitHub repo URL,
it fetches metadata via the GitHub API, computes features, and runs
all three prediction methods.

Usage:
    from hackathon_analysis.prediction.pipeline import predict_repo
    result = predict_repo("https://github.com/someone/some-repo")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from hackathon_analysis.data_extraction.github_extractor import (
    fetch_repo_metadata,
    parse_github_repo_url,
)
from hackathon_analysis.prediction.correlation_weighted import (
    CorrelationWeights,
    predict_correlation_weighted,
)
from hackathon_analysis.prediction.feature_engineering import (
    build_repo_analysis_features,
)
from hackathon_analysis.prediction.naive_rules import predict_naive
from hackathon_analysis.prediction.random_forest import (
    TrainedRandomForest,
    predict_random_forest,
)

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


# ---------------------------------------------------------------------------
# Load saved model artifacts (cached at module level)
# ---------------------------------------------------------------------------

_cached_corr_weights: CorrelationWeights | None = None
_cached_rf_model: TrainedRandomForest | None = None


def _load_correlation_weights(models_dir: Path = MODELS_DIR) -> CorrelationWeights:
    global _cached_corr_weights
    if _cached_corr_weights is None:
        path = models_dir / "correlation_weights.json"
        with open(path) as f:
            _cached_corr_weights = CorrelationWeights.from_dict(json.load(f))
    return _cached_corr_weights


def _load_rf_model(models_dir: Path = MODELS_DIR) -> TrainedRandomForest:
    global _cached_rf_model
    if _cached_rf_model is None:
        _cached_rf_model = TrainedRandomForest.load(models_dir / "rf_model.pkl")
    return _cached_rf_model


# ---------------------------------------------------------------------------
# GitHub metadata fetching
# ---------------------------------------------------------------------------


def fetch_single_repo_metadata(
    repo_url: str,
    github_token: str | None = None,
) -> pd.DataFrame:
    """Fetch GitHub metadata for a single repo and return as a 1-row DataFrame.

    Uses the existing github_extractor module which handles GraphQL queries,
    rate limiting, and retry logic.
    """
    if github_token is None:
        load_dotenv()
        github_token = os.getenv("GITHUB_TOKEN")

    owner, repo = parse_github_repo_url(repo_url)
    log.info("Fetching metadata for %s/%s ...", owner, repo)

    raw = fetch_repo_metadata(
        repo_urls=[repo_url],
        token=github_token,
        include_readme_text=True,
        fetch_readme=True,
    )

    if repo_url not in raw:
        raise RuntimeError(f"GitHub API returned no data for {repo_url}")

    metadata = raw[repo_url]

    # Check if it's an error response
    if "error" in metadata and "owner" in metadata and len(metadata) <= 3:
        log.warning("GitHub API error for %s: %s", repo_url, metadata.get("error"))

    # Flatten to a 1-row DataFrame
    df = pd.json_normalize(metadata)
    df.insert(0, "repo_url", repo_url)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def predict_repo(
    repo_url: str,
    github_token: str | None = None,
    models_dir: Path = MODELS_DIR,
    enrich_concepts: bool = True,
) -> dict:
    """Classify a GitHub repo as hackathon or not using all three methods.

    Args:
        repo_url: full GitHub repo URL (e.g. "https://github.com/user/repo")
        github_token: GitHub API token. If None, reads from GITHUB_TOKEN env var.
        models_dir: directory containing saved model artifacts.
        enrich_concepts: if True, attempt EPFL concept enrichment (falls back
            gracefully if credentials are missing or API is unavailable).

    Returns:
        dict with predictions from all three methods + repo metadata.
    """
    from hackathon_analysis.prediction.concepts import try_enrich_with_concepts

    # Step 1: Fetch metadata from GitHub API
    raw_df = fetch_single_repo_metadata(repo_url, github_token)

    # Step 1b: Optional concept enrichment
    if enrich_concepts:
        row_dict = raw_df.iloc[0].to_dict()
        concept_result = try_enrich_with_concepts(row_dict)
        raw_df["n_concepts"] = concept_result["n_concepts"]
        raw_df["repo_concept_names"] = [concept_result["repo_concept_names"]]
        raw_df["repo_top_concept"] = concept_result["repo_top_concept"]
        raw_df["repo_concepts_error"] = concept_result["repo_concepts_error"]

    # Step 2: Compute derived features
    df = build_repo_analysis_features(raw_df)

    # Step 3: Method 1 — Naive rules (no model needed)
    df = predict_naive(df)

    # Step 4: Method 2 — Correlation-weighted (needs saved weights)
    corr_weights = _load_correlation_weights(models_dir)
    df = predict_correlation_weighted(df, corr_weights)

    # Step 5: Method 3 — Random Forest (needs saved model)
    rf_model = _load_rf_model(models_dir)
    df = predict_random_forest(df, rf_model)

    # Step 6: Build result dict
    row = df.iloc[0]
    result = {
        "repo_url": repo_url,
        "naive_rule_based": {
            "flag": bool(row.get("repo_predicted_flag", False)),
            "score": int(row.get("repo_predicted_score", 0)),
            "signals": str(row.get("repo_predicted_signals", "")),
        },
        "correlation_weighted": {
            "flag": bool(row.get("corr_weighted_flag", False)),
            "score": round(float(row.get("corr_weighted_score", 0)), 4),
        },
        "random_forest": {
            "flag": bool(row.get("ml_predicted_flag", 0)),
            "probability": round(float(row.get("ml_predicted_proba", 0)), 4),
        },
        "metadata": {
            "owner": row.get("owner", ""),
            "repo": row.get("repo", ""),
            "language": row.get("primary_language", ""),
            "stars": int(row.get("stars", 0)),
            "forks": int(row.get("forks", 0)),
            "commits": int(row.get("commit_count_default_branch", 0)),
            "contributors": int(row.get("contributors_count", 0)),
            "active_days": int(row.get("active_days_default_branch", 0) or 0),
            "repo_age_days": int(row.get("repo_age_days", 0) or 0),
            "has_valid_metadata": bool(row.get("has_valid_metadata", False)),
        },
    }

    return result


def predict_repo_pretty(repo_url: str, **kwargs) -> str:
    """Run predict_repo and return a human-readable summary string."""
    r = predict_repo(repo_url, **kwargs)
    m = r["metadata"]

    lines = [
        f"Repository: {r['repo_url']}",
        f"  Language: {m['language']}  |  Stars: {m['stars']}  |  "
        f"Commits: {m['commits']}  |  Active days: {m['active_days']}",
        "",
        "Predictions:",
        f"  1. Naive Rules:         {'HACKATHON' if r['naive_rule_based']['flag'] else 'NOT hackathon'}"
        f"  (score: {r['naive_rule_based']['score']}, signals: {r['naive_rule_based']['signals']})",
        f"  2. Correlation-Weighted: {'HACKATHON' if r['correlation_weighted']['flag'] else 'NOT hackathon'}"
        f"  (score: {r['correlation_weighted']['score']:.2f})",
        f"  3. Random Forest:       {'HACKATHON' if r['random_forest']['flag'] else 'NOT hackathon'}"
        f"  (probability: {r['random_forest']['probability']:.2f})",
    ]

    # Consensus
    votes = sum([
        r["naive_rule_based"]["flag"],
        r["correlation_weighted"]["flag"],
        r["random_forest"]["flag"],
    ])
    if votes >= 2:
        lines.append(f"\n  Consensus: HACKATHON REPO ({votes}/3 methods agree)")
    else:
        lines.append(f"\n  Consensus: NOT a hackathon repo ({votes}/3 methods agree)")

    return "\n".join(lines)
