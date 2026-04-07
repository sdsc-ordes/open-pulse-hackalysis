"""Method 1: Naive Rule-Based hackathon repo detection.

Extracted from repo_analysis.ipynb cell 53.
Scores repos using keyword matches in text fields + GitHub stats thresholds.
No training data needed — purely deterministic rules.
"""

from __future__ import annotations

import json

import pandas as pd


# ---------------------------------------------------------------------------
# Keyword lists
# ---------------------------------------------------------------------------

STRONG_TERMS = [
    "hackathon",
    "lauzhack",
    "devpost",
    "mlh",
    "major league hacking",
    "hackerearth",
    "hackerone",
    "hackathon.com",
]

WEAK_TERMS = [
    "prototype",
    "weekend project",
    "submission",
    "challenge",
    "demo",
    "pitch",
    "built in 24 hours",
    "built in 48 hours",
    "built in 72 hours",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _topics_to_text(v) -> str:
    """Convert a topics field (list, JSON string, or plain string) to text."""
    if v is None:
        return ""
    if isinstance(v, list):
        return " ".join(str(x) for x in v if x is not None)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return ""
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return " ".join(str(x) for x in parsed if x is not None)
        except (json.JSONDecodeError, ValueError):
            pass
        return s
    return str(v)


def _get_num(v) -> float:
    """Safely extract a numeric value, defaulting to 0."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------


def predict_naive(df: pd.DataFrame) -> pd.DataFrame:
    """Apply rule-based scoring to each repo and return prediction columns.

    Scoring rules:
    - Text signals: +3 for strong hackathon terms, +1 for weak terms
    - Popularity: +1 if stars >= 1 or forks >= 1
    - Contributors: +1 if >= 2 contributors
    - Commits: +1 if >= 5 commits
    - Burst window: +2 if active_days <= 3, +1 if <= 7
    - Metadata error with no other signal: -1

    Threshold: score >= 4 → hackathon (True)

    Adds columns:
    - repo_predicted_score (int)
    - repo_predicted_flag (bool)
    - repo_predicted_signals (str)
    """
    scores = []
    flags = []
    signals = []

    for _, row in df.iterrows():
        text_blob = " ".join([
            str(row.get("repo_url", "") or ""),
            str(row.get("repo_name", "") or ""),
            str(row.get("primary_language", "") or ""),
            str(row.get("readme_title", "") or ""),
            str(row.get("readme_text", "") or ""),
            _topics_to_text(row.get("topics") or row.get("repo_topics")),
        ]).lower()

        score = 0
        row_signals = []

        # Text matching
        if any(t in text_blob for t in STRONG_TERMS):
            score += 3
            row_signals.append("strong_repo_text")
        elif any(t in text_blob for t in WEAK_TERMS):
            score += 1
            row_signals.append("weak_repo_text")

        # GitHub stats
        if _get_num(row.get("stars")) >= 1 or _get_num(row.get("forks")) >= 1:
            score += 1
            row_signals.append("repo_popularity_signal")

        if _get_num(row.get("contributors_count")) >= 2:
            score += 1
            row_signals.append("contributors_signal")

        if _get_num(row.get("commit_count_default_branch")) >= 5:
            score += 1
            row_signals.append("commit_activity_signal")

        # Burst commit window
        active_days = _get_num(row.get("active_days_default_branch"))
        if 0 < active_days <= 3:
            score += 2
            row_signals.append("burst_commit_window_strong")
        elif 0 < active_days <= 7:
            score += 1
            row_signals.append("burst_commit_window_weak")

        # Metadata error penalty (check both column name variants)
        error_val = row.get("error") or row.get("repo_error")
        has_error = pd.notna(error_val) and error_val != ""
        if has_error and score == 0:
            score -= 1
            row_signals.append("metadata_error")

        scores.append(score)
        flags.append(score >= 4)
        signals.append("|".join(row_signals) if row_signals else "no_signal")

    out = df.copy()
    out["repo_predicted_score"] = scores
    out["repo_predicted_flag"] = flags
    out["repo_predicted_signals"] = signals
    return out
