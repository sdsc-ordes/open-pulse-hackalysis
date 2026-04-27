"""Generate a static correlation bar chart for the report.

Shows Pearson correlation of each feature with the hackathon label,
sorted by value, with positive/negative bars colored differently.

Usage:
    uv run python scripts/generate_correlation_figure.py
"""

from pathlib import Path

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# --- Config ---
WEIGHTS_PATH = Path("src/hackathon_analysis/models/correlation_weights.json")
OUTPUT_PATH = Path("report/static/correlations.png")

POSITIVE_COLOR = "#5a9e3a"   # green — hackathon enriched
NEGATIVE_COLOR = "#4c72b0"   # blue — non-hackathon enriched

# Readable feature name mapping
FEATURE_LABELS = {
    "active_days_default_branch": "Active Days (Default Branch)",
    "commit_count_default_branch": "Commit Count (Default Branch)",
    "commit_log": "Commit Count (log)",
    "contributors_count_log": "Contributors Count (log)",
    "contributors_total": "Contributors Total",
    "files_total_count": "Files Total Count",
    "forks": "Forks",
    "has_ci": "Has CI",
    "has_contributing": "Has Contributing Guide",
    "has_docker": "Has Docker",
    "has_docs": "Has Docs",
    "has_license_file": "Has License File",
    "has_notebooks": "Has Notebooks",
    "has_readme_file": "Has README",
    "has_tests": "Has Tests",
    "is_archived": "Is Archived",
    "is_fork": "Is Fork",
    "issues_open": "Issues Open",
    "issues_total": "Issues Total",
    "n_concepts": "N Concepts",
    "pull_requests_closed": "Pull Requests Closed",
    "pull_requests_open": "Pull Requests Open",
    "pull_requests_total": "Pull Requests Total",
    "readme_length": "README Length",
    "releases_count": "Releases Count",
    "repo_age_days": "Repository Age (Days)",
    "stars": "Stars",
    "watchers": "Watchers",
}

# --- Load weights ---
with open(WEIGHTS_PATH) as f:
    data = json.load(f)

weights = data["weights"]

# Sort by value (ascending — most negative at top)
sorted_items = sorted(weights.items(), key=lambda x: x[1])
features = [FEATURE_LABELS.get(k, k) for k, _ in sorted_items]
values = [v for _, v in sorted_items]
colors = [POSITIVE_COLOR if v >= 0 else NEGATIVE_COLOR for v in values]

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 10))

bars = ax.barh(features, values, color=colors, edgecolor="white", height=0.7)

# Value labels
for bar, val in zip(bars, values):
    x_pos = val + 0.005 if val >= 0 else val - 0.005
    ha = "left" if val >= 0 else "right"
    ax.text(
        x_pos,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}",
        va="center",
        ha=ha,
        fontsize=7,
        color="#333333",
    )

# Zero line
ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

# Styling
ax.set_xlabel("Pearson Correlation with Hackathon Label", fontsize=11)
ax.set_xlim(min(values) - 0.07, max(values) + 0.07)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=9)
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="x", linestyle="--", alpha=0.4)

# Legend
neg_patch = mpatches.Patch(color=NEGATIVE_COLOR, label="Negatively correlated (non-hackathon signal)")
pos_patch = mpatches.Patch(color=POSITIVE_COLOR, label="Positively correlated (hackathon signal)")
ax.legend(handles=[neg_patch, pos_patch], loc="lower right", fontsize=9, frameon=False)

plt.tight_layout()
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
plt.close()
