"""Generate a static multi-panel feature distribution figure for the report.

Produces a 2x2 grid of box plots comparing hackathon vs non-hackathon
repositories for the four most discriminative features.

Usage:
    uv run python scripts/generate_feature_distribution_figure.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np

# --- Config ---
DATA_PATH = Path("data/repo_metadata_with_predictions.csv")
OUTPUT_PATH = Path("report/feature_distributions.png")

HACKATHON_COLOR = "#5a9e3a"
NON_HACKATHON_COLOR = "#4c72b0"

FEATURES = [
    ("active_days_default_branch", "Active Days\n(Default Branch)"),
    ("repo_age_days", "Repository Age\n(Days)"),
    ("commit_count_default_branch", "Commit Count\n(Default Branch)"),
    ("contributors_total", "Total\nContributors"),
]

# --- Load data ---
df = pd.read_csv(DATA_PATH)
valid = df[df["has_valid_metadata"] == True].copy()
hackathon = valid[valid["true_hackathon_repos"] == True]
non_hackathon = valid[valid["true_hackathon_repos"] == False]

# --- Plot ---
fig, axes = plt.subplots(1, 4, figsize=(14, 5))
fig.subplots_adjust(wspace=0.4)

for ax, (col, label) in zip(axes, FEATURES):
    hack_data = hackathon[col].dropna().values
    non_hack_data = non_hackathon[col].dropna().values

    bp = ax.boxplot(
        [hack_data, non_hack_data],
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color="black", linewidth=2),
    )

    bp["boxes"][0].set_facecolor(HACKATHON_COLOR)
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor(NON_HACKATHON_COLOR)
    bp["boxes"][1].set_alpha(0.7)

    # Overlay individual points with jitter
    for i, (data, color) in enumerate(
        [(hack_data, HACKATHON_COLOR), (non_hack_data, NON_HACKATHON_COLOR)], start=1
    ):
        jitter = np.random.uniform(-0.15, 0.15, size=len(data))
        ax.scatter(
            np.full(len(data), i) + jitter,
            data,
            alpha=0.25,
            s=8,
            color=color,
            zorder=3,
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Hackathon", "Non-\nHackathon"], fontsize=9)
    ax.set_ylabel(label, fontsize=9)
    ax.set_title(label, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)

# Legend
hack_patch = mpatches.Patch(color=HACKATHON_COLOR, alpha=0.7, label="Hackathon")
non_patch = mpatches.Patch(color=NON_HACKATHON_COLOR, alpha=0.7, label="Non-Hackathon")
fig.legend(
    handles=[hack_patch, non_patch],
    loc="upper center",
    ncol=2,
    bbox_to_anchor=(0.5, 1.02),
    fontsize=10,
    frameon=False,
)

fig.suptitle("", y=1.08)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved to {OUTPUT_PATH}")
plt.close()
