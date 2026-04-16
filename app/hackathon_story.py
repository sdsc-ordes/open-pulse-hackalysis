"""
Hackalysis: The Story of Detecting Hackathon Repositories
=========================================================
A data-driven narrative exploring how to distinguish hackathon repos
from regular GitHub repositories, using 487 repos (213 from 3 years of LauzHack
plus 274 personal repos from participants' accounts as non-hackathon examples).

Author: Eisha Tir Raazia
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import streamlit as st

from hackathon_analysis.common_utils import load_huggingface_dataset
from hackathon_analysis.data_extraction.dataset_resolver import get_hf_repo_from_env

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Color palette — SDSC brand: lime green, indigo blue, dark navy
SDSC_GREEN = "#90ca42"     # Primary brand — lime green
SDSC_BLUE = "#5561a6"      # Primary brand — indigo blue
SDSC_NAVY = "#26235c"      # Primary brand — dark purple/navy

# Extended palette derived from SDSC brand
GREEN_LIGHT = "#d4edaa"    # Light tint of brand green
BLUE_LIGHT = "#c8cde3"     # Light tint of brand blue
NAVY_LIGHT = "#e8e7f0"     # Light tint of brand navy

ROSE = "#dc2626"           # Alert / negative emphasis
SLATE = "#334155"          # Body text
SLATE_LIGHT = "#64748b"    # Muted text

# Data colors — SDSC green = hackathon (positive), SDSC blue = non-hackathon
HACKATHON_COLOR = SDSC_GREEN
NON_HACKATHON_COLOR = SDSC_BLUE
METHOD_COLORS = {
    "Naive Rule-Based": "#e8a838",    # Warm amber
    "Correlation-Weighted": SDSC_BLUE,  # Brand blue
    "Random Forest": SDSC_GREEN,        # Brand green
}

# Feature metadata for dashboard display
FEATURE_METADATA = {
    "Days with Code Changes": {
        "definition": "How many days elapsed between the first and last commit. Hackathon projects typically have short, intense development windows.",
        "is_derived": True,
        "calculation": "Calculated as: (last_commit_date − first_commit_date)",
    },
    "Repository Age (Days)": {
        "definition": "How many days the repository has existed. Shows whether a project is brand new or has been around for a while.",
        "is_derived": True,
        "calculation": "Calculated as: (pushed_at − created_at)",
    },
    "README Length (characters)": {
        "definition": "Size of the repository's README file. Hackathon repos often have shorter, quick-reference READMEs.",
        "is_derived": False,
        "calculation": None,
    },
    "Number of Commits": {
        "definition": "Total commits on the main branch. Indicates how much development work went into the project.",
        "is_derived": False,
        "calculation": None,
    },
    "Repository Stars": {
        "definition": "Number of GitHub stars (likes) a repository received. Popular projects attract more stars.",
        "is_derived": False,
        "calculation": None,
    },
    "Number of Forks": {
        "definition": "How many times the repository was forked. Shows how many people wanted to build on this project.",
        "is_derived": False,
        "calculation": None,
    },
    "Number of Contributors": {
        "definition": "How many different people contributed code. Hackathons typically involve teams of 2–5 people.",
        "is_derived": False,
        "calculation": None,
    },
    "Total Issues": {
        "definition": "Number of GitHub issues (bug reports, feature requests, discussions) ever opened.",
        "is_derived": False,
        "calculation": None,
    },
    "Total Pull Requests": {
        "definition": "Number of pull requests submitted. Shows collaboration and code review activity.",
        "is_derived": False,
        "calculation": None,
    },
    "Total Files": {
        "definition": "Count of all files in the repository. Larger projects have more files.",
        "is_derived": False,
        "calculation": None,
    },
    "Total Directories": {
        "definition": "Count of folders/directories. Complex projects organize code into more directories.",
        "is_derived": False,
        "calculation": None,
    },
    "Watchers": {
        "definition": "Number of people watching the repository for updates. Similar to stars, shows interest level.",
        "is_derived": False,
        "calculation": None,
    },
}

st.set_page_config(
    page_title="Hackalysis | Hackathon Repo Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* ---- Global spacing tightening ---- */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }
    /* Reduce gaps between elements */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.4rem;
    }
    /* Tighter heading margins */
    h1, h2, h3 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.2rem !important;
    }
    /* Reduce gap after subheader */
    .stMarkdown h2, .stMarkdown h3 {
        padding-top: 0 !important;
    }
    /* Reduce plotly chart top/bottom margins */
    .stPlotlyChart {
        margin-top: -0.5rem;
        margin-bottom: -0.5rem;
    }

    /* ---- Header accent bar ---- */
    .stApp > header {
        background: linear-gradient(90deg, #26235c 0%, #5561a6 40%, #90ca42 100%);
        height: 3px;
    }

    /* ---- Metric card styling ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f3f7ea 0%, #eef0f8 100%);
        border: 1px solid #d4edaa;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 1px 3px rgba(38,35,92,0.06);
    }
    div[data-testid="stMetric"] label {
        color: #475569;
        font-weight: 600;
        font-size: 0.82rem;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #26235c;
    }

    /* ---- Section dividers ---- */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #c8cde3, transparent);
        margin: 1.2rem 0;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f3f7ea 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.95rem;
    }

    /* ---- Tab styling ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #26235c;
        color: white;
    }

    /* ---- Expander styling ---- */
    details[data-testid="stExpander"] {
        border: 1px solid #d4edaa;
        border-radius: 10px;
        background: #fafcf5;
    }

    /* ---- Caption / small text ---- */
    .stCaption, div[data-testid="stCaptionContainer"] {
        margin-top: -0.3rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* ---- Alert boxes tighter ---- */
    div[data-testid="stAlert"] {
        padding: 12px 16px;
        margin-bottom: 0.4rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data Loading (cached)
# ---------------------------------------------------------------------------


@st.cache_data
def load_project_metadata() -> dict:
    """Load project metadata for all years to map project IDs to years."""
    project_year_map = {}
    for year in [2023, 2024, 2025]:
        year_dir = DATA_ROOT / f"lauzhack-{year}"
        projects_file = year_dir / "lauzhack_projects.parquet"
        if projects_file.exists():
            proj_df = pd.read_parquet(projects_file)
            for _, row in proj_df.iterrows():
                project_year_map[row["project_hard_id"]] = str(row["hackathon_year"])
    return project_year_map


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load predictions dataset from Hugging Face Hub (with local fallback for testing)."""
    local_csv = DATA_ROOT / "repo_metadata_with_predictions.csv"

    # Local-first fallback: use local file if it exists (useful during testing)
    if local_csv.exists():
        import logging
        logging.info("Loading predictions from local file (fallback): %s", local_csv)
        df = pd.read_csv(local_csv)
    else:
        # Primary: load from Hugging Face Hub
        try:
            import logging
            logging.info("Loading predictions from Hugging Face Hub...")
            repo = get_hf_repo_from_env()
            hf_dataset = load_huggingface_dataset(
                repo_id=repo.repo_id,
                split="train",
            )
            df = hf_dataset.to_pandas()
            logging.info("Successfully loaded %d repos from HF Hub", len(df))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load predictions from Hugging Face Hub: {e}\n"
                f"HF_REPO_ID: {repo.repo_id}\n"
                f"Make sure the dataset exists on HF Hub and HF_TOKEN is set if private."
            ) from e

    # Parse stringified lists
    for col in ["concept_list", "repo_concept_names", "concept_project_freq_buckets", "topics"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    # Load project metadata for mapping deleted repos to years
    project_year_map = load_project_metadata()

    # Derive source from the data itself (no external JSON file lookup needed)
    # Hackathon repos (have a project_foreign_key) → use created_at year or project metadata
    # Account repos (no project link) → group by owner
    created_year = pd.to_datetime(df["created_at"], errors="coerce").dt.year
    has_project = df["project_foreign_key"].fillna("").astype(str).str.strip().ne("")

    df["year"] = "Unknown Source"

    # For hackathon repos: use created_at year if available, otherwise look up from project metadata
    for idx in df[has_project].index:
        year_val = created_year[idx]
        if pd.notna(year_val):
            df.loc[idx, "year"] = str(int(year_val))
        else:
            # Try to get year from project metadata
            proj_fk = df.loc[idx, "project_foreign_key"]
            if pd.notna(proj_fk) and str(proj_fk).strip() in project_year_map:
                df.loc[idx, "year"] = project_year_map[str(proj_fk).strip()]
            else:
                df.loc[idx, "year"] = "Unknown Project Year"

    # For non-hackathon repos: group by owner
    df.loc[~has_project, "year"] = df.loc[~has_project, "owner"].apply(
        lambda o: f"Account: {o}" if pd.notna(o) and str(o).strip() else "Unknown Source"
    )
    df["is_hackathon_year"] = df["year"].str.match(r"^\d{4}$")

    # Prediction agreement
    naive_pred = df["repo_predicted_flag"].astype(int)
    corr_pred = df["corr_weighted_flag"].astype(int)
    ml_pred = df["ml_predicted_flag"].astype(int)
    df["prediction_agreement"] = naive_pred + corr_pred + ml_pred

    return df


def _safe_parse_list(val):
    if pd.isna(val) or val == "" or val == "nan":
        return []
    if isinstance(val, list):
        return val
    try:
        result = ast.literal_eval(str(val))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


@st.cache_data
def compute_fisher_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Fisher's exact test for each concept vs hackathon label."""
    valid = df[df["has_valid_metadata"]].copy()
    hack_mask = valid["true_hackathon_repos"].astype(bool)
    n_hack = hack_mask.sum()
    n_non = (~hack_mask).sum()

    concept_counts: dict[str, dict] = {}
    for _, row in valid.iterrows():
        concepts = row.get("concept_list", [])
        if not isinstance(concepts, list):
            continue
        is_hack = bool(row["true_hackathon_repos"])
        for c in concepts:
            if c not in concept_counts:
                concept_counts[c] = {"hackathon": 0, "non_hackathon": 0}
            if is_hack:
                concept_counts[c]["hackathon"] += 1
            else:
                concept_counts[c]["non_hackathon"] += 1

    rows = []
    for concept, counts in concept_counts.items():
        a = counts["hackathon"]
        b = counts["non_hackathon"]
        c = n_hack - a
        d = n_non - b
        if (a + b) < 2:
            continue
        table = np.array([[a, b], [c, d]])
        odds_ratio, p_value = fisher_exact(table)
        log2_or = np.log2(odds_ratio) if odds_ratio > 0 else 0
        lift = (a / max(n_hack, 1)) / (max((a + b), 1) / max(n_hack + n_non, 1))
        importance = -np.log10(max(p_value, 1e-300)) * np.sign(log2_or)
        rows.append(
            {
                "concept": concept,
                "hackathon_repos": a,
                "non_hackathon_repos": b,
                "total_repos": a + b,
                "odds_ratio": odds_ratio,
                "log2_odds_ratio": log2_or,
                "fisher_p": p_value,
                "lift": lift,
                "importance": importance,
            }
        )
    result = pd.DataFrame(rows)

    # Apply Benjamini-Hochberg FDR correction for multiple comparisons
    if len(result) > 0:
        reject, p_adjusted, _, _ = multipletests(result["fisher_p"], method="fdr_bh", alpha=0.05)
        result["fisher_p_adjusted"] = p_adjusted
        result["significant_fdr"] = reject
        # Recompute importance using corrected p-values
        result["importance"] = -np.log10(result["fisher_p_adjusted"].clip(lower=1e-300)) * np.sign(result["log2_odds_ratio"])

    return result.sort_values("importance", key=abs, ascending=False)


@st.cache_data
def compute_rf_importances(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Refit Random Forest to extract feature importances."""
    valid = df[df["has_valid_metadata"]].copy()
    target = valid["true_hackathon_repos"].astype(int)

    # Same feature selection as notebook: exclude leakers, concept features, metadata artifacts
    exclude = {
        "true_hackathon_repos",
        "repo_url",
        "url",
        "repo",
        "repo_name",
        "name_with_owner",
        "owner",
        "description",
        "readme_text",
        "readme_title",
        "default_branch",
        "homepage_url",
        "topics",
        "contributors_top",
        "languages_top",
        "files_root_entries",
        "created_at",
        "updated_at",
        "pushed_at",
        "first_commit_date_default_branch",
        "last_commit_date_default_branch",
        "last_commit_oid_default_branch",
        "latest_release_date",
        "latest_release_tag",
        "license_name",
        "license_spdx",
        "parent_repo",
        "parent_url",
        "project_foreign_key",
        "error",
        "repo_concepts",
        "repo_concept_names",
        "repo_top_concept",
        "repo_concepts_error",
        "concept_project_freq_buckets",
        "concept_list",
        "repo_predicted_signals",
        "year",
        "prediction_agreement",
        # Label leakers
        "is_linked_project",
        "repo_predicted_score",
        "repo_predicted_flag",
        "corr_weighted_score",
        "corr_weighted_flag",
        "ml_predicted_flag",
        "ml_predicted_proba",
        # Biased concept features
        "has_top_hackathon_concept",
        "concept_hackathon_score",
        "min_concept_projects",
        "max_concept_projects",
    }
    # Also exclude one-hot concept columns
    concept_onehot = [c for c in valid.columns if c.startswith("concept_is_")]
    exclude.update(concept_onehot)

    feature_cols = [
        c
        for c in valid.columns
        if c not in exclude and valid[c].dtype in [np.float64, np.int64, "bool", np.bool_]
    ]

    X = valid[feature_cols].astype(float).fillna(0)

    # Remove constant columns
    feature_cols = [c for c in feature_cols if X[c].nunique() > 1]
    X = X[feature_cols]

    # Remove multicollinear (>0.85)
    corr_matrix = X.corrwith(target).abs()
    pair_corr = X.corr().abs()
    to_drop = set()
    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            if pair_corr.iloc[i, j] > 0.85:
                ci, cj = feature_cols[i], feature_cols[j]
                drop = ci if corr_matrix[ci] < corr_matrix[cj] else cj
                to_drop.add(drop)
    feature_cols = [c for c in feature_cols if c not in to_drop]
    X = X[feature_cols]

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X, target)

    importance_df = (
        pd.DataFrame({"feature": feature_cols, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, feature_cols


# ---------------------------------------------------------------------------
# Helper: Concept Network Graph
# ---------------------------------------------------------------------------


@st.cache_data
def compute_concept_network(df: pd.DataFrame, min_similarity: float = 0.15, max_concept_freq: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a network of repos connected by shared *rare* concepts using Jaccard similarity.

    Generic concepts (appearing in >max_concept_freq of repos) are excluded so connections
    reflect genuinely shared domain knowledge, not "both use Python."

    Returns node and edge DataFrames for plotting.
    """
    from itertools import combinations

    has_concepts = df[df["concept_list"].apply(lambda x: isinstance(x, list) and len(x) > 0)].copy()
    n_repos = len(has_concepts)

    # Compute concept document frequency and filter out common ones
    concept_freq: dict[str, int] = {}
    for _, row in has_concepts.iterrows():
        for c in row["concept_list"]:
            concept_freq[c] = concept_freq.get(c, 0) + 1

    freq_cutoff = int(n_repos * max_concept_freq)
    rare_concepts = {c for c, n in concept_freq.items() if 2 <= n <= freq_cutoff}

    # Build per-repo rare concept sets
    repo_rare: dict[int, set[str]] = {}
    for idx, row in has_concepts.iterrows():
        rare = {c for c in row["concept_list"] if c in rare_concepts}
        if len(rare) >= 2:
            repo_rare[idx] = rare

    # Build inverted index for efficiency (only iterate co-occurring pairs)
    concept_to_idxs: dict[str, set[int]] = {}
    for idx, concepts in repo_rare.items():
        for c in concepts:
            concept_to_idxs.setdefault(c, set()).add(idx)

    # Compute Jaccard similarity for co-occurring pairs
    pair_shared: dict[tuple[int, int], set[str]] = {}
    for concept, idxs in concept_to_idxs.items():
        if len(idxs) > 40:
            continue
        idx_list = sorted(idxs)
        for a in range(len(idx_list)):
            for b in range(a + 1, len(idx_list)):
                key = (idx_list[a], idx_list[b])
                if key not in pair_shared:
                    pair_shared[key] = set()
                pair_shared[key].add(concept)

    edges = []
    for (i, j), shared in pair_shared.items():
        if len(shared) < 5:  # require at least 5 rare concepts in common to avoid noise
            continue
        union_size = len(repo_rare[i] | repo_rare[j])
        jaccard = len(shared) / union_size if union_size > 0 else 0
        if jaccard >= min_similarity:
            # Pick top-5 rarest shared concepts as examples
            examples = sorted(shared, key=lambda c: concept_freq.get(c, 999))[:5]
            edges.append({
                "source": i,
                "target": j,
                "jaccard": round(jaccard, 3),
                "shared_count": len(shared),
                "example_concepts": ", ".join(examples),
            })

    edge_df = pd.DataFrame(edges) if edges else pd.DataFrame(
        columns=["source", "target", "jaccard", "shared_count", "example_concepts"]
    )

    # Build nodes
    connected_idxs = set()
    if len(edge_df) > 0:
        connected_idxs = set(edge_df["source"]) | set(edge_df["target"])

    node_rows = []
    for idx in connected_idxs:
        row = has_concepts.loc[idx]
        desc = row.get("description", "")
        node_rows.append({
            "idx": idx,
            "name": row.get("name_with_owner", str(idx)),
            "is_hackathon": bool(row.get("true_hackathon_repos", False)),
            "n_concepts": len(row["concept_list"]) if isinstance(row["concept_list"], list) else 0,
            "n_rare_concepts": len(repo_rare.get(idx, set())),
            "primary_language": row.get("primary_language", "Unknown"),
            "description": str(desc) if pd.notna(desc) else "",
        })
    node_df = pd.DataFrame(node_rows) if node_rows else pd.DataFrame(
        columns=["idx", "name", "is_hackathon", "n_concepts", "n_rare_concepts", "primary_language", "description"]
    )

    return node_df, edge_df


# ---------------------------------------------------------------------------
# Helper: Confusion Matrix Plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(y_true, y_pred, title="", color_scale="Teal"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Not Hackathon", "Hackathon"],
        y=["Not Hackathon", "Hackathon"],
        color_continuous_scale=color_scale,
    )
    fig.update_layout(title=title, width=420, height=400, coloraxis_showscale=False)
    fig.update_traces(textfont_size=18)
    return fig


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }


# ---------------------------------------------------------------------------
# Act 1: The Question
# ---------------------------------------------------------------------------


def render_introduction(df: pd.DataFrame):
    st.markdown(
        """
    <h1 style='text-align:center; color:#26235c; font-size:2.8rem;'>
        Can You Tell a Hackathon Repo from a Regular One?
    </h1>
    <p style='text-align:center; color:#64748b; font-size:1.15rem; max-width:800px; margin:auto;'>
        Not all repositories linked to hackathon project pages are actual hackathon code.
        Some are pre-existing libraries, framework forks, or personal utilities.
        We built a general-purpose pipeline to identify <b>true hackathon repos</b> &mdash;
        and validated it on <b>LauzHack</b> (2023&ndash;2025) as our case study.
    </p>
    <p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:8px;'>
        The methodology is hackathon-agnostic; LauzHack-specific details are clearly marked throughout.
        Navigate the story using the sidebar.
    </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Metric cards
    valid = df[df["has_valid_metadata"]]
    n_projects = df["project_foreign_key"].dropna().nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GitHub Repos", f"{len(df)}", help=f"{len(df)} total linked repositories, {len(valid)} with complete GitHub metadata")
    data_quality = (len(valid) / len(df) * 100) if len(df) > 0 else 0
    c2.metric("Data Completeness", f"{len(valid)}", help=f"{data_quality:.1f}% of repositories have valid GitHub metadata")
    c3.metric("Unique Lauzhack Projects", f"{n_projects}", help="Unique hackathon projects (one project can link to multiple repos)")
    unique_concepts = df["concept_list"].explode().nunique()
    c4.metric(
        "Topic Tags",
        f"{unique_concepts:,}",
        help="Topics (like 'machine learning' or 'web development') automatically assigned to each "
        "repository by a knowledge database. Think of them as smart tags describing what a project is about.",
    )

    st.markdown("---")

    # Data quality + label breakdown — two separate concerns
    total = len(df)
    with_metadata = int(df["has_valid_metadata"].sum())
    linked = int(df["is_linked_project"].sum())
    true_hack = int(df["true_hackathon_repos"].sum())

    dq_col, label_col = st.columns(2)

    with dq_col:
        st.subheader("Data Pipeline")
        st.caption(
            "From raw collection to usable data. Some repos were deleted or private "
            "and couldn't be fetched. Of those with valid metadata, a subset were "
            "confirmed as actual hackathon code."
        )

        true_hack_with_metadata = int(
            (df["true_hackathon_repos"] & df["has_valid_metadata"]).sum()
        )
        pipeline_data = pd.DataFrame(
            {
                "Stage": [
                    "All Repositories Collected",
                    "Have Valid GitHub Metadata",
                    "Confirmed Hackathon (with metadata)",
                ],
                "Count": [total, with_metadata, true_hack_with_metadata],
            }
        )

        fig_pipeline = px.funnel(
            pipeline_data,
            x="Count",
            y="Stage",
            color_discrete_sequence=[SDSC_BLUE],
        )
        fig_pipeline.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
        )
        fig_pipeline.update_traces(textinfo="value+percent initial", textfont_size=14)
        st.plotly_chart(fig_pipeline, use_container_width=True)

    with label_col:
        st.subheader("Ground Truth Labels")
        st.caption(
            "How repos were labeled — independent of data quality. "
            "Labels come from project page links + manual review."
        )

        hack_and_linked = int(((df["true_hackathon_repos"]) & (df["is_linked_project"])).sum())
        hack_not_linked = true_hack - hack_and_linked

        label_data = pd.DataFrame(
            {
                "Category": [
                    "Non-Hackathon Repos",
                    "Hackathon (linked to project page)",
                    "Hackathon (manually confirmed)",
                ],
                "Count": [total - true_hack, hack_and_linked, hack_not_linked],
                "Type": ["Non-Hackathon", "Hackathon", "Hackathon"],
            }
        )

        fig_labels = px.bar(
            label_data,
            x="Count",
            y="Category",
            orientation="h",
            color="Type",
            color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
            text="Count",
        )
        fig_labels.update_layout(
            height=300, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="white",
            yaxis_title="",
            xaxis_title="Number of Repositories",
            showlegend=False,
        )
        fig_labels.update_traces(textposition="outside", textfont_size=14)
        st.plotly_chart(fig_labels, use_container_width=True)

    # LauzHack projects vs repositories
    st.markdown("### LauzHack Repositories by Year")

    # Projects summary stats
    years_to_show = ["2023", "2024", "2025"]
    project_counts = {}
    for year in years_to_show:
        project_counts[year] = df[df["year"] == year]["project_foreign_key"].dropna().nunique()

    total_projects = sum(project_counts.values())
    total_repos = len(df[df["year"].isin(years_to_show)])

    st.caption(
        f"**{total_projects} projects** with **{total_repos} repositories** (of 210 total LauzHack projects). "
        f"Green = accessible repos, Red = deleted/inaccessible."
    )

    # Repositories stacked bar
    repo_data = []
    for year in years_to_show:
        year_df = df[df["year"] == year]
        valid_count = year_df["has_valid_metadata"].sum()
        inaccessible_count = len(year_df) - valid_count
        total = len(year_df)
        repo_data.append({
            "Year": year,
            "Valid Metadata": valid_count,
            "Inaccessible": inaccessible_count,
            "Total": total
        })

    repo_df = pd.DataFrame(repo_data)

    fig_repos = go.Figure()

    # Add stacked bars
    fig_repos.add_trace(go.Bar(
        x=repo_df["Year"],
        y=repo_df["Valid Metadata"],
        name="Valid Metadata",
        marker_color=SDSC_GREEN,
    ))

    fig_repos.add_trace(go.Bar(
        x=repo_df["Year"],
        y=repo_df["Inaccessible"],
        name="Inaccessible",
        marker_color=ROSE,
    ))

    # Add total labels on top of each bar
    for _, row in repo_df.iterrows():
        fig_repos.add_annotation(
            x=row["Year"],
            y=row["Total"],
            text=f"<b>{int(row['Total'])}</b>",
            showarrow=False,
            yshift=12,
            font=dict(size=11, color="black"),
        )

    fig_repos.update_layout(
        barmode="stack",
        height=380,
        showlegend=True,
        xaxis_title="",
        yaxis_title="Number of Repositories",
        hovermode="x unified",
    )
    st.plotly_chart(fig_repos, use_container_width=True)

    st.markdown("---")

    st.subheader("Language Landscape")
    st.caption(
        "Programming languages used across repositories. Shows whether hackathon projects "
        "favor different languages (e.g., more Python for quick prototyping)."
    )
    lang_df = df.copy()
    lang_df["primary_language"] = lang_df["primary_language"].fillna("Unknown").replace("", "Unknown")
    lang_df["label"] = lang_df["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"})

    lang_counts = (
        lang_df.groupby(["primary_language", "label"]).size().reset_index(name="count")
    )
    top_langs = lang_counts.groupby("primary_language")["count"].sum().nlargest(12).index
    lang_counts = lang_counts[lang_counts["primary_language"].isin(top_langs)]

    fig_lang = px.treemap(
        lang_counts,
        path=["label", "primary_language"],
        values="count",
        color="label",
        color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
    )
    fig_lang.update_layout(height=450, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white")
    fig_lang.update_traces(textinfo="label+value", textfont_size=13)
    st.plotly_chart(fig_lang, use_container_width=True)


# ---------------------------------------------------------------------------
# Act 2: Feature Engineering & EDA
# ---------------------------------------------------------------------------


def render_features(df: pd.DataFrame):
    st.markdown(
        """
    <h1 style='color:#26235c;'>What Makes a Hackathon Repo?</h1>
    <p style='color:#64748b; font-size:1.05rem;'>
        Now that we've seen the dataset, let's look at what makes hackathon repositories
        different from regular ones. We extracted <b>numeric and boolean features</b>
        for each repository &mdash; things like how many days it was active, how long the README is,
        whether it has tests, and what topics it covers. These <b>structural features</b> are
        hackathon-agnostic and would apply to any hackathon dataset.
    </p>
    <p style='color:#94a3b8; font-size:0.9rem;'>
        We also enriched repos with topic tags from a knowledge database
        (EPFL Graph API in our LauzHack case study) — those results are marked with 🏷️ below.
    </p>
    """,
        unsafe_allow_html=True,
    )

    valid = df[df["has_valid_metadata"]].copy()
    valid["label"] = valid["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"})

    st.markdown("---")

    # Interactive feature selector with box plots
    st.subheader("Distribution of Key Features")
    st.caption(
        "Box plots show the typical range (box), median (line), and outliers (dots). "
        "Green = Hackathon repos, Blue = Non-hackathon repos. Look for non-overlapping boxes — that's where the groups differ most."
    )

    feature_options = {
        "Days with Code Changes": "active_days_default_branch",
        "Repository Age (Days)": "repo_age_days",
        "README Length (characters)": "readme_length",
        "Number of Commits": "commit_count_default_branch",
        "Repository Stars": "stars",
        "Number of Forks": "forks",
        "Number of Contributors": "contributors_count",
        "Total Issues": "issues_total",
        "Total Pull Requests": "pull_requests_total",
        "Total Files": "files_total_count",
        "Total Directories": "dirs_total_count",
        "Watchers": "watchers",
    }

    selected_feature_name = st.selectbox(
        "Choose a metric to visualize:",
        options=list(feature_options.keys()),
        key="feature_selector"
    )
    selected_col = feature_options[selected_feature_name]

    # Display feature definition and calculation
    feature_info = FEATURE_METADATA[selected_feature_name]
    with st.expander(f"ℹ️ What is '{selected_feature_name}'?"):
        st.markdown(f"**Definition:** {feature_info['definition']}")
        if feature_info["is_derived"]:
            st.markdown(f"📐 **Derived Feature** — {feature_info['calculation']}")
        else:
            st.markdown("📊 **Raw Feature** — Collected directly from GitHub")

    st.markdown("")  # Add spacing

    fig_box = go.Figure()

    for label, color in [("Hackathon", HACKATHON_COLOR), ("Non-Hackathon", NON_HACKATHON_COLOR)]:
        subset_mask = valid["label"] == label
        subset_data = valid[subset_mask][selected_col].dropna()
        subset_data = subset_data.clip(lower=0)

        # Get corresponding repo URLs for hover
        repo_urls = valid[subset_mask].loc[subset_data.index, "repo_url"].values

        fig_box.add_trace(go.Box(
            y=subset_data,
            name=label,
            marker_color=color,
            boxmean="sd",
            opacity=0.8,
            boxpoints=False,  # Hide default points
            customdata=repo_urls,
            hovertemplate="<b>%{customdata}</b><br>" +
                          selected_feature_name + ": %{y:,.0f}<br>" +
                          "<extra></extra>",
        ))

        # Add scatter points overlay for individual repos (markers only, no lines)
        fig_box.add_trace(go.Scatter(
            y=subset_data,
            x=[label] * len(subset_data),
            mode="markers",
            marker=dict(color=color, size=6, opacity=0.5),
            customdata=repo_urls,
            hovertemplate="<b>%{customdata}</b><br>" +
                          selected_feature_name + ": %{y:,.0f}<br>" +
                          "<extra></extra>",
            showlegend=False,
            name="",
        ))

    fig_box.update_layout(
        height=500,
        title_text=f"<b>{selected_feature_name}</b><br><sub>Green vs Blue — larger box = wider spread of values</sub>",
        yaxis_title=selected_feature_name,
        xaxis_title="Repository Type",
        plot_bgcolor="#fafcf8",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#e2e8f0"),
        showlegend=True,
        legend=dict(orientation="v", x=1.05, y=1),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("---")

    # Boolean feature heatmap
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.subheader("Project Quality Signals")
        st.caption("Proportion of repos with each quality indicator")

        bool_features = [
            "has_ci",
            "has_tests",
            "has_docker",
            "has_docs",
            "has_readme_file",
            "has_license_file",
            "has_contributing",
            "has_notebooks",
        ]

        bool_data = []
        for feat in bool_features:
            for label in ["Hackathon", "Non-Hackathon"]:
                subset = valid[valid["label"] == label]
                pct = subset[feat].mean() * 100
                bool_data.append({"Feature": feat.replace("has_", "").replace("_", " ").title(), "Class": label, "% Present": pct})

        bool_df = pd.DataFrame(bool_data)

        fig_bool = px.bar(
            bool_df,
            x="% Present",
            y="Feature",
            color="Class",
            barmode="group",
            orientation="h",
            color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
            text=bool_df["% Present"].round(1),
        )
        fig_bool.update_layout(
            height=420, xaxis_title="% of Repos", yaxis_title="",
            plot_bgcolor="#fafcf8", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e2e8f0"),
        )
        fig_bool.update_traces(textposition="outside", texttemplate="%{text:.0f}%")
        st.plotly_chart(fig_bool, use_container_width=True)

    with col_b:
        st.subheader("Topic Tag Density 🏷️")
        st.caption(
            "🏷️ *Case-study-specific (EPFL topics).* How many topic tags each repository received. "
            "Hackathon repos (green) tend to cluster around fewer topics because they focus on "
            "a single prototype, while non-hackathon repos (blue) often span more topics."
        )

        fig_concepts = px.histogram(
            valid,
            x="n_concepts",
            color="label",
            nbins=40,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
            labels={"n_concepts": "Number of Concepts", "label": ""},
        )
        fig_concepts.update_layout(
            height=420, plot_bgcolor="#fafcf8", paper_bgcolor="white",
            xaxis=dict(gridcolor="#e2e8f0"), yaxis=dict(gridcolor="#e2e8f0"),
        )
        st.plotly_chart(fig_concepts, use_container_width=True)

    st.markdown("---")

    # Fisher's exact test - discriminative concepts
    st.subheader("Topics That Best Distinguish Hackathon from Non-Hackathon Repos 🏷️")
    st.caption(
        "🏷️ *Case-study-specific (EPFL topics).* We used a statistical test (Fisher's exact test) to find "
        "which topics appear far more often in one group than the other. Bars pointing **right** (green) are "
        "topics strongly associated with hackathon repos. Bars pointing **left** (blue) signal non-hackathon "
        "repos. Longer bars = stronger association. A value of +2 means the odds of appearing in a hackathon "
        "repo are roughly 4× higher. These specific topics are tied to our LauzHack dataset and may differ "
        "for other hackathons."
    )
    st.caption(
        "*P-values have been corrected for multiple comparisons using Benjamini-Hochberg FDR correction. "
        "Only topics that remain statistically significant after correction are shown with full confidence.*"
    )

    fisher_df = compute_fisher_stats(df)
    # Show only FDR-significant concepts with meaningful effect size (|log2_OR| >= 0.5 means ~1.4x difference)
    significant = fisher_df[fisher_df["significant_fdr"]] if "significant_fdr" in fisher_df.columns else fisher_df
    meaningful = significant[significant["log2_odds_ratio"].abs() >= 0.5]

    top_n = min(25, len(meaningful))  # Adaptive: show up to 25, or fewer if not enough meaningful topics
    n_half = top_n // 2
    top_concepts = pd.concat([meaningful.head(n_half + 1), meaningful.tail(n_half)])
    top_concepts = top_concepts.drop_duplicates(subset="concept").sort_values("log2_odds_ratio")

    fig_fisher = px.bar(
        top_concepts,
        x="log2_odds_ratio",
        y="concept",
        orientation="h",
        color="log2_odds_ratio",
        color_continuous_scale=[[0, SDSC_BLUE], [0.5, "#f8fafc"], [1, SDSC_GREEN]],
        color_continuous_midpoint=0,
        hover_data={
            "hackathon_repos": True,
            "non_hackathon_repos": True,
            "fisher_p_adjusted": ":.2e",
            "odds_ratio": ":.2f",
            "log2_odds_ratio": ":.2f",
        },
        labels={
            "log2_odds_ratio": "Strength of Association (log₂ odds ratio)",
            "concept": "",
            "hackathon_repos": "In hackathon repos",
            "non_hackathon_repos": "In non-hackathon repos",
            "fisher_p_adjusted": "Adjusted p-value (FDR)",
            "odds_ratio": "Odds ratio",
        },
    )
    fig_fisher.update_layout(
        height=max(500, top_n * 22),
        coloraxis_colorbar_title="Strength",
        yaxis=dict(dtick=1),
    )
    st.plotly_chart(fig_fisher, use_container_width=True)

    with st.expander("📚 Further Reading: Fisher's Exact Test & Odds Ratios"):
        st.markdown(
            """
        ## What This Method Does 

        **In Plain English:**
        We're asking: "Which topics appear way more often in hackathon repos vs non-hackathon repos?"
        For each topic, we build a 2×2 table (Topic: Yes/No × Repo Type: Hackathon/Non-hackathon) and
        test whether the association is real or just random chance using Fisher's exact test.
        Topics that show a strong, statistically significant pattern are flagged.

        **Key Metrics:**
        - **Odds Ratio (OR)**: How many times more/less likely a topic appears in hackathon repos
          - OR = 4 means hackathon repos are 4× more likely to have this topic
          - OR = 0.25 means hackathon repos are 4× *less* likely to have it
        - **Log₂ Odds Ratio**: Converted to log scale for visualization
          - +2 = 4× more likely (because 2² = 4)
          - -2 = 4× less likely
        - **P-value**: Probability this pattern happened by random chance (lower = more confident)
        - **FDR Correction**: Adjusts p-values because we test hundreds of topics at once

        ---

        ## Mathematical Formulas

        **1. Fisher's Exact Test - 2×2 Contingency Table:**
        """
        )
        st.latex(r"""
        \begin{array}{c|cc}
        & \text{Topic Present} & \text{Topic Absent} \\
        \hline
        \text{Hackathon Repo} & a & c \\
        \text{Non-Hackathon Repo} & b & d \\
        \end{array}
        """)

        st.markdown(
            """
        Where:
        - **a** = count of hackathon repos WITH the topic
        - **b** = count of non-hackathon repos WITH the topic
        - **c** = count of hackathon repos WITHOUT the topic (= n_hack - a)
        - **d** = count of non-hackathon repos WITHOUT the topic (= n_non - b)

        **Fisher's exact p-value** (hypergeometric distribution):
        """
        )
        st.latex(r"""
        p = \frac{\binom{a+b}{a}\binom{c+d}{c}}{\binom{n}{a+c}}
        """)

        st.markdown(
            """
        **2. Odds Ratio (OR):**
        """
        )
        st.latex(r"""
        \text{OR} = \frac{a \cdot d}{b \cdot c}
        """)

        st.markdown(
            """
        Interpretation:
        - OR > 1: Topic favors hackathon repos
        - OR < 1: Topic favors non-hackathon repos
        - OR = 1: No association

        **3. Log₂ Odds Ratio (What You See in Chart):**
        """
        )
        st.latex(r"""
        \text{Log}_2(\text{OR}) = \log_2\left(\frac{a \cdot d}{b \cdot c}\right)
        """)

        st.markdown(
            """
        **4. FDR Correction (Benjamini-Hochberg):**

        When testing hundreds of topics, some will appear significant by random chance.
        FDR correction adjusts all p-values upward to control the False Discovery Rate at 5%.

        - Only topics with FDR-adjusted p-value < 0.05 are shown with full confidence
        - Protects against false positives in multiple comparisons

        **5. Importance Score (For Ranking):**
        """
        )
        st.latex(r"""
        \text{Importance} = -\log_{10}(p_{\text{adjusted}}) \times \text{sign}(\text{Log}_2(\text{OR}))
        """)

        st.markdown(
            """
        Combines strength (p-value) and direction (positive for hackathon, negative for non-hackathon).

        ---

        ## Why This Matters c of hackathons
        - **Statistically Rigorous**: Fisher's exact test makes no distributional assumptions
        - **Multiple Testing Controlled**: FDR correction prevents false alarms
        - **Interpretable**: Log₂ scale shows practical effect sizes (2× difference = log₂ of 1)
        """
        )

    with st.expander("How We Chose Which Features to Use"):
        st.markdown(
            """
        1. Started with the raw dataset columns (metadata, text fields, numeric features, and derived scores)
        2. Removed text fields (URLs, descriptions, dates) that aren't numeric
        3. Removed **features that "cheat"** — columns that already contain the answer we're trying to predict
           (like "is this linked to a hackathon project?" or prediction scores themselves)
        4. Removed **topic-based one-hot features** that were too specific to this particular hackathon and wouldn't generalize
        5. Removed **constant columns** that have no variation across repos
        6. Removed **highly correlated pairs** (correlation > 0.85) — when two features track essentially the same thing, we kept the one more correlated with the target
        7. The final model uses the surviving features for prediction
        """
        )


# ---------------------------------------------------------------------------
# Act 3: Three Prediction Methods
# ---------------------------------------------------------------------------


def _friendly_feature_name(raw: str) -> str:
    """Convert raw column names to human-readable labels."""
    mapping = {
        "active_days_default_branch": "Days with code changes",
        "commit_count_default_branch": "Number of commits",
        "readme_length": "README length",
        "repo_age_days": "Repository age (days)",
        "n_concepts": "Number of topic tags",
        "files_total_count": "Total files",
        "dirs_total_count": "Total directories",
        "contributors_count": "Number of contributors",
        "contributors_total": "Total contributors",
        "forks": "Fork count",
        "stars": "Star count",
        "watchers": "Watcher count",
        "issues_total": "Total issues",
        "issues_open": "Open issues",
        "issues_closed": "Closed issues",
        "pull_requests_total": "Total pull requests",
        "pull_requests_merged": "Merged pull requests",
        "releases_count": "Number of releases",
        "commit_log": "Commit count (log scale)",
        "stars_log": "Stars (log scale)",
        "contributors_count_log": "Contributors (log scale)",
        "has_ci": "Has CI/CD pipeline",
        "has_tests": "Has test suite",
        "has_docker": "Has Docker setup",
        "has_docs": "Has documentation",
        "has_readme_file": "Has README",
        "has_license_file": "Has license",
        "has_contributing": "Has contributing guide",
        "has_notebooks": "Has Jupyter notebooks",
        "is_fork": "Is a fork",
        "is_archived": "Is archived",
    }
    return mapping.get(raw, raw.replace("_", " ").replace("default branch", "").strip().title())


def render_methods(df: pd.DataFrame):
    st.markdown(
        """
    <h1 style='color:#26235c;'>Three Attempts at Prediction</h1>
    <p style='color:#64748b; font-size:1.05rem;'>
        We found clear differences between hackathon and non-hackathon repos. But can we use
        those differences to <b>automatically classify</b> new repos? Here are three approaches
        we tried, from simplest to most sophisticated. Each reveals different trade-offs between
        <b>precision</b> (of repos we flagged as hackathon, how many actually were?) and
        <b>recall</b> (of all actual hackathon repos, how many did we catch?).
    </p>
    """,
        unsafe_allow_html=True,
    )

    valid = df[df["has_valid_metadata"]].copy()
    y_true = valid["true_hackathon_repos"].astype(int)

    n_hack = int(y_true.sum())
    n_non = int(len(y_true) - n_hack)
    st.caption(
        f"Evaluation set: **{len(y_true)} repos** with complete metadata "
        f"({n_hack} hackathon, {n_non} non-hackathon — a {n_hack / len(y_true):.0%} / {n_non / len(y_true):.0%} split). "
        f"For context, always guessing the majority class would give {max(n_hack, n_non) / len(y_true):.0%} accuracy."
    )

    tab_naive, tab_corr, tab_rf = st.tabs(
        ["1. Simple Keyword Rules", "2. Statistical Weighting", "3. Machine Learning (Random Forest)"]
    )

    # --- Tab 1: Naive ---
    with tab_naive:
        st.markdown(
            f"""
        <div style='padding:16px; background:linear-gradient(135deg, #fef9ee, #fdf3dc); border-radius:12px; border-left:4px solid #e8a838;'>
            <h3 style='margin:0; color:#b47818;'>Method 1: Simple Keyword Rules</h3>
            <p style='margin:8px 0 0 0; color:#475569;'>
                A hand-crafted scoring system using <b>5 signal categories</b>:
                text matches (does the README mention "hackathon"?),
                popularity signals (stars, forks),
                contributor patterns (how many people worked on it?),
                commit activity (how many changes were made?),
                and burst commit windows (was all the work done in 1-2 days, like at a hackathon?).
                If enough signals fire, we predict "hackathon repo."
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        y_pred_naive = valid["repo_predicted_flag"].astype(int)
        metrics_naive = compute_metrics(y_true, y_pred_naive)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics_naive['Accuracy']:.1%}", help="Of all repos, what % did we classify correctly?")
        m2.metric("Precision", f"{metrics_naive['Precision']:.1%}", help="Of repos we called 'hackathon', what % actually were? Higher = fewer false alarms.")
        m3.metric("Recall", f"{metrics_naive['Recall']:.1%}", help="Of all actual hackathon repos, what % did we find? Higher = fewer missed.")
        m4.metric("F1 Score", f"{metrics_naive['F1']:.1%}", help="Balanced combination of Precision and Recall. Above 80% is generally good.")

        col_sig, col_cm = st.columns([3, 2])

        with col_sig:
            st.markdown("**Rule Breakdown**: Which rules triggered, and how often did they match actual hackathon vs. non-hackathon repos?")
            # Parse signals
            signal_data = []
            for _, row in valid.iterrows():
                signals = row.get("repo_predicted_signals", "")
                if pd.isna(signals) or signals == "":
                    continue
                is_hack = bool(row["true_hackathon_repos"])
                for sig in str(signals).split("|"):
                    sig = sig.strip()
                    if sig:
                        signal_data.append({"Signal": sig.replace("_", " ").title(), "True Label": "Hackathon" if is_hack else "Non-Hackathon"})

            if signal_data:
                sig_df = pd.DataFrame(signal_data)
                sig_counts = sig_df.groupby(["Signal", "True Label"]).size().reset_index(name="Count")

                fig_sig = px.bar(
                    sig_counts,
                    x="Count",
                    y="Signal",
                    color="True Label",
                    barmode="group",
                    orientation="h",
                    color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
                )
                fig_sig.update_layout(
                    height=350, yaxis=dict(categoryorder="total ascending"),
                    plot_bgcolor="#fafcf8", paper_bgcolor="white",
                    xaxis=dict(gridcolor="#e2e8f0"),
                )
                st.plotly_chart(fig_sig, use_container_width=True)

        with col_cm:
            fig_cm = plot_confusion_matrix(y_true, y_pred_naive, "Simple Keyword Rules", "Oranges")
            st.plotly_chart(fig_cm, use_container_width=True)

        st.info(
            f"**Insight:** High precision ({metrics_naive['Precision']:.0%}) but misses "
            f"{1 - metrics_naive['Recall']:.0%} of true hackathon repos. "
            "The rules are too conservative — they catch obvious cases but miss subtle ones."
        )

    # --- Tab 2: Correlation-Weighted ---
    with tab_corr:
        st.markdown(
            f"""
        <div style='padding:16px; background:linear-gradient(135deg, #eef0f8, #e1e4f0); border-radius:12px; border-left:4px solid #5561a6;'>
            <h3 style='margin:0; color:#3d4780;'>Method 2: Statistical Weighting</h3>
            <p style='margin:8px 0 0 0; color:#475569;'>
                This method measures how strongly each characteristic is linked to being a
                hackathon repo, then combines them into a single score. If a repo's combined
                score is above 0.5 (on a 0-to-1 scale), we predict "hackathon." It's data-driven
                but assumes each characteristic contributes independently — which isn't always true.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        y_pred_corr = valid["corr_weighted_flag"].astype(int)
        metrics_corr = compute_metrics(y_true, y_pred_corr)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics_corr['Accuracy']:.1%}", help="Of all repos, what % did we classify correctly?")
        m2.metric("Precision", f"{metrics_corr['Precision']:.1%}", help="Of repos we called 'hackathon', what % actually were? Higher = fewer false alarms.")
        m3.metric("Recall", f"{metrics_corr['Recall']:.1%}", help="Of all actual hackathon repos, what % did we find? Higher = fewer missed.")
        m4.metric("F1 Score", f"{metrics_corr['F1']:.1%}", help="Balanced combination of Precision and Recall. Above 80% is generally good.")

        col_dist, col_cm2 = st.columns([3, 2])

        with col_dist:
            st.markdown("**Score Distribution**: Why this method over-predicts")
            fig_score = px.histogram(
                valid,
                x="corr_weighted_score",
                color=valid["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"}),
                nbins=50,
                barmode="overlay",
                opacity=0.75,
                color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
                labels={"corr_weighted_score": "Correlation-Weighted Score", "color": ""},
            )
            fig_score.add_vline(x=0.5, line_dash="dash", line_color=ROSE, line_width=2, annotation_text="Threshold = 0.5")
            fig_score.update_layout(height=380, plot_bgcolor="#fafcf8", paper_bgcolor="white")
            st.plotly_chart(fig_score, use_container_width=True)

        with col_cm2:
            fig_cm2 = plot_confusion_matrix(y_true, y_pred_corr, "Statistical Weighting", "Purples")
            st.plotly_chart(fig_cm2, use_container_width=True)

        st.warning(
            f"**Insight:** {metrics_corr['Recall']:.0%} recall but only {metrics_corr['Precision']:.0%} precision. "
            "This method calls almost everything a hackathon repo. "
            "The distributions heavily overlap — linear weighting can't separate the classes."
        )

    # --- Tab 3: Random Forest ---
    with tab_rf:
        st.markdown(
            f"""
        <div style='padding:16px; background:linear-gradient(135deg, #f3f7ea, #e6edcc); border-radius:12px; border-left:4px solid #90ca42;'>
            <h3 style='margin:0; color:#5a8020;'>Method 3: Machine Learning (Random Forest)</h3>
            <p style='margin:8px 0 0 0; color:#475569;'>
                A machine learning model that builds 100 decision trees, each learning slightly
                different patterns. To ensure trustworthy results, we tested the model on data
                it had never seen during training (splitting the data into 5 parts, rotating
                which part is the test set). We also adjusted for the fact that non-hackathon
                repos slightly outnumber hackathon ones.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        y_pred_rf = valid["ml_predicted_flag"].astype(int)
        metrics_rf = compute_metrics(y_true, y_pred_rf)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{metrics_rf['Accuracy']:.1%}", help="Of all repos, what % did we classify correctly?")
        m2.metric("Precision", f"{metrics_rf['Precision']:.1%}", help="Of repos we called 'hackathon', what % actually were? Higher = fewer false alarms.")
        m3.metric("Recall", f"{metrics_rf['Recall']:.1%}", help="Of all actual hackathon repos, what % did we find? Higher = fewer missed.")
        m4.metric("F1 Score", f"{metrics_rf['F1']:.1%}", help="Balanced combination of Precision and Recall. Above 80% is generally good.")

        col_imp, col_cm3 = st.columns([3, 2])

        with col_imp:
            st.markdown("**What the Model Learned**: Which features mattered most?")
            st.caption("Taller bars mean the model relied more heavily on that feature when deciding. "
                       "These importances come from a model refit on all data for illustration purposes.")
            importance_df, feature_cols = compute_rf_importances(df)
            top_15 = importance_df.head(15).sort_values("importance")
            top_15["display_name"] = top_15["feature"].apply(_friendly_feature_name)

            fig_imp = px.bar(
                top_15,
                x="importance",
                y="display_name",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, GREEN_LIGHT], [1, SDSC_GREEN]],
                labels={"importance": "How Much This Feature Helped Decide", "display_name": ""},
            )
            fig_imp.update_layout(height=450, coloraxis_showscale=False)
            st.plotly_chart(fig_imp, use_container_width=True)
            st.caption(
                "*Feature importances are from a model refit on all data for illustration. "
                "The predictions in the confusion matrix come from out-of-fold cross-validation "
                "(never trained on the data being predicted).*"
            )

        with col_cm3:
            fig_cm3 = plot_confusion_matrix(y_true, y_pred_rf, "Random Forest", "Blues")
            st.plotly_chart(fig_cm3, use_container_width=True)

        # Probability distribution
        st.markdown(
            "**How Confident Is the Model?** Scores near 0 = confident it's NOT hackathon; "
            "scores near 1 = confident it IS. Scores near 0.5 = uncertain."
        )
        fig_proba = px.histogram(
            valid,
            x="ml_predicted_proba",
            color=valid["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"}),
            nbins=50,
            barmode="overlay",
            opacity=0.75,
            color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
            labels={"ml_predicted_proba": "Model Confidence (0 = not hackathon, 1 = hackathon)", "color": ""},
        )
        fig_proba.add_vline(x=0.5, line_dash="dash", line_color=ROSE, line_width=2, annotation_text="Cutoff: above = predicted hackathon")
        fig_proba.update_layout(height=350, plot_bgcolor="#fafcf8", paper_bgcolor="white")
        st.plotly_chart(fig_proba, use_container_width=True)

        st.success(
            "**Insight:** Balanced performance across all metrics. "
            "The model captures complex combinations of characteristics — for example, a repo "
            "that is both very young AND has very few contributors is more likely to be hackathon "
            "code, even if neither characteristic alone would be enough to tell. "
            "The probability distributions show much cleaner separation between classes."
        )


# ---------------------------------------------------------------------------
# Act 4: The Verdict
# ---------------------------------------------------------------------------


def render_verdict(df: pd.DataFrame):
    st.markdown(
        """
    <h1 style='color:#26235c;'>The Verdict</h1>
    <p style='color:#64748b; font-size:1.05rem;'>
        Each method had strengths and weaknesses. Let's put them head to head
        to see which one actually works best.
    </p>
    """,
        unsafe_allow_html=True,
    )

    valid = df[df["has_valid_metadata"]].copy()
    y_true = valid["true_hackathon_repos"].astype(int)

    methods = {
        "Naive Rule-Based": valid["repo_predicted_flag"].astype(int),
        "Correlation-Weighted": valid["corr_weighted_flag"].astype(int),
        "Random Forest": valid["ml_predicted_flag"].astype(int),
    }

    all_metrics = {}
    for name, y_pred in methods.items():
        all_metrics[name] = compute_metrics(y_true, y_pred)

    # Radar chart
    st.subheader("Performance Radar")
    st.caption(
        "Each colored shape represents one method. A perfect method would fill the entire pentagon. "
        "The method covering the most area performs best overall. **Specificity** = of all non-hackathon "
        "repos, what % did we correctly identify as non-hackathon (recall for the negative class)."
    )
    categories = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]

    fig_radar = go.Figure()
    dashes = ["solid", "dash", "dot"]
    for idx, (name, metrics) in enumerate(all_metrics.items()):
        values = [metrics[c] for c in categories] + [metrics[categories[0]]]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=name,
                line_color=METHOD_COLORS[name],
                fillcolor=METHOD_COLORS[name],
                opacity=0.25,
                line_width=3,
                line_dash=dashes[idx],
            )
        )
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%", gridcolor="#e2e8f0"),
            angularaxis=dict(gridcolor="#e2e8f0"),
            bgcolor="#fafbff",
        ),
        height=500,
        legend=dict(orientation="h", y=-0.1, font=dict(size=13)),
        paper_bgcolor="white",
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # Confusion matrix triptych
    st.subheader("Confusion Matrix Comparison")
    st.caption(
        "Each grid shows what the method predicted (columns) vs. reality (rows). "
        "**Top-left** and **bottom-right** cells are correct predictions. "
        "**Top-right** = repos wrongly called 'hackathon' (false alarms). "
        "**Bottom-left** = real hackathon repos that were missed. Bigger numbers on the diagonal = better."
    )
    cols = st.columns(3)
    color_scales = ["Oranges", "Purples", "Greens"]
    for idx, (name, y_pred) in enumerate(methods.items()):
        with cols[idx]:
            f1 = all_metrics[name]["F1"]
            fig = plot_confusion_matrix(y_true, y_pred, f"{name}\nF1={f1:.1%}", color_scales[idx])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Metrics table
    st.subheader("Metrics Summary")
    metrics_table = pd.DataFrame(all_metrics).T
    metrics_table = metrics_table[categories]

    # Style the table
    def highlight_best(s):
        is_max = s == s.max()
        return ["background-color: #f3f7ea; font-weight: bold; color: #5a8020" if v else "" for v in is_max]

    styled = metrics_table.style.format("{:.1%}").apply(highlight_best)
    st.dataframe(styled, use_container_width=True, height=160)
    st.caption(
        "Highlighted cells show the best score per column. **F1 score** matters most here because "
        "we want to balance catching hackathon repos (recall) without too many false alarms (precision)."
    )

    st.markdown("---")

    # Error analysis scatter
    st.subheader("Error Analysis: Where Do Methods Disagree?")
    st.caption(
        "This chart compares the Random Forest (x-axis) and statistical weighting (y-axis) side by side. "
        "Dots in the top-right were flagged as hackathon by both. Dots in the bottom-left were rejected by both. "
        "The other two corners show disagreements — the hardest cases. X-shaped dots are repos the Random Forest got wrong."
    )

    valid["rf_correct"] = (valid["ml_predicted_flag"].astype(int) == y_true).map(
        {True: "RF Correct", False: "RF Wrong"}
    )
    valid["true_label"] = valid["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"})

    fig_error = px.scatter(
        valid,
        x="ml_predicted_proba",
        y="corr_weighted_score",
        color="true_label",
        symbol="rf_correct",
        hover_data=["name_with_owner", "description", "repo_predicted_flag", "primary_language"],
        color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
        symbol_map={"RF Correct": "circle", "RF Wrong": "x"},
        opacity=0.8,
        labels={
            "ml_predicted_proba": "Random Forest Probability",
            "corr_weighted_score": "Correlation-Weighted Score",
            "true_label": "True Label",
            "rf_correct": "RF Result",
        },
    )
    fig_error.add_vline(x=0.5, line_dash="dot", line_color="#94a3b8", line_width=1.5)
    fig_error.add_hline(y=0.5, line_dash="dot", line_color="#94a3b8", line_width=1.5)
    fig_error.update_layout(
        height=550,
        plot_bgcolor="#fafcf8",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
    )
    fig_error.update_traces(marker=dict(size=9, line=dict(width=0.5, color="white")))

    # Add quadrant annotations
    fig_error.add_annotation(x=0.15, y=0.9, text="Statistical method<br>says hackathon,<br>RF disagrees", showarrow=False, font=dict(color="#94a3b8", size=10))
    fig_error.add_annotation(x=0.85, y=0.9, text="Both agree:<br>Hackathon", showarrow=False, font=dict(color=SDSC_GREEN, size=10))
    fig_error.add_annotation(x=0.15, y=0.1, text="Both agree:<br>Not hackathon", showarrow=False, font=dict(color=SDSC_BLUE, size=10))
    fig_error.add_annotation(x=0.85, y=0.1, text="RF says hackathon,<br>statistical method<br>disagrees", showarrow=False, font=dict(color="#94a3b8", size=10))

    st.plotly_chart(fig_error, use_container_width=True)


# ---------------------------------------------------------------------------
# Act 5: Conclusions & Explorer
# ---------------------------------------------------------------------------


def render_conclusions(df: pd.DataFrame):
    st.markdown(
        """
    <h1 style='color:#26235c;'>What We Learned</h1>
    <p style='color:#64748b; font-size:1.05rem;'>
        After comparing all three approaches, here's what we learned — and what surprised us.
    </p>
    """,
        unsafe_allow_html=True,
    )

    # Compute dynamic metrics for conclusions
    valid = df[df["has_valid_metadata"]].copy()
    y_true = valid["true_hackathon_repos"].astype(int)
    rf_metrics = compute_metrics(y_true, valid["ml_predicted_flag"].astype(int))
    naive_metrics = compute_metrics(y_true, valid["repo_predicted_flag"].astype(int))
    corr_metrics = compute_metrics(y_true, valid["corr_weighted_flag"].astype(int))
    unique_concepts = df["concept_list"].explode().nunique()

    # General takeaways
    st.markdown("#### General Findings (hackathon-agnostic)")
    st.success(
        "**Hackathon repos have a distinctive temporal fingerprint.** "
        "Burst activity in 1-2 days, then silence. The number of days with code changes "
        "is the single strongest signal separating hackathon repos from regular ones."
    )
    st.warning(
        f"**Simple rules achieve high precision but miss nuanced cases.** "
        f"The keyword approach would miss {1 - naive_metrics['Recall']:.0%} of actual hackathon repos. "
        f"Machine learning captures the complex reality."
    )

    # Case-study-specific takeaways
    st.markdown("#### LauzHack Case Study Findings 🏷️")
    st.info(
        f"**Topic tags add useful signal beyond code structure.** "
        f"The EPFL knowledge database identified {unique_concepts:,}+ topics in the LauzHack dataset. "
        f"A statistical test revealed which topics are strongly linked to hackathon vs. production code. "
        f"Different knowledge databases would yield different topic distributions for other hackathons."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div style='padding:20px; background:linear-gradient(135deg, #f3f7ea, #e6edcc); border-radius:12px; border:1px solid #c5e085;'>
            <h3 style='color:#5a8020; margin-top:0;'>What Worked</h3>
            <ul style='color:#475569;'>
                <li>The ML model correctly classified <b>{rf_metrics['Accuracy']:.1%}</b> of repos, with an F1 score of <b>{rf_metrics['F1']:.1%}</b></li>
                <li>Adding topic tags (🏷️ from EPFL in our case study) gave the model information it couldn't get from code structure alone</li>
                <li>We carefully avoided letting the model "peek" at the answers during training</li>
                <li>Adjusting for the class split prevented the model from being biased toward the majority</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div style='padding:20px; background:linear-gradient(135deg, #eef0f8, #e1e4f0); border-radius:12px; border:1px solid #c8cde3;'>
            <h3 style='color:#5561a6; margin-top:0;'>What Didn't Work</h3>
            <ul style='color:#475569;'>
                <li>Keyword matching alone was <b>too cautious</b> &mdash; it missed {1 - naive_metrics['Recall']:.0%} of actual hackathon repos</li>
                <li>Statistical weighting <b>flagged too many repos</b> as hackathon (only {corr_metrics['Precision']:.0%} were correct)</li>
                <li>Some labels in our training data were noisy &mdash; organizational repos linked to projects but not actually hackathon code</li>
                <li>🏷️ Using individual topic tags as yes/no features made the model too specific to the LauzHack case study</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Limitations
    with st.expander("Limitations & Future Work"):
        st.markdown(
            f"""
        **Known limitations of this analysis:**
        - **Small dataset** — {len(valid)} repos is enough for meaningful patterns but limits the complexity of models we can reliably train. Results should be validated on a larger corpus.
        - **Single hackathon as case study** — All data comes from LauzHack. While the methodology is general, the trained model and topic-based features might not generalize to hackathons with different tech stacks, team sizes, or durations (e.g., Devpost, MLH events).
        - **Noisy labels** — Some repos are linked to hackathon projects via organizational accounts but don't contain hackathon code. This adds noise to both training and evaluation.
        - **No hyperparameter tuning** — The Random Forest parameters (100 trees, max depth 8) were reasonable defaults, not optimized via grid search or Bayesian optimization.
        - **No confidence intervals** — Metrics are point estimates from a single 5-fold CV run. Per-fold variance is not shown.

        **Future directions:**
        - Validate on held-out hackathons from Devpost or other platforms
        - Experiment with gradient boosting (XGBoost/LightGBM) which often outperforms RF on tabular data
        - Add temporal features (e.g., commit burst detection, time-of-day patterns)
        - Use SHAP values for better model interpretability
        """
        )

    st.markdown("---")

    # Concept Network Graph
    st.subheader("Concept Network: How Repos Are Connected 🏷️")
    st.caption(
        "🏷️ *Case-study-specific (EPFL topics).* Two repos are linked when they share enough "
        "**rare, domain-specific** topics — generic ones like 'Python' or 'GitHub' that appear "
        "in 20%+ of repos are excluded so that connections reflect genuine thematic similarity."
    )

    net_col1, net_col2 = st.columns([1, 3])
    with net_col1:
        min_similarity = st.number_input(
            "Min Jaccard similarity",
            min_value=0.01,
            max_value=1.0,
            value=0.15,
            step=0.01,
            format="%.2f",
            help="Jaccard similarity = (shared rare concepts) / (combined rare concepts). "
                 "0.15 means ≥15% topic overlap. Higher = fewer, stronger connections.",
        )
        max_concept_pct = st.number_input(
            "Max concept frequency (%)",
            min_value=1,
            max_value=50,
            value=15,
            step=1,
            help="Exclude concepts appearing in more than this % of repos. "
                 "Lower = stricter (only truly rare concepts count).",
        )

    node_df, edge_df = compute_concept_network(df, min_similarity=min_similarity, max_concept_freq=max_concept_pct / 100)

    with net_col2:
        if len(node_df) > 0:
            n_hack = node_df["is_hackathon"].sum()
            n_non = len(node_df) - n_hack
            st.markdown(
                f"**{len(node_df)} repos** ({n_hack} hackathon, {n_non} non-hackathon) "
                f"connected by **{len(edge_df)} links**"
            )
        else:
            st.markdown("No connections at this threshold.")

    if len(edge_df) > 0 and len(node_df) > 0:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(node_df["idx"].tolist())
        for _, e in edge_df.iterrows():
            G.add_edge(int(e["source"]), int(e["target"]), weight=e["jaccard"])
        positions = nx.spring_layout(G, seed=42, k=1.5 / (len(G.nodes) ** 0.5 + 1), iterations=50, weight="weight")

        node_df = node_df.copy()
        node_df["x"] = node_df["idx"].map(lambda i: positions[i][0])
        node_df["y"] = node_df["idx"].map(lambda i: positions[i][1])
        node_df["label"] = node_df["is_hackathon"].map({True: "Hackathon", False: "Non-Hackathon"})

        fig_net = go.Figure()

        # Edges as a single trace
        edge_x, edge_y = [], []
        for _, e in edge_df.iterrows():
            s, t = int(e["source"]), int(e["target"])
            edge_x += [positions[s][0], positions[t][0], None]
            edge_y += [positions[s][1], positions[t][1], None]

        fig_net.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(width=0.5, color="rgba(100,116,139,0.25)"),
            hoverinfo="none",
            showlegend=False,
        ))

        # Nodes
        for label, color in [("Hackathon", HACKATHON_COLOR), ("Non-Hackathon", NON_HACKATHON_COLOR)]:
            subset = node_df[node_df["label"] == label]
            fig_net.add_trace(go.Scatter(
                x=subset["x"], y=subset["y"],
                mode="markers",
                marker=dict(
                    size=subset["n_rare_concepts"].clip(lower=5, upper=60).apply(lambda v: 7 + v * 0.12),
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                name=label,
                text=subset.apply(
                    lambda r: f"<b>{r['name']}</b><br>"
                              f"Language: {r['primary_language']}<br>"
                              f"Total topics: {r['n_concepts']} ({r['n_rare_concepts']} rare)<br>"
                              f"{r['description'][:80] + '...' if len(r['description']) > 80 else r['description']}",
                    axis=1,
                ),
                hoverinfo="text",
            ))

        fig_net.update_layout(
            height=650,
            plot_bgcolor="#fafcf8",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            legend=dict(orientation="h", y=1.02, font=dict(size=13)),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_net, use_container_width=True)

        # --- Repo similarity explorer ---
        st.markdown("**Find similar repos:** select a repository to see its closest matches by topic similarity.")
        repo_names = sorted(node_df["name"].tolist())
        selected_repo = st.selectbox("Select a repository", repo_names, index=None, placeholder="Type to search...")

        if selected_repo:
            sel_idx = int(node_df[node_df["name"] == selected_repo]["idx"].iloc[0])
            # Find edges involving this repo
            neighbors = edge_df[(edge_df["source"] == sel_idx) | (edge_df["target"] == sel_idx)].copy()
            neighbors["neighbor_idx"] = neighbors.apply(
                lambda r: int(r["target"]) if int(r["source"]) == sel_idx else int(r["source"]),
                axis=1,
            )
            neighbors = neighbors.sort_values("jaccard", ascending=False)

            if len(neighbors) > 0:
                neighbor_info = []
                for _, e in neighbors.iterrows():
                    n_idx = e["neighbor_idx"]
                    n_row = node_df[node_df["idx"] == n_idx]
                    if len(n_row) == 0:
                        continue
                    n_row = n_row.iloc[0]
                    neighbor_info.append({
                        "Repository": n_row["name"],
                        "Similarity": f"{e['jaccard']:.0%}",
                        "Shared Rare Concepts": e["shared_count"],
                        "Example Concepts": e["example_concepts"],
                        "Type": "Hackathon" if n_row["is_hackathon"] else "Non-Hackathon",
                        "Language": n_row["primary_language"],
                    })
                st.dataframe(pd.DataFrame(neighbor_info), use_container_width=True, hide_index=True)
            else:
                st.info("No connections for this repo at the current threshold.")

        with st.expander("How this graph works"):
            st.markdown(
                """
            **Why filter out common concepts?** Topics like "Python", "GitHub", or "Open source" appear
            in 20-44% of all repos. Sharing these means nothing — it just says both repos use Python.
            By excluding concepts above the frequency threshold, connections only form when repos share
            genuinely distinctive topics (e.g., "gesture recognition", "semantic web", "AI alignment").

            **Jaccard similarity** measures overlap as: (shared rare concepts) ÷ (total rare concepts in either repo).
            This normalizes for repo size — a small repo sharing 5 rare concepts with another small repo
            scores higher than a huge repo sharing 5 out of 200.

            **What to look for:**
            - **Clusters of same color** → repos in similar domains tend to be the same type (hackathon or not)
            - **Mixed clusters** → some domains have both hackathon and non-hackathon repos
            - **Isolated pairs** → repos with very niche, unique topic overlap
            - Use the **"Find similar repos"** dropdown to explore specific connections
            """
            )
    else:
        st.info(
            f"No connections at Jaccard ≥ {min_similarity:.0%} with concept frequency ≤ {max_concept_pct}%. "
            "Try lowering similarity or raising the frequency cutoff."
        )

    st.markdown("---")

    # Interactive Explorer
    st.subheader("Interactive Repository Explorer")
    st.caption(
        "Explore individual repositories. Each dot is a repo — hover for details. "
        "Dot size reflects commit count. Filter by whether the three methods agree or disagree."
    )

    valid = df[df["has_valid_metadata"]].copy()

    # Sidebar-like filter within main area
    filter_col1, filter_col2 = st.columns([1, 3])
    with filter_col1:
        agreement_filter = st.selectbox(
            "Prediction Agreement",
            ["All", "All 3 agree (hackathon)", "All 3 agree (not hackathon)", "Methods disagree"],
        )

    filtered = valid.copy()
    if agreement_filter == "All 3 agree (hackathon)":
        filtered = filtered[filtered["prediction_agreement"] == 3]
    elif agreement_filter == "All 3 agree (not hackathon)":
        filtered = filtered[filtered["prediction_agreement"] == 0]
    elif agreement_filter == "Methods disagree":
        filtered = filtered[filtered["prediction_agreement"].isin([1, 2])]

    with filter_col2:
        st.markdown(f"**Repos shown:** {len(filtered)} of {len(valid)} total")

    filtered["true_label"] = filtered["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"})

    # Create meaningful derived dimensions
    filtered["log_commits"] = np.log1p(filtered["commit_count_default_branch"])
    filtered["log_age"] = np.log1p(filtered["repo_age_days"])
    filtered["activity_intensity"] = filtered["commit_count_default_branch"] / filtered["active_days_default_branch"].replace(0, 1)
    filtered["log_intensity"] = np.log1p(filtered["activity_intensity"])

    # Rename columns for hover display
    filtered = filtered.rename(
        columns={
            "name_with_owner": "Repository",
            "description": "Description",
            "primary_language": "Language",
            "repo_predicted_flag": "Rule-based prediction",
            "corr_weighted_flag": "Statistical prediction",
            "ml_predicted_flag": "ML prediction",
            "ml_predicted_proba": "ML confidence",
            "prediction_agreement": "Methods agreeing (of 3)",
        }
    )

    # Axis selector for interactivity
    ax_options = {
        "ML Confidence Score": "ML confidence",
        "Repo Age (log days)": "log_age",
        "Commit Count (log)": "log_commits",
        "Number of Topic Tags": "n_concepts",
        "Activity Intensity (log commits/day)": "log_intensity",
        "README Length": "readme_length",
        "Files Count": "files_total_count",
    }
    ax_col1, ax_col2 = st.columns(2)
    with ax_col1:
        x_choice = st.selectbox("X-axis", list(ax_options.keys()), index=0)
    with ax_col2:
        y_choice = st.selectbox("Y-axis", list(ax_options.keys()), index=1)

    x_col = ax_options[x_choice]
    y_col = ax_options[y_choice]

    fig_explorer = px.scatter(
        filtered,
        x=x_col,
        y=y_col,
        size="commit_count_default_branch",
        size_max=30,
        color="true_label",
        hover_data={
            "Repository": True,
            "Description": True,
            "Language": True,
            "Rule-based prediction": True,
            "Statistical prediction": True,
            "ML prediction": True,
            "ML confidence": ":.2f",
            "Methods agreeing (of 3)": True,
        },
        color_discrete_map={"Hackathon": HACKATHON_COLOR, "Non-Hackathon": NON_HACKATHON_COLOR},
        labels={
            x_col: x_choice,
            y_col: y_choice,
            "commit_count_default_branch": "Commits",
            "true_label": "True Label",
        },
        opacity=0.8,
        marginal_x="histogram",
        marginal_y="histogram",
    )
    fig_explorer.update_layout(
        height=650,
        plot_bgcolor="#fafcf8",
        paper_bgcolor="white",
        xaxis=dict(gridcolor="#e2e8f0"),
        yaxis=dict(gridcolor="#e2e8f0"),
        legend=dict(orientation="h", y=1.02, font=dict(size=13)),
    )
    st.plotly_chart(fig_explorer, use_container_width=True)

    st.markdown("---")

    # Methodology evolution timeline
    st.subheader("Research Journey")
    st.markdown(
        """
    | Phase | What We Did | Key Decision |
    |-------|-------------|--------------|
    | **Data Collection** | Built general-purpose tools to gather hackathon + GitHub data | Analyzed individual repositories rather than whole projects |
    | **Case Study** | Applied pipeline to LauzHack (2023–2025) as validation dataset | 213 hackathon repos + 274 account repos for contrast |
    | **Labeling** | 3 people manually checked repos + automatic matching via project links | Combined human judgment with automated matching for scale |
    | **Topic Tagging** | 🏷️ Used EPFL's knowledge database to tag repos with topics | Only tagged repos that hadn't been tagged yet, saving API calls |
    | **Feature Engineering** | Extracted numeric and boolean features per repo (activity, quality, topics) | Carefully excluded features that would let the model "cheat" |
    | **Predicting** | Compared three methods: keyword rules, statistical weighting, ML | Machine learning was the clear winner with balanced performance |
    """
    )


# ---------------------------------------------------------------------------
# Act 6: Live Prediction
# ---------------------------------------------------------------------------


def render_predict():
    st.markdown(
        """
    <h1 style='color:#26235c;'>Try It Yourself</h1>
    <p style='color:#64748b; font-size:1.05rem;'>
        Paste any public GitHub repository URL below and all three methods
        will classify it in real time.
    </p>
    """,
        unsafe_allow_html=True,
    )

    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/owner/repo",
        key="predict_url",
    )

    enrich_concepts = st.checkbox(
        "Enrich with EPFL concepts",
        value=False,
        help="Call the EPFL Graph API to extract semantic topics. "
             "Requires EPFL credentials in your .env file. "
             "If unavailable, prediction runs without concepts.",
    )

    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if not repo_url or not repo_url.strip():
            st.error("Please enter a GitHub URL first.")
            return
        if "github.com" not in repo_url:
            st.error("Please enter a valid GitHub URL (e.g. https://github.com/owner/repo)")
            return

        try:
            from hackathon_analysis.prediction.pipeline import (
                fetch_single_repo_metadata,
                _load_correlation_weights,
                _load_rf_model,
            )
            from hackathon_analysis.prediction.feature_engineering import build_repo_analysis_features
            from hackathon_analysis.prediction.naive_rules import predict_naive
            from hackathon_analysis.prediction.correlation_weighted import predict_correlation_weighted
            from hackathon_analysis.prediction.random_forest import predict_random_forest
            from hackathon_analysis.prediction.concepts import try_enrich_with_concepts

            status = st.status("Running prediction pipeline...", expanded=True)

            status.update(label="Step 1/5: Fetching GitHub metadata...")
            raw_df = fetch_single_repo_metadata(repo_url.strip())

            if enrich_concepts:
                status.update(label="Step 2/5: Enriching with EPFL concepts...")
                row_dict = raw_df.iloc[0].to_dict()
                concept_result = try_enrich_with_concepts(row_dict)
                raw_df["n_concepts"] = concept_result["n_concepts"]
                raw_df["repo_concept_names"] = [concept_result["repo_concept_names"]]
                raw_df["repo_top_concept"] = concept_result["repo_top_concept"]
                raw_df["repo_concepts_error"] = concept_result["repo_concepts_error"]

            status.update(label="Step 3/5: Computing features...")
            df = build_repo_analysis_features(raw_df)

            status.update(label="Step 4/5: Running rule-based & correlation predictions...")
            df = predict_naive(df)
            corr_weights = _load_correlation_weights()
            df = predict_correlation_weighted(df, corr_weights)

            status.update(label="Step 5/5: Running Random Forest prediction...")
            rf_model = _load_rf_model()
            df = predict_random_forest(df, rf_model)

            # Build result dict
            row = df.iloc[0]
            result = {
                "repo_url": repo_url.strip(),
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
                    "n_concepts": int(row.get("n_concepts", 0)),
                    "concepts_error": row.get("repo_concepts_error"),
                },
            }

            status.update(label="Prediction complete!", state="complete", expanded=False)
            st.session_state["predict_result"] = result
        except Exception as e:
            st.error(f"Failed to fetch repository: {e}")
            return

    # Show results if we have them (persists across rerenders)
    if "predict_result" not in st.session_state:
        return

    result = st.session_state["predict_result"]

    meta = result["metadata"]
    naive = result["naive_rule_based"]
    corr = result["correlation_weighted"]
    rf = result["random_forest"]

    st.markdown("---")

    # Repo info card
    st.markdown(
        f"""
    <div style='padding:16px; background:linear-gradient(135deg, #f8fafc, #eef0f8);
         border-radius:12px; border:1px solid #c8cde3; margin-bottom:1rem;'>
        <h3 style='margin:0; color:#26235c;'>{meta.get('owner', '')}/{meta.get('repo', '')}</h3>
        <p style='margin:6px 0 0 0; color:#64748b;'>
            Language: <b>{meta.get('language', 'N/A')}</b> &nbsp;|&nbsp;
            Stars: <b>{meta.get('stars', 0):,}</b> &nbsp;|&nbsp;
            Forks: <b>{meta.get('forks', 0):,}</b> &nbsp;|&nbsp;
            Commits: <b>{meta.get('commits', 0):,}</b> &nbsp;|&nbsp;
            Contributors: <b>{meta.get('contributors', 0)}</b> &nbsp;|&nbsp;
            Active days: <b>{meta.get('active_days', 0)}</b> &nbsp;|&nbsp;
            Repo age: <b>{meta.get('repo_age_days', 0):,} days</b> &nbsp;|&nbsp;
            Topics: <b>{meta.get('n_concepts', 0)}</b>
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Consensus
    votes = sum([naive["flag"], corr["flag"], rf["flag"]])
    if votes >= 2:
        verdict_color = SDSC_GREEN
        verdict_bg = "#f3f7ea"
        verdict_border = "#c5e085"
        verdict_text = "HACKATHON REPO"
        verdict_detail = f"{votes}/3 methods agree this is a hackathon repository"
    else:
        verdict_color = SDSC_BLUE
        verdict_bg = "#eef0f8"
        verdict_border = "#c8cde3"
        verdict_text = "NOT a hackathon repo"
        verdict_detail = f"Only {votes}/3 methods flagged this as hackathon"

    st.markdown(
        f"""
    <div style='padding:20px; background:{verdict_bg}; border-radius:12px;
         border:2px solid {verdict_border}; text-align:center; margin-bottom:1rem;'>
        <h2 style='margin:0; color:{verdict_color}; font-size:2rem;'>{verdict_text}</h2>
        <p style='margin:8px 0 0 0; color:#64748b;'>{verdict_detail}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Concept enrichment notice
    concepts_error = meta.get("concepts_error")
    n_concepts = meta.get("n_concepts", 0)
    if n_concepts > 0:
        st.caption(f"🏷️ Enriched with {n_concepts} EPFL concepts")
    elif enrich_concepts and concepts_error and concepts_error != "not_attempted":
        st.caption(f"🏷️ Concept enrichment attempted but failed: {concepts_error}")
    elif not enrich_concepts:
        st.caption("🏷️ Concept enrichment skipped — check the box above to enable")

    # Method-by-method results
    col1, col2, col3 = st.columns(3)

    with col1:
        _render_method_card(
            "1. Simple Keyword Rules",
            naive["flag"],
            f"Score: {naive['score']}",
            naive["signals"].replace("|", ", "),
            "#e8a838",
            "#fef9ee",
            "#fdf3dc",
        )

    with col2:
        _render_method_card(
            "2. Statistical Weighting",
            corr["flag"],
            f"Score: {corr['score']:.2f}",
            "Threshold: 0.5",
            SDSC_BLUE,
            "#eef0f8",
            "#e1e4f0",
        )

    with col3:
        _render_method_card(
            "3. Random Forest",
            rf["flag"],
            f"Probability: {rf['probability']:.2f}",
            "Threshold: 0.5",
            SDSC_GREEN,
            "#f3f7ea",
            "#e6edcc",
        )

    # Signal breakdown
    if naive["signals"] and naive["signals"] != "no_signal":
        st.markdown("---")
        st.subheader("Signal Breakdown")
        st.caption("Which rule-based signals fired for this repository:")
        for signal in naive["signals"].split("|"):
            signal_display = signal.replace("_", " ").title()
            st.markdown(f"- **{signal_display}**")


def _render_method_card(
    title: str,
    flag: bool,
    score_text: str,
    detail: str,
    accent_color: str,
    bg_start: str,
    bg_end: str,
):
    """Render a single method result card."""
    label = "HACKATHON" if flag else "Not hackathon"
    label_color = SDSC_GREEN if flag else SDSC_BLUE
    st.markdown(
        f"""
    <div style='padding:16px; background:linear-gradient(135deg, {bg_start}, {bg_end});
         border-radius:12px; border-left:4px solid {accent_color}; height:160px;'>
        <h4 style='margin:0 0 8px 0; color:#26235c; font-size:0.95rem;'>{title}</h4>
        <p style='margin:0; color:{label_color}; font-weight:700; font-size:1.3rem;'>{label}</p>
        <p style='margin:4px 0 0 0; color:#475569; font-size:0.9rem;'>{score_text}</p>
        <p style='margin:2px 0 0 0; color:#94a3b8; font-size:0.8rem;'>{detail}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main App Router
# ---------------------------------------------------------------------------


def main():
    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            """
        <div style='text-align:center; padding:16px 0 20px 0;'>
            <h2 style='color:#26235c; margin:0; font-size:1.6rem; letter-spacing:-0.02em;'>Hackalysis</h2>
            <p style='color:#94a3b8; font-size:0.82rem; margin:4px 0 0 0;'>Hackathon Repo Detection</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        section = st.radio(
            "Navigate the Story",
            [
                "1. The Question",
                "2. What Makes a Hackathon Repo?",
                "3. Three Prediction Methods",
                "4. The Verdict",
                "5. What We Learned",
                "6. Try It Yourself",
            ],
        )

        st.markdown("---")
        st.markdown(
            """
        <div style='color:#94a3b8; font-size:0.8rem;'>
            <b>Author:</b> Eisha Tir Raazia<br>
            <b>Case Study:</b> LauzHack 2023-2025<br>
            <b>Repos:</b> 487 analyzed<br>
            <b>Methods:</b> 3 compared<br>
            <span style='font-size:0.75rem;'>🏷️ = case-study-specific</span>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Load data
    df = load_predictions()

    # Route to section
    if section.startswith("1"):
        render_introduction(df)
    elif section.startswith("2"):
        render_features(df)
    elif section.startswith("3"):
        render_methods(df)
    elif section.startswith("4"):
        render_verdict(df)
    elif section.startswith("5"):
        render_conclusions(df)
    elif section.startswith("6"):
        render_predict()


if __name__ == "__main__":
    main()
