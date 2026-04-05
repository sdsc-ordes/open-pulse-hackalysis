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
def load_predictions() -> pd.DataFrame:
    df = pd.read_csv(DATA_ROOT / "repo_metadata_with_predictions.csv")

    # Parse stringified lists
    for col in ["concept_list", "repo_concept_names", "concept_project_freq_buckets", "topics"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_parse_list)

    # Derive year from per-year metadata JSONs
    year_map = {}
    for year in [2023, 2024, 2025]:
        meta_path = DATA_ROOT / f"lauzhack-{year}" / "lauzhack_github_repo_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                urls = list(json.load(f).keys())
            for url in urls:
                year_map[url] = year
    df["year"] = df["repo_url"].map(year_map).apply(lambda x: str(int(x)) if pd.notna(x) else "Account Repos")

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
        We built a pipeline to identify <b>true hackathon repos</b> &mdash; and the answer
        required more than just looking at commit dates.
    </p>
    <p style='text-align:center; color:#94a3b8; font-size:0.9rem; margin-top:8px;'>
        Navigate the story using the sidebar on the left.
    </p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # Metric cards
    valid = df[df["has_valid_metadata"]]
    n_projects = df["project_foreign_key"].dropna().nunique()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hackathon Projects", f"{n_projects}", help="Unique projects from LauzHack 2023, 2024, and 2025")
    c2.metric("GitHub Repos", f"{len(df)}", help=f"{len(df)} total linked repositories, {len(valid)} with complete GitHub metadata")
    hackathon_years = sorted(y for y in df["year"].unique() if y != "Account Repos")
    c3.metric("Years Analyzed", str(len(hackathon_years)), help=f"{', '.join(hackathon_years)} editions of the LauzHack hackathon")
    unique_concepts = df["concept_list"].explode().nunique()
    c4.metric(
        "Topic Tags",
        f"{unique_concepts:,}",
        help="Topics (like 'machine learning' or 'web development') automatically assigned to each "
        "repository by EPFL's knowledge database. Think of them as smart tags describing what a project is about.",
    )

    st.markdown("---")

    # Dataset composition
    st.subheader("Dataset Composition")
    st.caption(
        "Not every repository is usable or represents hackathon code. These bars show how many "
        "repositories have each attribute. Note: these are **independent properties**, not "
        "sequential filters — for example, a repo can have topic tags but lack complete GitHub metadata."
    )

    total = len(df)
    with_metadata = int(df["has_valid_metadata"].sum())
    linked = int(df["is_linked_project"].sum())
    true_hack = int(df["true_hackathon_repos"].sum())
    with_concepts = int((df["n_concepts"] > 0).sum())

    composition_data = pd.DataFrame(
        {
            "Attribute": [
                "All Repositories Collected",
                "Have Complete GitHub Info",
                "Have Topic Tags",
                "Linked to a Hackathon Project Page",
                "Confirmed Hackathon Code",
            ],
            "Count": [total, with_metadata, with_concepts, linked, true_hack],
        }
    )

    fig_composition = px.bar(
        composition_data,
        x="Count",
        y="Attribute",
        orientation="h",
        color_discrete_sequence=[SDSC_BLUE],
        text="Count",
    )
    fig_composition.update_layout(
        height=320, margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        yaxis=dict(categoryorder="array", categoryarray=list(reversed(composition_data["Attribute"]))),
        xaxis_title="Number of Repositories",
        yaxis_title="",
    )
    fig_composition.update_traces(textposition="outside", textfont_size=14)
    st.plotly_chart(fig_composition, use_container_width=True)
    st.caption(
        f"*{true_hack - linked} repos are confirmed as hackathon code via manual labeling "
        f"even though they are not directly linked to a hackathon project page.*"
    )

    # Year breakdown + Language treemap side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Repositories by Source")
        st.caption(
            "Each bar shows repos from a LauzHack edition. 'Account Repos' are other repositories "
            "from participants' GitHub accounts, included as non-hackathon examples for comparison."
        )
        year_counts = df["year"].value_counts().reset_index()
        year_counts.columns = ["Source", "Repos"]
        year_counts = year_counts.sort_values("Source")

        fig_year = px.bar(
            year_counts,
            x="Source",
            y="Repos",
            color="Source",
            color_discrete_map={
                "2023": SDSC_GREEN,
                "2024": SDSC_BLUE,
                "2025": SDSC_NAVY,
                "Account Repos": "#e8a838",
            },
            text="Repos",
        )
        fig_year.update_layout(
            height=400,
            showlegend=False,
            xaxis_title="",
            yaxis_title="Number of Repositories",
        )
        fig_year.update_traces(textposition="outside")
        st.plotly_chart(fig_year, use_container_width=True)

    with col_right:
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
        fig_lang.update_layout(height=400, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="white")
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
        different from regular ones. From the raw data, we extracted <b>numeric and boolean features</b>
        for each repository &mdash; things like how many days it was active, how long the README is,
        whether it has tests, and what topics it covers. We then enriched repos with smart topic tags
        from the EPFL Graph API.
    </p>
    """,
        unsafe_allow_html=True,
    )

    valid = df[df["has_valid_metadata"]].copy()
    valid["label"] = valid["true_hackathon_repos"].map({True: "Hackathon", False: "Non-Hackathon"})

    st.markdown("---")

    # Violin plots for key numeric features
    st.subheader("Distribution of Key Features")
    st.caption(
        "Each shape shows how repositories are spread across a range of values. "
        "The wider the shape, the more repos have that value. The box shows the typical range "
        "(with the median line), and the dashed line shows the mean. Look for where green and blue "
        "shapes DON'T overlap — that's where the two groups differ most."
    )

    violin_features = [
        ("active_days_default_branch", "Days with Code Changes"),
        ("repo_age_days", "Repository Age (Days)"),
        ("readme_length", "README Length (chars)"),
        ("commit_count_default_branch", "Number of Commits"),
    ]

    fig_violins = make_subplots(
        rows=2, cols=2,
        subplot_titles=[t for _, t in violin_features],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for idx, (col, title) in enumerate(violin_features):
        row, col_idx = idx // 2 + 1, idx % 2 + 1
        for label, color in [("Hackathon", HACKATHON_COLOR), ("Non-Hackathon", NON_HACKATHON_COLOR)]:
            subset = valid[valid["label"] == label][col].dropna()
            subset = subset.clip(lower=0)  # sanitize: no negative ages/lengths
            fig_violins.add_trace(
                go.Violin(
                    y=subset,
                    name=label,
                    legendgroup=label,
                    showlegend=(idx == 0),
                    marker_color=color,
                    fillcolor=color,
                    line_color=color,
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.75,
                    spanmode="soft",
                ),
                row=row,
                col=col_idx,
            )

    fig_violins.update_layout(
        height=620,
        violinmode="group",
        legend=dict(orientation="h", y=1.06, font=dict(size=13)),
        plot_bgcolor="#fafcf8",
        paper_bgcolor="white",
    )
    # Add subtle gridlines
    fig_violins.update_yaxes(gridcolor="#e2e8f0", gridwidth=1, zeroline=False)
    st.plotly_chart(fig_violins, use_container_width=True)

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
        st.subheader("Topic Tag Density")
        st.caption(
            "How many topic tags each repository received. Hackathon repos (green) tend to "
            "cluster around fewer topics because they focus on a single prototype, while "
            "non-hackathon repos (blue) often span more topics."
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
    st.subheader("Topics That Best Distinguish Hackathon from Non-Hackathon Repos")
    st.caption(
        "We used a statistical test (Fisher's exact test) to find which topics appear far more often "
        "in one group than the other. Bars pointing **right** (green) are topics strongly associated with "
        "hackathon repos. Bars pointing **left** (blue) signal non-hackathon repos. Longer bars = stronger "
        "association. A value of +2 means the odds of appearing in a hackathon repo are roughly 4× higher."
    )
    st.caption(
        "*P-values have been corrected for multiple comparisons using Benjamini-Hochberg FDR correction. "
        "Only topics that remain statistically significant after correction are shown with full confidence.*"
    )

    fisher_df = compute_fisher_stats(df)
    # Show only FDR-significant concepts; fall back to top by importance if too few
    significant = fisher_df[fisher_df["significant_fdr"]] if "significant_fdr" in fisher_df.columns else fisher_df
    top_n = 25
    n_half = top_n // 2
    top_concepts = pd.concat([significant.head(n_half + 1), significant.tail(n_half)])
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

    # Key takeaways
    st.success(
        "**Hackathon repos have a distinctive temporal fingerprint.** "
        "Burst activity in 1-2 days, then silence. The number of days with code changes "
        "is the single strongest signal separating hackathon repos from regular ones."
    )
    st.info(
        f"**Topic tags add useful signal beyond code structure.** "
        f"The EPFL knowledge database identified {unique_concepts:,}+ topics. A statistical test revealed "
        f"which topics are strongly linked to hackathon vs. production code."
    )
    st.warning(
        f"**Simple rules achieve high precision but miss nuanced cases.** "
        f"The keyword approach would miss {1 - naive_metrics['Recall']:.0%} of actual hackathon repos. "
        f"Machine learning captures the complex reality."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
        <div style='padding:20px; background:linear-gradient(135deg, #f3f7ea, #e6edcc); border-radius:12px; border:1px solid #c5e085;'>
            <h3 style='color:#5a8020; margin-top:0;'>What Worked</h3>
            <ul style='color:#475569;'>
                <li>The ML model correctly classified <b>{rf_metrics['Accuracy']:.1%}</b> of repos, with an F1 score of <b>{rf_metrics['F1']:.1%}</b></li>
                <li>Adding topic tags from EPFL gave the model information it couldn't get from code structure alone</li>
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
                <li>Using individual topic tags as yes/no features made the model too specific to LauzHack</li>
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
        - **Single hackathon** — All data comes from LauzHack. The model might not generalize to hackathons with different tech stacks, team sizes, or durations (e.g., Devpost, MLH events).
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
    | **Data Collection** | Built tools to automatically gather data from LauzHack and GitHub | Analyzed individual repositories rather than whole projects |
    | **Labeling** | 3 people manually checked repos + automatic matching via project links | Combined human judgment with automated matching for scale |
    | **Topic Tagging** | Used EPFL's knowledge database to tag repos with topics | Only tagged repos that hadn't been tagged yet, saving API calls |
    | **Feature Engineering** | Extracted numeric and boolean features per repo (activity, quality, topics) | Carefully excluded features that would let the model "cheat" |
    | **Predicting** | Compared three methods: keyword rules, statistical weighting, ML | Machine learning was the clear winner with balanced performance |
    """
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
            ],
        )

        st.markdown("---")
        st.markdown(
            """
        <div style='color:#94a3b8; font-size:0.8rem;'>
            <b>Author:</b> Eisha Tir Raazia<br>
            <b>Data:</b> LauzHack 2023-2025<br>
            <b>Repos:</b> 487 analyzed<br>
            <b>Methods:</b> 3 compared
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


if __name__ == "__main__":
    main()
