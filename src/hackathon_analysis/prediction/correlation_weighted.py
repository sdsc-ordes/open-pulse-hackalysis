"""Method 2: Correlation-Weighted hackathon repo scoring.

Extracted from repo_analysis.ipynb cell 60.
Each feature is weighted by its Pearson correlation with the target.
Features are normalized to [0, 1], then combined into a single score.

Unlike the naive rules, this method requires training data to compute
the correlation weights and normalization parameters. These are saved
as a dict and loaded at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CorrelationWeights:
    """Pre-computed weights and normalization params from training data.

    Attributes:
        weights: feature_name → signed Pearson correlation with target.
        feat_min: feature_name → min value seen in training data.
        feat_max: feature_name → max value seen in training data.
        threshold: score above this → predicted as hackathon.
        score_min: min raw score seen in training data (for normalization).
        score_max: max raw score seen in training data (for normalization).
    """

    weights: dict[str, float]
    feat_min: dict[str, float]
    feat_max: dict[str, float]
    threshold: float = 0.5
    score_min: float = 0.0
    score_max: float = 1.0

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "feat_min": self.feat_min,
            "feat_max": self.feat_max,
            "threshold": self.threshold,
            "score_min": self.score_min,
            "score_max": self.score_max,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CorrelationWeights:
        return cls(
            weights=d["weights"],
            feat_min=d["feat_min"],
            feat_max=d["feat_max"],
            threshold=d.get("threshold", 0.5),
            score_min=d.get("score_min", 0.0),
            score_max=d.get("score_max", 1.0),
        )


# ---------------------------------------------------------------------------
# Training: compute weights from labeled data
# ---------------------------------------------------------------------------


def train_correlation_weights(
    df: pd.DataFrame,
    features: list[str],
    target: str = "true_hackathon_repos",
    threshold: float = 0.5,
) -> CorrelationWeights:
    """Compute correlation weights and normalization params from training data.

    Args:
        df: DataFrame with feature columns and target column.
        features: list of feature column names to use.
        target: name of the boolean target column.
        threshold: score threshold for classification.

    Returns:
        CorrelationWeights with everything needed for prediction.
    """
    feature_df = df[features].copy()
    for col in features:
        feature_df[col] = feature_df[col].astype(float)
    feature_df = feature_df.fillna(0)

    target_series = df[target].fillna(False).astype(float)
    correlations = feature_df.corrwith(target_series, method="pearson").dropna()

    feat_min = feature_df[correlations.index].min()
    feat_max = feature_df[correlations.index].max()

    # Compute raw score range for normalization
    feat_range = (feat_max - feat_min).replace(0, 1)
    normalized = (feature_df[correlations.index] - feat_min) / feat_range
    raw_scores = normalized.mul(correlations, axis=1).sum(axis=1)

    return CorrelationWeights(
        weights=correlations.to_dict(),
        feat_min=feat_min.to_dict(),
        feat_max=feat_max.to_dict(),
        threshold=threshold,
        score_min=float(raw_scores.min()),
        score_max=float(raw_scores.max()),
    )


# ---------------------------------------------------------------------------
# Prediction: apply saved weights to new data
# ---------------------------------------------------------------------------


def predict_correlation_weighted(
    df: pd.DataFrame,
    trained_weights: CorrelationWeights,
) -> pd.DataFrame:
    """Score repos using pre-computed correlation weights.

    Args:
        df: DataFrame with the same feature columns used during training.
        trained_weights: weights from train_correlation_weights().

    Adds columns:
    - corr_weighted_score (float, 0-1 normalized)
    - corr_weighted_flag (bool)
    """
    out = df.copy()
    w = trained_weights

    # Use only features that exist in both the data and the saved weights
    available = [f for f in w.weights if f in df.columns]

    feature_df = out[available].copy()
    for col in available:
        feature_df[col] = feature_df[col].astype(float)
    feature_df = feature_df.fillna(0)

    # Normalize features to [0, 1] using training min/max
    for col in available:
        col_min = w.feat_min[col]
        col_range = w.feat_max[col] - col_min
        if col_range == 0:
            col_range = 1
        feature_df[col] = (feature_df[col] - col_min) / col_range

    # Weighted score
    weights_series = pd.Series({c: w.weights[c] for c in available})
    raw_scores = feature_df.mul(weights_series, axis=1).sum(axis=1)

    # Normalize to [0, 1] using training score range
    score_range = w.score_max - w.score_min
    if score_range == 0:
        score_range = 1
    normalized = (raw_scores - w.score_min) / score_range
    normalized = normalized.clip(0, 1)

    out["corr_weighted_score"] = normalized.values
    out["corr_weighted_flag"] = (normalized.values >= w.threshold).astype(bool)
    return out
