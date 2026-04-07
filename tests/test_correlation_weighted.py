"""Tests for prediction.correlation_weighted module."""

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.correlation_weighted import (
    CorrelationWeights,
    predict_correlation_weighted,
    train_correlation_weights,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_df():
    """Labeled DataFrame for training weights."""
    rng = np.random.RandomState(42)
    n = 100
    target = rng.choice([True, False], size=n, p=[0.4, 0.6])
    return pd.DataFrame({
        "true_hackathon_repos": target,
        "active_days": rng.randint(1, 100, n).astype(float),
        "commits": rng.randint(1, 200, n).astype(float),
        "stars": rng.randint(0, 50, n).astype(float),
    })


@pytest.fixture
def features():
    return ["active_days", "commits", "stars"]


@pytest.fixture
def trained_weights(training_df, features):
    return train_correlation_weights(training_df, features)


# ---------------------------------------------------------------------------
# Tests: CorrelationWeights serialization
# ---------------------------------------------------------------------------


def test_weights_roundtrip():
    """to_dict → from_dict should preserve all values."""
    original = CorrelationWeights(
        weights={"a": 0.5, "b": -0.3},
        feat_min={"a": 0.0, "b": 1.0},
        feat_max={"a": 10.0, "b": 5.0},
        threshold=0.6,
        score_min=-1.0,
        score_max=2.0,
    )
    restored = CorrelationWeights.from_dict(original.to_dict())
    assert restored.weights == original.weights
    assert restored.feat_min == original.feat_min
    assert restored.feat_max == original.feat_max
    assert restored.threshold == original.threshold
    assert restored.score_min == original.score_min
    assert restored.score_max == original.score_max


def test_weights_from_dict_defaults():
    """from_dict should use defaults for optional fields."""
    d = {"weights": {"a": 0.5}, "feat_min": {"a": 0.0}, "feat_max": {"a": 1.0}}
    w = CorrelationWeights.from_dict(d)
    assert w.threshold == 0.5
    assert w.score_min == 0.0
    assert w.score_max == 1.0


# ---------------------------------------------------------------------------
# Tests: train_correlation_weights
# ---------------------------------------------------------------------------


def test_train_returns_weights_for_all_features(trained_weights, features):
    for f in features:
        assert f in trained_weights.weights


def test_train_weights_are_correlations(training_df, features):
    """Weights should be Pearson correlations (between -1 and 1)."""
    w = train_correlation_weights(training_df, features)
    for val in w.weights.values():
        assert -1 <= val <= 1


def test_train_captures_min_max(trained_weights, training_df, features):
    """feat_min/max should reflect actual data range."""
    for f in features:
        assert trained_weights.feat_min[f] == pytest.approx(training_df[f].min())
        assert trained_weights.feat_max[f] == pytest.approx(training_df[f].max())


def test_train_score_range(trained_weights):
    """score_min should be less than score_max."""
    assert trained_weights.score_min < trained_weights.score_max


# ---------------------------------------------------------------------------
# Tests: predict_correlation_weighted
# ---------------------------------------------------------------------------


def test_predict_adds_columns(training_df, trained_weights):
    result = predict_correlation_weighted(training_df, trained_weights)
    assert "corr_weighted_score" in result.columns
    assert "corr_weighted_flag" in result.columns


def test_predict_scores_between_0_and_1(training_df, trained_weights):
    result = predict_correlation_weighted(training_df, trained_weights)
    scores = result["corr_weighted_score"]
    assert (scores >= 0).all()
    assert (scores <= 1).all()


def test_predict_flag_matches_threshold(training_df, trained_weights):
    result = predict_correlation_weighted(training_df, trained_weights)
    for _, row in result.iterrows():
        expected = row["corr_weighted_score"] >= trained_weights.threshold
        assert row["corr_weighted_flag"] == expected


def test_predict_does_not_modify_input(training_df, trained_weights):
    original_cols = set(training_df.columns)
    predict_correlation_weighted(training_df, trained_weights)
    assert set(training_df.columns) == original_cols


def test_predict_handles_missing_features(trained_weights):
    """If a feature from training is missing, prediction still works."""
    # Only has 'stars', missing 'active_days' and 'commits'
    df = pd.DataFrame({"stars": [10.0, 20.0, 0.0]})
    result = predict_correlation_weighted(df, trained_weights)
    assert len(result) == 3
    assert "corr_weighted_score" in result.columns


def test_predict_custom_threshold(training_df, features):
    """Different thresholds should produce different flag counts."""
    w_low = train_correlation_weights(training_df, features, threshold=0.1)
    w_high = train_correlation_weights(training_df, features, threshold=0.9)

    result_low = predict_correlation_weighted(training_df, w_low)
    result_high = predict_correlation_weighted(training_df, w_high)

    # Lower threshold → more flagged as hackathon
    assert result_low["corr_weighted_flag"].sum() >= result_high["corr_weighted_flag"].sum()


def test_predict_single_repo(trained_weights):
    """Should work on a single-row DataFrame."""
    df = pd.DataFrame([{"active_days": 2.0, "commits": 15.0, "stars": 1.0}])
    result = predict_correlation_weighted(df, trained_weights)
    assert len(result) == 1
    assert 0 <= result.iloc[0]["corr_weighted_score"] <= 1


def test_predict_with_nan_values(trained_weights):
    """NaN feature values should be treated as 0."""
    df = pd.DataFrame([{"active_days": float("nan"), "commits": 10.0, "stars": 5.0}])
    result = predict_correlation_weighted(df, trained_weights)
    assert not pd.isna(result.iloc[0]["corr_weighted_score"])
