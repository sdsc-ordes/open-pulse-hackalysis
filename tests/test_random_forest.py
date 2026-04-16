"""Tests for prediction.random_forest module."""

import numpy as np
import pandas as pd
import pytest

from hackathon_analysis.prediction.random_forest import (
    TrainedRandomForest,
    evaluate_cross_validated,
    predict_random_forest,
    train_random_forest,
    _build_feature_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def training_df():
    """Labeled DataFrame with enough rows for 5-fold CV."""
    rng = np.random.RandomState(42)
    n = 100
    # Make features loosely correlated with target (not perfectly separable)
    target = rng.choice([True, False], size=n, p=[0.4, 0.6])
    active_days = np.where(target, rng.randint(1, 30, n), rng.randint(5, 300, n))
    commits = np.where(target, rng.randint(3, 50, n), rng.randint(10, 500, n))
    stars = rng.randint(0, 20, n)
    return pd.DataFrame({
        "true_hackathon_repos": target,
        "active_days": active_days.astype(float),
        "commits": commits.astype(float),
        "stars": stars.astype(float),
    })


@pytest.fixture
def features():
    return ["active_days", "commits", "stars"]


@pytest.fixture
def trained_model(training_df, features):
    return train_random_forest(training_df, features)


# ---------------------------------------------------------------------------
# Tests: _build_feature_matrix
# ---------------------------------------------------------------------------


def test_build_feature_matrix_shape(training_df, features):
    X = _build_feature_matrix(training_df, features)
    assert X.shape == (len(training_df), len(features))


def test_build_feature_matrix_missing_column():
    """Missing columns should be filled with 0."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    X = _build_feature_matrix(df, ["a", "b", "missing_col"])
    assert X.shape == (2, 3)
    assert X[0, 2] == 0.0
    assert X[1, 2] == 0.0


def test_build_feature_matrix_handles_nan():
    df = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
    X = _build_feature_matrix(df, ["a"])
    assert X[1, 0] == 0.0


# ---------------------------------------------------------------------------
# Tests: train_random_forest
# ---------------------------------------------------------------------------


def test_train_returns_correct_type(trained_model):
    assert isinstance(trained_model, TrainedRandomForest)
    assert isinstance(trained_model.model, object)


def test_train_stores_features(trained_model, features):
    assert trained_model.features == features


def test_train_computes_importances(trained_model, features):
    assert len(trained_model.importances) == len(features)
    for f in features:
        assert f in trained_model.importances
        assert 0 <= trained_model.importances[f] <= 1


def test_train_importances_sum_to_one(trained_model):
    total = sum(trained_model.importances.values())
    assert total == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# Tests: predict_random_forest
# ---------------------------------------------------------------------------


def test_predict_adds_columns(training_df, trained_model):
    result = predict_random_forest(training_df, trained_model)
    assert "ml_predicted_flag" in result.columns
    assert "ml_predicted_proba" in result.columns


def test_predict_flag_values(training_df, trained_model):
    result = predict_random_forest(training_df, trained_model)
    assert set(result["ml_predicted_flag"].unique()).issubset({0, 1})


def test_predict_proba_range(training_df, trained_model):
    result = predict_random_forest(training_df, trained_model)
    assert (result["ml_predicted_proba"] >= 0).all()
    assert (result["ml_predicted_proba"] <= 1).all()


def test_predict_does_not_modify_input(training_df, trained_model):
    original_cols = set(training_df.columns)
    predict_random_forest(training_df, trained_model)
    assert set(training_df.columns) == original_cols


def test_predict_single_repo(trained_model):
    """Should work on a single-row DataFrame."""
    df = pd.DataFrame([{"active_days": 2.0, "commits": 15.0, "stars": 1.0}])
    result = predict_random_forest(df, trained_model)
    assert len(result) == 1
    assert result.iloc[0]["ml_predicted_flag"] in (0, 1)


def test_predict_handles_missing_feature(trained_model):
    """Missing features should not crash — filled with 0."""
    df = pd.DataFrame([{"active_days": 2.0, "stars": 1.0}])  # missing 'commits'
    result = predict_random_forest(df, trained_model)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: evaluate_cross_validated
# ---------------------------------------------------------------------------


def test_cv_adds_columns(training_df, features):
    result = evaluate_cross_validated(training_df, features)
    assert "ml_predicted_flag" in result.columns
    assert "ml_predicted_proba" in result.columns


def test_cv_predictions_are_out_of_fold(training_df, features):
    """CV predictions should not be trivially perfect (would indicate leakage)."""
    result = evaluate_cross_validated(training_df, features)
    y_true = training_df["true_hackathon_repos"].astype(int)
    y_pred = result["ml_predicted_flag"]
    # Should have SOME errors (not 100% accuracy from overfitting)
    accuracy = (y_true == y_pred).mean()
    assert accuracy < 1.0, "Perfect accuracy suggests data leakage"
    # But should be better than random (50%)
    assert accuracy > 0.5, "Worse than random suggests a bug"


# ---------------------------------------------------------------------------
# Tests: save / load
# ---------------------------------------------------------------------------


def test_save_and_load_roundtrip(trained_model, training_df, tmp_path):
    """Save → load should produce identical predictions."""
    model_path = tmp_path / "test_model.pkl"
    trained_model.save(model_path)

    loaded = TrainedRandomForest.load(model_path)
    assert loaded.features == trained_model.features
    assert loaded.importances == pytest.approx(trained_model.importances)

    # Predictions should be identical
    result_original = predict_random_forest(training_df, trained_model)
    result_loaded = predict_random_forest(training_df, loaded)

    assert (result_original["ml_predicted_flag"] == result_loaded["ml_predicted_flag"]).all()
    assert np.allclose(
        result_original["ml_predicted_proba"],
        result_loaded["ml_predicted_proba"],
    )


def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        TrainedRandomForest.load("/nonexistent/model.pkl")
