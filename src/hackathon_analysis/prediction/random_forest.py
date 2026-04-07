"""Method 3: Random Forest hackathon repo classification.

Extracted from repo_analysis.ipynb cell 63.
Trains a Random Forest classifier on labeled data and produces
out-of-fold predictions via stratified k-fold cross-validation.

For production use, the model is trained once on all labeled data,
saved to disk, and loaded at prediction time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


@dataclass
class TrainedRandomForest:
    """A trained Random Forest model with its metadata.

    Attributes:
        model: the fitted sklearn RandomForestClassifier.
        features: ordered list of feature names the model expects.
        importances: feature_name → importance score.
    """

    model: RandomForestClassifier
    features: list[str]
    importances: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save the trained model to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "features": self.features,
                    "importances": self.importances,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path) -> TrainedRandomForest:
        """Load a trained model from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        return cls(
            model=data["model"],
            features=data["features"],
            importances=data.get("importances", {}),
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_random_forest(
    df: pd.DataFrame,
    features: list[str],
    target: str = "true_hackathon_repos",
    n_estimators: int = 100,
    max_depth: int = 8,
    min_samples_leaf: int = 5,
    random_state: int = 42,
) -> TrainedRandomForest:
    """Train a Random Forest on all labeled data.

    Args:
        df: DataFrame with feature columns and target column.
        features: ordered list of feature column names.
        target: name of the boolean target column.

    Returns:
        TrainedRandomForest ready for prediction or saving.
    """
    X = _build_feature_matrix(df, features)
    y = df[target].fillna(False).astype(int).values

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(X, y)

    importances = dict(zip(features, clf.feature_importances_))

    return TrainedRandomForest(
        model=clf,
        features=features,
        importances=importances,
    )


def evaluate_cross_validated(
    df: pd.DataFrame,
    features: list[str],
    target: str = "true_hackathon_repos",
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run stratified k-fold CV and return out-of-fold predictions.

    Every row gets a prediction from a model that never trained on it.
    Use this for honest evaluation — NOT for production prediction.

    Adds columns:
    - ml_predicted_flag (int, 0 or 1)
    - ml_predicted_proba (float, probability of hackathon)
    """
    out = df.copy()
    X = _build_feature_matrix(df, features)
    y = df[target].fillna(False).astype(int).values

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    out["ml_predicted_flag"] = cross_val_predict(clf, X, y, cv=cv, method="predict")
    out["ml_predicted_proba"] = cross_val_predict(
        clf, X, y, cv=cv, method="predict_proba"
    )[:, 1]

    return out


# ---------------------------------------------------------------------------
# Prediction: apply a saved model to new data
# ---------------------------------------------------------------------------


def predict_random_forest(
    df: pd.DataFrame,
    trained: TrainedRandomForest,
) -> pd.DataFrame:
    """Classify repos using a pre-trained Random Forest.

    Args:
        df: DataFrame with the same feature columns used during training.
        trained: model from train_random_forest() or TrainedRandomForest.load().

    Adds columns:
    - ml_predicted_flag (int, 0 or 1)
    - ml_predicted_proba (float, probability of hackathon)
    """
    out = df.copy()
    X = _build_feature_matrix(df, trained.features)

    out["ml_predicted_flag"] = trained.model.predict(X)
    out["ml_predicted_proba"] = trained.model.predict_proba(X)[:, 1]

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Build a numeric feature matrix, filling missing columns with 0."""
    feature_df = pd.DataFrame(index=df.index)
    for col in features:
        if col in df.columns:
            feature_df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            feature_df[col] = 0.0
    return feature_df.values
