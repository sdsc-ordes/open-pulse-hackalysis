"""One-time training script: trains all models and saves artifacts.

Run this once on your labeled dataset to produce the model files
that the prediction pipeline needs. Re-run whenever you add new
labeled data and want to update the models.

Usage:
    python -m hackathon_analysis.prediction.train_model

Or from the repo root:
    .venv/bin/python -m hackathon_analysis.prediction.train_model
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from hackathon_analysis.common_utils import upload_to_hugging_face
from hackathon_analysis.prediction.correlation_weighted import (
    train_correlation_weights,
)
from hackathon_analysis.prediction.feature_engineering import (
    build_repo_analysis_features,
    select_features,
)
from hackathon_analysis.prediction.random_forest import (
    evaluate_cross_validated,
    train_random_forest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Default paths
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_CSV = "repo_metadata_with_predictions.csv"


def train_and_save(
    data_path: Path,
    models_dir: Path = MODELS_DIR,
    target: str = "true_hackathon_repos",
) -> dict:
    """Train all models on labeled data and save artifacts.

    Args:
        data_path: path to the CSV with labeled repo data.
        models_dir: directory to save model files into.
        target: name of the boolean target column.

    Returns:
        dict with training summary (feature count, metrics, paths).
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    log.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)
    log.info("Loaded %d repos", len(df))

    # --- Feature engineering ---
    log.info("Building features...")
    df = build_repo_analysis_features(df)

    valid = df[df["has_valid_metadata"]].copy()
    log.info("Valid repos for training: %d", len(valid))

    if target not in valid.columns:
        raise ValueError(
            f"Target column '{target}' not found. "
            f"Available columns: {sorted(valid.columns)}"
        )

    # --- Feature selection ---
    features = select_features(valid, target=target)
    log.info("Selected %d features: %s", len(features), features)

    # --- Save feature list ---
    features_path = models_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(features, f, indent=2)
    log.info("Saved feature list to %s", features_path)

    # --- Method 2: Correlation Weights ---
    log.info("Training correlation weights...")
    corr_weights = train_correlation_weights(valid, features, target=target)
    corr_path = models_dir / "correlation_weights.json"
    with open(corr_path, "w") as f:
        json.dump(corr_weights.to_dict(), f, indent=2)
    log.info("Saved correlation weights to %s", corr_path)

    # --- Method 3: Random Forest ---
    log.info("Training Random Forest on all data...")
    trained_rf = train_random_forest(valid, features, target=target)
    rf_path = models_dir / "rf_model.pkl"
    trained_rf.save(rf_path)
    log.info("Saved Random Forest to %s", rf_path)

    # --- Cross-validated evaluation ---
    log.info("Running 5-fold cross-validation for evaluation...")
    cv_result = evaluate_cross_validated(valid, features, target=target)
    y_true = valid[target].astype(int)
    y_pred_cv = cv_result["ml_predicted_flag"]
    cv_accuracy = (y_true == y_pred_cv).mean()
    log.info("Cross-validated accuracy: %.1f%%", cv_accuracy * 100)

    # --- Top feature importances ---
    top_features = sorted(
        trained_rf.importances.items(), key=lambda x: -x[1]
    )[:10]
    log.info("Top 10 feature importances:")
    for feat, imp in top_features:
        log.info("  %-35s %.4f", feat, imp)

    # --- Save metadata ---
    metadata = {
        "data_path": str(data_path),
        "total_repos": len(df),
        "valid_repos": len(valid),
        "n_features": len(features),
        "features": features,
        "cv_accuracy": round(cv_accuracy, 4),
        "top_importances": {f: round(v, 4) for f, v in top_features},
        "correlation_weights_path": str(corr_path),
        "rf_model_path": str(rf_path),
    }
    metadata_path = models_dir / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved metadata to %s", metadata_path)

    # --- Upload models to Hugging Face ---
    upload_to_hugging_face(models_dir, path_in_repo="models/")

    log.info("Training complete! Models saved to %s", models_dir)
    return metadata


def main():
    load_dotenv()
    data_root = os.getenv("DATA_ROOT")
    if not data_root:
        raise RuntimeError("DATA_ROOT environment variable not set. Check your .env file.")

    data_path = Path(data_root) / DEFAULT_CSV

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Run the notebook to generate it first."
        )

    train_and_save(data_path)


if __name__ == "__main__":
    main()
