# AGENTS.md

Instructions for AI coding agents working in this repository.

## Build & Test Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_naive_rules.py -v

# Run Streamlit dashboard
uv run streamlit run app/hackathon_story.py

# CLI: predict a repo
uv run hackalysis predict "https://github.com/user/repo"

# CLI: retrain models
uv run hackalysis train-model
```

## Project Structure

```
src/hackathon_analysis/
  data_extraction/       # Scraping (LauzHack, Devpost) + GitHub API client
    models.py            # Pydantic v2 schemas (GitHubRepoMetadata, etc.)
    github_extractor.py  # GitHub REST/GraphQL client with retry logic
    config.py            # Hackathon provider configs
  data_enrichment/       # EPFL concept enrichment (optional)
  data_analysis/         # Notebooks (do not modify without permission, and report any issue if spotted during development)
  prediction/            # Prediction pipeline
    feature_engineering.py   # Derived features from raw metadata
    naive_rules.py           # Method 1: keyword + stats rules
    correlation_weighted.py  # Method 2: correlation-weighted scoring
    random_forest.py         # Method 3: Random Forest classifier
    concepts.py              # Optional EPFL concept enrichment
    pipeline.py              # Main entry: URL -> prediction result
    train_model.py           # Train and save model artifacts
  models/                # Saved model artifacts (features.json, rf_model.pkl, etc.)
  cli.py                 # Typer CLI commands
app/
  hackathon_story.py     # Streamlit dashboard (single-file app)
  .streamlit/config.toml # Streamlit theme and server config
notebooks/               # Analysis notebooks (only contains exploratory notebooks)
tests/                   # pytest test suite
```

## Key Conventions

- Data models use **Pydantic v2** with validators — see `data_extraction/models.py`
- Prediction modules follow a **train/predict pattern**: `train_*()` learns from labeled data, `predict_*()` applies to new data
- Feature engineering always goes through `build_repo_analysis_features()` then `select_features()`
- Tests use **pytest** with fixtures — follow the existing pattern in `tests/`
- The Streamlit app uses SDSC brand colors defined as constants at the top of `hackathon_story.py`
- Column name variants exist (`topics`/`repo_topics`, `error`/`repo_error`) — handle both with fallbacks

## Known Limitations

- **Devpost extraction is not implemented** — `data_extraction/devpost.py` is a stub. Ignore Devpost references in CLI commands and configs for now.

## Things to Avoid

- **Do not modify notebooks** — extract reusable code into Python modules instead
- **Do not hardcode absolute paths** — use the `DATA_ROOT` environment variable
- **Do not commit credentials** — `.env` files are gitignored
- **Do not add `StandardScaler` before Random Forest** — tree models don't need feature scaling
- **Do not add concept one-hot features to ML models** — they're too specific to one hackathon and don't generalize
- **Do not use `st.metric` delta for non-change values** — the delta parameter shows green/red arrows meant for changes

## Environment

- **Python**: 3.13
- **Package manager**: uv (dependencies in `pyproject.toml`)
- **Required env vars**: `DATA_ROOT`, `GITHUB_TOKEN` (in `.env` file)
- **Optional env vars**: `EPFL_GRAPH_API_HOST`, `EPFL_GRAPH_API_PORT`, `EPFL_GRAPH_API_USER`, `EPFL_GRAPH_API_PASSWORD`
- **Data directory**: configured via `DATA_ROOT`, contains per-year folders (`lauzhack-2023/`, etc.) and account repo folders (`github-account-*/`)
