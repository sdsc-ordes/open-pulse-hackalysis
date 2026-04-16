<p align="center">
  <img src="./docs/assets/logo.svg" alt="project logo" width="250">
</p>

<h1 align="center">
  open-pulse-hackathon-analysis
</h1>
<p align="center">
</p>

[![Current Release](https://img.shields.io/github/release/sdsc-ordes/open-pulse-hackathon-analysis.svg?label=release)](https://github.com/sdsc-ordes/open-pulse-hackathon-analysis/releases/latest)
[![Pipeline Status](https://img.shields.io/github/actions/workflow/status/sdsc-ordes/open-pulse-hackathon-analysis/normal.yaml?label=ci)](https://github.com/sdsc-ordes/open-pulse-hackathon-analysis/actions/workflows/normal.yaml)

**Authors:**

- [Eisha Tir Raazia](mailto:eisha.raazia@epfl.ch)

## Overview

This project analyses hackathon and open source repositories to understand what distinguishes them. It extracts data from hackathon platforms and GitHub, enriches repositories with semantic concepts, and identifies the signals that separate hackathon projects from regular open source work — things like burst commit patterns, repo age, contributor count, and topic composition.

The analysis pipeline was validated on LauzHack (2023-2025) as a case study with 487 repositories. Three prediction methods were developed and compared, and the best-performing model (Random Forest, 86% CV accuracy) is available for live classification of any GitHub repo.

### Key Features

- **Data extraction** from LauzHack, Devpost, and GitHub (REST + GraphQL APIs)
- **Concept enrichment** via the EPFL Graph API (optional)
- **Exploratory analysis** of hackathon vs open source repo characteristics
- **Three prediction methods** for hackathon repo detection:
  - Naive rule-based (keyword matching + GitHub stats)
  - Correlation-weighted scoring
  - Random Forest classifier
- **Live prediction** — give it any GitHub URL, get a classification
- **Interactive Streamlit dashboard** with EDA, method comparison, concept network graph, and a live prediction tab

## Installation

```bash
uv sync
```

Create a `.env` file in the project root:

```env
DATA_ROOT=/path/to/your/data
GITHUB_TOKEN=ghp_your_token_here

# Optional: EPFL concept enrichment
EPFL_GRAPH_API_HOST=https://graphai.epfl.ch
EPFL_GRAPH_API_PORT=443
EPFL_GRAPH_API_USER=your_user
EPFL_GRAPH_API_PASSWORD=your_password
```

## Usage

### Predict a Repository

Classify any public GitHub repo as hackathon or not:

```bash
uv run hackalysis predict "https://github.com/user/repo"
```

Get JSON output:

```bash
uv run hackalysis predict "https://github.com/user/repo" --json
```

### Retrain Models

After adding new labeled data, retrain the prediction models:

```bash
uv run hackalysis train-model
```

### Streamlit Dashboard

```bash
uv run streamlit run app/hackathon_story.py
```

Navigate to "6. Try It Yourself" in the sidebar to classify repos interactively.

### Extract Hackathon Data

Extract project data from LauzHack:

```bash
uv run hackalysis extract -o ./data -p lauzhack -y 2023,2024,2025
```

Extract from Devpost:

```bash
uv run hackalysis extract -o ./data -p devpost -n "ExampleHackathonName"
```

### Extract GitHub Metadata

Fetch GitHub metadata for repos referenced by an extracted hackathon dataset:

```bash
uv run hackalysis github-extract -o ./data -p lauzhack -y 2023
```

Extract repos from a GitHub organization or user account:

```bash
uv run hackalysis github-extract-account -o ./data -a openai --no-upload
```

## Project Structure

```
src/hackathon_analysis/
  data_extraction/     # Scraping + GitHub API client
  data_enrichment/     # EPFL concept enrichment
  data_analysis/       # EDA notebooks
  prediction/          # Prediction pipeline
    feature_engineering.py   # Derived features from GitHub metadata
    naive_rules.py           # Method 1: keyword + stats rules
    correlation_weighted.py  # Method 2: correlation-weighted scoring
    random_forest.py         # Method 3: Random Forest classifier
    concepts.py              # Optional EPFL concept enrichment
    pipeline.py              # Main entry point: URL -> prediction
    train_model.py           # Train and save model artifacts
  models/              # Saved model artifacts
app/                   # Streamlit dashboard
notebooks/             # Analysis notebooks
tests/                 # Test suite
```

## Development

Read first the [Contribution Guidelines](/CONTRIBUTING.md).

For technical documentation on setup and development, see the
[Development Guide](docs/development-guide.md)

### Running Tests

```bash
uv run pytest
```

## Acknowledgement

Acknowledge all contributors and external collaborators here.

## Copyright

Copyright © 2026-2028 Swiss Data Science Center (SDSC),
[www.datascience.ch](http://www.datascience.ch/). All rights reserved. The SDSC
is jointly established and legally represented by the École Polytechnique
Fédérale de Lausanne (EPFL) and the Eidgenössische Technische Hochschule Zürich
(ETH Zürich). This copyright encompasses all materials, software, documentation,
and other content created and developed by the SDSC.
