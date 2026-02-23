# Testing Guide

## Running Tests

**Run a specific test file:**
```bash
uv run pytest tests/test_extraction_helpers.py -v
```

**Run all tests:**
```bash
uv run pytest tests/ -v
```

**Run tests with coverage:**
```bash
uv run pytest tests/ --cov=hackathon_analysis --cov-report=html
```

**Run directly as Python (for simple tests):**
```bash
uv run python tests/test_extraction_helpers.py
```

