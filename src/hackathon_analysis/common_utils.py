"""Common utilities for hackathon data extraction."""

import os
from pathlib import Path

import typer
from huggingface_hub import upload_folder
from dotenv import load_dotenv


def upload_to_hugging_face(
    output_folder: Path,
    path_in_repo: str | None = None,
    verbose: bool = True,
) -> bool:
    """
    Upload extracted hackathon data to Hugging Face Hub.

    Args:
        output_folder: Path to folder containing extracted data
        path_in_repo: Optional destination path inside the Hugging Face repo
        verbose: Whether to print upload status messages (default: True)

    Returns:
        bool: True if upload succeeded, False otherwise

    Environment Variables:
        HF_REPO_ID: Hugging Face repository ID (default: SDSC/open-pulse-hackathon-data-analysis)
        HF_REPO_TYPE: Type of repository - "dataset" or "model" (default: dataset)
        HF_TOKEN: Hugging Face authentication token (optional if already logged in)
    """
    load_dotenv(override=False)

    # Always get values from environment variables / .env file
    repo_id = os.getenv(
        "HF_REPO_ID", "SDSC/open-pulse-hackathon-data-analysis")
    repo_type = os.getenv("HF_REPO_TYPE", "dataset")

    try:
        if verbose:
            typer.echo("\n  Uploading data to Hugging Face...")
        upload_folder(
            folder_path=str(output_folder),
            repo_id=repo_id,
            repo_type=repo_type,
            path_in_repo=path_in_repo,
        )
        if verbose:
            typer.echo("  ✓ Successfully uploaded to Hugging Face")
        return True
    except Exception as e:
        if verbose:
            typer.echo(
                f"  ⚠ Warning: Failed to upload to Hugging Face: {e}", err=True
            )
            typer.echo(
                (
                    "  Hint: set `HF_REPO_ID` to an existing dataset repo and "
                    "authenticate with `HF_TOKEN` or `huggingface-cli login`, "
                    "or rerun with `--no-upload`."
                ),
                err=True,
            )
        return False


def load_huggingface_dataset(
    repo_id: str,
    split: str | list[str] | None = "train",
    cache_dir: str | None = None,
    token: str | None = None,
    **kwargs,
):
    """
    Load a dataset from the Hugging Face Hub using the `datasets` library.

    Args:
        repo_id: Name of the dataset repo on Hugging Face (e.g., "sdsco/example").
        split: Split name or list of splits to retrieve (default "train").
            Pass None to load all splits as a DatasetDict.
        cache_dir: Optional cache directory to use for downloaded data.
        token: HuggingFace auth token for private repos. Falls back to HF_TOKEN env var.
        **kwargs: Additional keyword args forwarded to :func:`datasets.load_dataset`.

    Returns:
        A ``datasets.Dataset`` or ``datasets.DatasetDict`` depending on the split.

    This helper centralizes dataset loading so callers don't need to import
    `datasets` directly, and allows mocking in tests.
    """
    try:
        from datasets import load_dataset  # noqa: PLC0415
    except ImportError as e:
        raise RuntimeError(
            "datasets library is required to load Hugging Face datasets") from e

    resolved_token = token or os.getenv("HF_TOKEN")
    load_args: dict = {"path": repo_id}
    if split is not None:
        load_args["split"] = split
    if cache_dir:
        load_args["cache_dir"] = cache_dir
    if resolved_token:
        load_args["token"] = resolved_token
    load_args.update(kwargs)

    return load_dataset(**load_args)


def load_predictions_from_hf(repo_id: str, repo_type: str = "dataset") -> "pd.DataFrame":
    """Load the predictions CSV from a HF Hub dataset repo using a three-stage fallback.

    Stages tried in order:
      A) datasets.load_dataset(repo_id, split="train")  — works when a dataset card exists
      B) load_dataset with data_files="data/repo_metadata_with_predictions.csv"
         — works when data is stored as a raw CSV under data/
      C) huggingface_hub.hf_hub_download of the first *prediction*.csv in the repo
         — last resort, bypasses datasets entirely

    Returns a pandas DataFrame. Raises RuntimeError (with diagnostic detail) if all
    three stages fail.
    """
    import logging
    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download

    token = os.getenv("HF_TOKEN")

    # Stage A: standard split load — fastest when dataset card is present
    try:
        ds = load_huggingface_dataset(repo_id=repo_id, split="train")
        df = ds.to_pandas()
        logging.info("HF Stage A: loaded %d rows via split='train'", len(df))
        return df
    except Exception as e_a:
        logging.info("HF Stage A failed: %s", e_a)

    # Stage B: explicit data_files — works for raw CSVs in the data/ folder
    try:
        ds = load_huggingface_dataset(
            repo_id=repo_id,
            split="train",
            data_files="data/repo_metadata_with_predictions.csv",
        )
        df = ds.to_pandas()
        logging.info("HF Stage B: loaded %d rows via data_files", len(df))
        return df
    except Exception as e_b:
        logging.info("HF Stage B failed: %s", e_b)

    # Stage C: direct file download — most explicit, bypasses dataset card entirely
    try:
        api = HfApi()
        all_files = list(api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=token))
        csv_files = (
            [f for f in all_files if f.endswith(".csv") and "prediction" in f.lower()]
            or [f for f in all_files if f.endswith(".csv")]
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{repo_id}'. Available: {all_files[:15]}"
            )
        target = csv_files[0]
        logging.info("HF Stage C: downloading '%s'", target)
        local_path = hf_hub_download(
            repo_id=repo_id, filename=target, repo_type=repo_type, token=token
        )
        df = pd.read_csv(local_path)
        logging.info("HF Stage C: loaded %d rows", len(df))
        return df
    except Exception as e_c:
        raise RuntimeError(
            f"All three HF loading strategies failed for repo '{repo_id}'.\n"
            f"Stage C error: {e_c}\n"
            f"Ensure HF_REPO_ID is set correctly and HF_TOKEN is set if the repo is private."
        ) from e_c
