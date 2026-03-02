"""Common utilities for hackathon data extraction."""

import os
from pathlib import Path

import typer
from huggingface_hub import upload_folder


def upload_to_hugging_face(
    output_folder: Path,
    verbose: bool = True,
) -> bool:
    """
    Upload extracted hackathon data to Hugging Face Hub.

    Args:
        output_folder: Path to folder containing extracted data
        verbose: Whether to print upload status messages (default: True)

    Returns:
        bool: True if upload succeeded, False otherwise

    Environment Variables:
        HF_REPO_ID: Hugging Face repository ID (default: SDSC/open-pulse-hackathon-data-analysis)
        HF_REPO_TYPE: Type of repository - "dataset" or "model" (default: dataset)
        HF_TOKEN: Hugging Face authentication token (optional if already logged in)
    """
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
        )
        if verbose:
            typer.echo("  ✓ Successfully uploaded to Hugging Face")
        return True
    except Exception as e:
        if verbose:
            typer.echo(
                f"  ⚠ Warning: Failed to upload to Hugging Face: {e}", err=True
            )
        return False


def load_huggingface_dataset(
    repo_id: str,
    split: str | list[str] = "train",
    cache_dir: str | None = None,
    **kwargs,
):
    """
    Load a dataset from the Hugging Face Hub using the `datasets` library.

    Args:
        repo_id: Name of the dataset repo on Hugging Face (e.g., "sdsco/example").
        split: Split name or list of splits to retrieve (default "train").
        cache_dir: Optional cache directory to use for downloaded data.
        **kwargs: Additional keyword args forwarded to :func:`datasets.load_dataset`.

    Returns:
        A ``datasets.Dataset`` or ``datasets.DatasetDict`` depending on the split.

    This helper centralizes dataset loading so callers don't need to import
    `datasets` directly, and allows mocking in tests.
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "datasets library is required to load Hugging Face datasets") from e

    load_args = {"path": repo_id, "split": split}
    if cache_dir:
        load_args["cache_dir"] = cache_dir
    # merge kwargs
    load_args.update(kwargs)

    return load_dataset(**load_args)
