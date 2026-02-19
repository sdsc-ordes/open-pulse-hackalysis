"""Common utilities for hackathon data extraction."""

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from huggingface_hub import upload_folder

# Load environment variables from .env file
load_dotenv()


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
            typer.echo(f"\n  Uploading data to Hugging Face...")
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
                f"  ⚠ Warning: Failed to upload to Hugging Face: {str(e)}", err=True
            )
        return False
