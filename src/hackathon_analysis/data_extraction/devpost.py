"""Devpost data extraction module."""

from pathlib import Path
from typing import Optional

import typer

from hackathon_analysis.data_extraction.config import HACKATHON_CONFIGS


def extract_devpost(
    output_folder: Path, hackathon_name: str, years: Optional[str] = None
):
    """
    Extract hackathon data from Devpost.

    Args:
        output_folder: Path to save extracted data
        hackathon_name: Devpost hackathon name (e.g., 'treehacks-2026')
        years: Comma-separated years for organizing data. If None, extracts single instance.
    """
    config = HACKATHON_CONFIGS["devpost"]
    typer.echo(f"Extracting data from {config.name}...")
    typer.echo(f"  → Hackathon: {hackathon_name}")

    # For Devpost, years are optional (used for organizing output)
    if years:
        years_to_extract = [int(y.strip()) for y in years.split(",")]
    else:
        years_to_extract = [None]  # Single extraction without year

    for year in years_to_extract:
        if year:
            typer.echo(f"\n  Processing {year}:")
        projects_url = config.projects_url_template.format(
            hackathon_name=hackathon_name
        )
        metadata_url = config.metadata_url_template.format(
            hackathon_name=hackathon_name
        )

        typer.echo(f"    → Projects URL: {projects_url}")
        typer.echo(f"    → Metadata URL: {metadata_url}")

        # Create output folder (year-specific if provided)
        if year:
            output_dir = output_folder / str(year)
        else:
            output_dir = output_folder / hackathon_name

        output_dir.mkdir(parents=True, exist_ok=True)

        output_projects = output_dir / "devpost_projects.json"
        output_metadata = output_dir / "devpost_metadata.json"

        typer.echo(f"    → Data will be saved to:")
        typer.echo(f"       - {output_projects}")
        typer.echo(f"       - {output_metadata}")

        # TODO: Implement Devpost extraction logic
        # - Fetch projects data from {projects_url}
        # - Fetch hackathon metadata from {metadata_url}
        # - Save to output_projects and output_metadata
        # - Handle pagination if necessary
        # - Merge the hackathon metadata with each project entry for context such as category, tags, prize, hackathon details, etc.
        # - now write the merged data to output_projects and the hackathon metadata to output_metadata, make sure that it has github url of the project, if it's not there then put N/A there and other links should be in labelled columns too

        status_msg = f"{year} extraction" if year else "Extraction"
        typer.echo(f"    ℹ {status_msg} not yet implemented")
