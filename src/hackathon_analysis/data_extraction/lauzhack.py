"""LauzHack data extraction module."""

from pathlib import Path
from typing import Optional

import typer

from hackathon_analysis.data_extraction.common_utils import upload_to_hugging_face
from hackathon_analysis.data_extraction.config import HACKATHON_CONFIGS
from hackathon_analysis.data_extraction.lauzhack_extractor import (
    extract_year_data,
)


def extract_lauzhack(output_folder: Path, years: Optional[str] = None):
    """
    Extract hackathon data from LauzHack.

    Args:
        output_folder: Path to save extracted data
        years: Comma-separated years (e.g., "2025,2024,2023"). If None, uses all available years.
    """
    config = HACKATHON_CONFIGS["lauzhack"]
    typer.echo(f"Extracting data from {config.name}...")

    # Determine which years to extract
    if years:
        years_to_extract = [int(y.strip()) for y in years.split(",")]
    else:
        years_to_extract = config.years

    typer.echo(f"  → Years to extract: {years_to_extract}")

    for year in years_to_extract:
        typer.echo(f"\n  Processing {year}:")
        projects_url = config.projects_url_template.format(year=year)
        metadata_url = config.metadata_url_template.format(year=year)

        typer.echo(f"    → Projects URL: {projects_url}")
        typer.echo(f"    → Metadata URL: {metadata_url}")

        # Create output folder with hackathon name and year
        hackathon_year_folder = output_folder / f"{config.name.lower()}-{year}"
        hackathon_year_folder.mkdir(parents=True, exist_ok=True)

        output_projects = hackathon_year_folder / "lauzhack_projects.parquet"
        output_metadata = hackathon_year_folder / "lauzhack_metadata.json"

        typer.echo(f"    → Data will be saved to:")
        typer.echo(f"       - {output_projects}")
        typer.echo(f"       - {output_metadata}")

        # Extract data using helper functions
        try:
            stats = extract_year_data(
                projects_url=projects_url,
                metadata_url=metadata_url,
                output_projects=output_projects,
                output_metadata=output_metadata,
                merge_data=True,
            )
            typer.echo(
                f"    ✓ Completed: {stats['total_projects']} projects extracted"
            )
        except Exception as e:
            typer.echo(f"    ✗ Error: {str(e)}", err=True)

    # Upload entire output folder to Hugging Face
    # This maintains the folder structure (lauzhack-2023/, lauzhack-2024/, etc.)
    # and works with both LauzHack and Devpost data
    upload_to_hugging_face(output_folder)
