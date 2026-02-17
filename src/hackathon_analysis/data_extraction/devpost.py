"""Devpost data extraction module."""

from pathlib import Path
import typer


def extract_devpost(output_folder: Path):
    """
    Extract hackathon data from Devpost.

    Args:
        output_folder: Path to save extracted data
    """
    typer.echo("Extracting data from Devpost...")
    
    # TODO: Implement Devpost extraction logic
    # Example: Fetch data from Devpost API or scrape website
    # Save to output_folder / "devpost_data.json" or similar
    
    output_file = output_folder / "devpost_data.json"
    typer.echo(f"  → Data will be saved to: {output_file}")
    
    # Placeholder for actual implementation
    typer.echo("  ℹ Devpost extraction not yet implemented")
