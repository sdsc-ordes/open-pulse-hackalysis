from pathlib import Path
from typing import Annotated

import typer

from hackathon_analysis.data_extraction.lauzhack import extract_lauzhack
from hackathon_analysis.data_extraction.devpost import extract_devpost

app = typer.Typer(help="Hackathon Analysis Tool")


@app.command()
def extract(
    output_folder: Annotated[
        Path,
        typer.Option(
            "--output_folder",
            "-o",
            help="Output folder path for extracted data",
        ),
    ],
    hackathon_provider: Annotated[
        str,
        typer.Option(
            "--hackathon_provider",
            "-p",
            help="Hackathon provider (lauzhack or devpost)",
        ),
    ],
):
    """Extract hackathon data from specified provider."""
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    typer.echo(
        f"Extracting data from {hackathon_provider} to {output_folder}..."
    )

    if hackathon_provider.lower() == "lauzhack":
        extract_lauzhack(output_folder)
    elif hackathon_provider.lower() == "devpost":
        extract_devpost(output_folder)
    else:
        typer.echo(
            f"Error: Unknown provider '{hackathon_provider}'. "
            "Supported providers: lauzhack, devpost",
            err=True,
        )
        raise typer.Exit(code=1)

    typer.echo("✓ Extraction complete!")


if __name__ == "__main__":
    app()
