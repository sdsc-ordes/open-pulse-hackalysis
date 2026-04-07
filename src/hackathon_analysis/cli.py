from pathlib import Path
from typing import Annotated, Optional, List
import os
import typer
from dotenv import load_dotenv

from hackathon_analysis.common_utils import upload_to_hugging_face
from hackathon_analysis.data_extraction.lauzhack import extract_lauzhack
from hackathon_analysis.data_extraction.devpost import extract_devpost
from hackathon_analysis.data_extraction.dataset_resolver import (
    resolve_lauzhack_years,
    get_projects_parquet_path,
    ensure_local_file_from_hf,
)
from hackathon_analysis.data_extraction.github_extractor import (
    run_repo_metadata_from_projects_parquet,
    run_repo_metadata_from_account,
)

app = typer.Typer(help="Hackathon Analysis Tool")


def _parse_years(years: Optional[str]) -> Optional[List[int]]:
    if not years:
        return None
    return [int(y.strip()) for y in years.split(",") if y.strip()]


def _get_github_token() -> Optional[str]:
    """Load .env values for CLI execution and return GitHub token if available."""
    load_dotenv(override=False)
    return os.getenv("GITHUB_TOKEN")


def _ensure_projects_parquet_or_exit(
    output_folder: Path,
    local_parquet: Path,
    hf_subpath: str,
    *,
    provider_label: str,
) -> Path:
    """Ensure the projects parquet exists locally or can be downloaded."""
    if local_parquet.exists():
        return local_parquet

    typer.echo(f"Downloading from Hugging Face: {hf_subpath}")
    try:
        return ensure_local_file_from_hf(output_folder, hf_subpath)
    except Exception:
        typer.echo(
            (
                f"Error: Could not find the source project dataset for {provider_label} "
                f"locally or on Hugging Face.\n"
                "Run the extract command first to create it locally, then rerun "
                "github-extract."
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command()
def extract(
    output_folder: Annotated[
        Path,
        typer.Option("--output_folder", "-o", help="Output folder path for extracted data"),
    ],
    hackathon_provider: Annotated[
        str,
        typer.Option("--hackathon_provider", "-p", help="Hackathon provider (lauzhack or devpost)"),
    ],
    years: Annotated[
        Optional[str],
        typer.Option("--years", "-y", help="Comma-separated years to extract for lauzhack"),
    ] = None,
    hackathon_name: Annotated[
        Optional[str],
        typer.Option("--hackathon_name", "-n", help="Hackathon name for Devpost"),
    ] = None,
    upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Upload outputs back to Hugging Face"),
    ] = True,
):
    output_folder.mkdir(parents=True, exist_ok=True)

    if hackathon_provider.lower() == "lauzhack":
        extract_lauzhack(output_folder, years, upload=upload)
    elif hackathon_provider.lower() == "devpost":
        if not hackathon_name:
            typer.echo("Error: --hackathon_name is required for devpost provider", err=True)
            raise typer.Exit(code=1)
        extract_devpost(output_folder, hackathon_name, years, upload=upload)
    else:
        typer.echo(f"Error: Unknown provider '{hackathon_provider}'", err=True)
        raise typer.Exit(code=1)

    typer.echo("✓ Extraction complete!")


@app.command(name="github-extract")
def github_extract( # noqa: PLR0913 because we want to keep all these parameters defined below
    output_folder: Annotated[
        Path,
        typer.Option("--output_folder", "-o", help="Root folder holding extracted hackathon data"),
    ],
    hackathon_provider: Annotated[
        str,
        typer.Option("--hackathon_provider", "-p", help="Hackathon provider (lauzhack or devpost)"),
    ],
    years: Annotated[
        Optional[str],
        typer.Option("--years", "-y", help="Comma-separated years to process for lauzhack"),
    ] = None,
    hackathon_name: Annotated[
        Optional[str],
        typer.Option("--hackathon_name", "-n", help="Devpost hackathon name"),
    ] = None,
    top_contributors: Annotated[
        int,
        typer.Option("--top_contributors", help="Number of top contributors to fetch per repo"),
    ] = 8,
    upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Upload outputs back to Hugging Face"),
    ] = True,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    provider = hackathon_provider.lower().strip()
    gh_token = _get_github_token()

    if provider == "lauzhack":
        years_list = resolve_lauzhack_years(output_folder, _parse_years(years))
        if not years_list:
            typer.echo("Error: No LauzHack years found locally or on Hugging Face", err=True)
            raise typer.Exit(code=1)

        for year in years_list:
            local_parquet, hf_subpath = get_projects_parquet_path(output_folder, "lauzhack", year=year)
            local_parquet = _ensure_projects_parquet_or_exit(
                output_folder,
                local_parquet,
                hf_subpath,
                provider_label=f"LauzHack {year}",
            )

            hackathon_folder = output_folder / f"lauzhack-{year}"
            summary = run_repo_metadata_from_projects_parquet(
                projects_parquet=local_parquet,
                hackathon_folder=hackathon_folder,
                provider_prefix="lauzhack",
                token=gh_token,
                top_contributors=top_contributors,
            )

            typer.echo(f"LauzHack {year}: {summary['repos_found']} GitHub URLs found")
            typer.echo(f"Wrote {summary['outputs']['json']}")
            typer.echo(f"Wrote {summary['outputs']['parquet']}")

    elif provider == "devpost":
        if not hackathon_name:
            typer.echo("Error: --hackathon_name is required for devpost provider", err=True)
            raise typer.Exit(code=1)

        local_parquet, hf_subpath = get_projects_parquet_path(output_folder, "devpost", hackathon_name=hackathon_name)
        local_parquet = _ensure_projects_parquet_or_exit(
            output_folder,
            local_parquet,
            hf_subpath,
            provider_label=f"Devpost {hackathon_name}",
        )

        hackathon_folder = output_folder / f"devpost-{hackathon_name}"
        summary = run_repo_metadata_from_projects_parquet(
            projects_parquet=local_parquet,
            hackathon_folder=hackathon_folder,
            provider_prefix="devpost",
            token=gh_token,
            top_contributors=top_contributors,
        )

        typer.echo(f"Devpost {hackathon_name}: {summary['repos_found']} GitHub URLs found")
        typer.echo(f"Wrote {summary['outputs']['json']}")
        typer.echo(f"Wrote {summary['outputs']['parquet']}")

    else:
        typer.echo(f"Error: Unknown provider '{hackathon_provider}'", err=True)
        raise typer.Exit(code=1)

    if upload:
        upload_to_hugging_face(output_folder)

    typer.echo("✓ GitHub extraction complete!")


@app.command(name="github-extract-account")
def github_extract_account( # noqa: PLR0913 because we want to keep all these parameters defined below
    output_folder: Annotated[
        Path,
        typer.Option("--output_folder", "-o", help="Root folder holding extracted GitHub account data"),
    ],
    account_name: Annotated[
        str,
        typer.Option("--account_name", "-a", help="GitHub organization or username"),
    ],
    account_type: Annotated[
        str,
        typer.Option("--account_type", help="Account type: auto, org, or user"),
    ] = "auto",
    top_contributors: Annotated[
        int,
        typer.Option("--top_contributors", help="Number of top contributors to fetch per repo"),
    ] = 8,
    upload: Annotated[
        bool,
        typer.Option("--upload/--no-upload", help="Upload outputs back to Hugging Face"),
    ] = True,
):
    output_folder.mkdir(parents=True, exist_ok=True)
    gh_token = _get_github_token()

    account_slug = account_name.strip()
    account_folder = output_folder / f"github-account-{account_slug}"
    summary = run_repo_metadata_from_account(
        account_name=account_slug,
        hackathon_folder=account_folder,
        provider_prefix="github_account",
        token=gh_token,
        account_type=account_type,
        top_contributors=top_contributors,
    )

    typer.echo(f"GitHub account {account_slug}: {summary['repos_found']} repos found")
    typer.echo(f"Wrote {summary['outputs']['json']}")
    typer.echo(f"Wrote {summary['outputs']['parquet']}")
    typer.echo(f"Wrote {summary['outputs']['project_json']}")
    typer.echo(f"Wrote {summary['outputs']['project_parquet']}")

    if upload:
        upload_to_hugging_face(account_folder, path_in_repo=account_folder.name)

    typer.echo("✓ GitHub account extraction complete!")


@app.command()
def predict(
    repo_url: Annotated[
        str,
        typer.Argument(help="GitHub repository URL to classify"),
    ],
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output raw JSON instead of human-readable text"),
    ] = False,
):
    """Predict whether a GitHub repo is a hackathon project."""
    import json as json_mod

    from hackathon_analysis.prediction.pipeline import (
        predict_repo,
        predict_repo_pretty,
    )

    gh_token = _get_github_token()

    try:
        if json_output:
            result = predict_repo(repo_url, github_token=gh_token)
            typer.echo(json_mod.dumps(result, indent=2))
        else:
            output = predict_repo_pretty(repo_url, github_token=gh_token)
            typer.echo(output)
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1) from None


@app.command(name="train-model")
def train_model_cmd(
    data_path: Annotated[
        Optional[Path],
        typer.Option("--data", "-d", help="Path to labeled CSV. Defaults to DATA_ROOT/repo_metadata_with_predictions.csv"),
    ] = None,
):
    """Train prediction models on labeled data and save artifacts."""
    from hackathon_analysis.prediction.train_model import train_and_save

    load_dotenv()

    if data_path is None:
        data_root = os.getenv("DATA_ROOT")
        if not data_root:
            typer.echo("Error: DATA_ROOT not set and no --data path provided", err=True)
            raise typer.Exit(code=1)
        data_path = Path(data_root) / "repo_metadata_with_predictions.csv"

    if not data_path.exists():
        typer.echo(f"Error: Data file not found: {data_path}", err=True)
        raise typer.Exit(code=1)

    metadata = train_and_save(data_path)
    typer.echo(f"Trained on {metadata['valid_repos']} repos with {metadata['n_features']} features")
    typer.echo(f"Cross-validated accuracy: {metadata['cv_accuracy']:.1%}")


if __name__ == "__main__":
    load_dotenv()
    app()
