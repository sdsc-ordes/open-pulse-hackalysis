from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from huggingface_hub import HfApi, hf_hub_download
'''
module is responsible for finding and retrieving the correct hackathon dataset files (projects parquet files) either locally or from Hugging Face. 
It provides functions to list available years for Lauzhack datasets and to ensure that the required parquet files are present locally, downloading from Hugging Face if necessary.
'''

@dataclass(frozen=True)
class HFRepo:
    repo_id: str
    repo_type: str


def get_hf_repo_from_env() -> HFRepo:
    repo_id = os.getenv("HF_REPO_ID", "SDSC/open-pulse-hackathon-data-analysis")
    repo_type = os.getenv("HF_REPO_TYPE", "dataset")
    return HFRepo(repo_id=repo_id, repo_type=repo_type)


def list_hf_files(repo: HFRepo) -> List[str]:
    api = HfApi()
    return api.list_repo_files(repo_id=repo.repo_id, repo_type=repo.repo_type)


def local_lauzhack_years(output_folder: Path) -> List[int]:
    years: set[int] = set()
    for p in output_folder.glob("lauzhack-*"):
        if not p.is_dir():
            continue
        m = re.match(r"^lauzhack-(\d{4})$", p.name)
        if not m:
            continue
        parquet = p / "lauzhack_projects.parquet"
        if parquet.exists():
            years.add(int(m.group(1)))
    return sorted(years)


def hf_lauzhack_years(repo: HFRepo) -> List[int]:
    files = list_hf_files(repo)
    years: set[int] = set()
    pat = re.compile(r"^lauzhack-(\d{4})/lauzhack_projects\.parquet$")
    for f in files:
        m = pat.match(f)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)


def ensure_local_file_from_hf(output_folder: Path, subpath: str, repo: Optional[HFRepo] = None) -> Path:
    """
    Ensures output_folder/subpath exists.
    If missing, downloads from Hugging Face into output_folder while preserving paths.
    """
    repo = repo or get_hf_repo_from_env()
    local_path = output_folder / subpath
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path

    downloaded = hf_hub_download(
        repo_id=repo.repo_id,
        repo_type=repo.repo_type,
        filename=subpath,
        local_dir=str(output_folder),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def resolve_lauzhack_years(output_folder: Path, years: Optional[List[int]] = None) -> List[int]:
    """
    If years is provided, use it.
    Else, prefer local years if present.
    Else, use Hugging Face years.
    """
    if years:
        return years

    local = local_lauzhack_years(output_folder)
    if local:
        return local

    repo = get_hf_repo_from_env()
    hf = hf_lauzhack_years(repo)
    return hf


def get_projects_parquet_path(
    output_folder: Path,
    provider: str,
    *,
    year: Optional[int] = None,
    hackathon_name: Optional[str] = None,
) -> Tuple[Path, str]:
    """
    Returns (local_path, hf_subpath) for the provider projects parquet.
    It does not download automatically.
    """
    provider = provider.lower().strip()

    if provider == "lauzhack":
        if year is None:
            raise ValueError("year is required for lauzhack projects parquet path")
        folder = output_folder / f"lauzhack-{year}"
        return folder / "lauzhack_projects.parquet", f"lauzhack-{year}/lauzhack_projects.parquet"

    if provider == "devpost":
        if not hackathon_name:
            raise ValueError("hackathon_name is required for devpost projects parquet path")
        folder = output_folder / f"devpost-{hackathon_name}"
        return folder / "devpost_projects.parquet", f"devpost-{hackathon_name}/devpost_projects.parquet"

    raise ValueError(f"Unknown provider '{provider}'")