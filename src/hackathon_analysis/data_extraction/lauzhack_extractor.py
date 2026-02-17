"""LauzHack data extraction helper functions."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import typer
from bs4 import BeautifulSoup


def _normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    normalized = re.sub(r"\s+", " ", title).strip().lower()
    return normalized


def fetch_projects_data(projects_url: str) -> List[Dict[str, Any]]:
    """
    Fetch projects data from LauzHack projects page.

    Args:
        projects_url: URL to the LauzHack projects page

    Returns:
        List of project dictionaries

    Raises:
        Exception: If fetching or parsing fails
    """
    typer.echo(f"      → Fetching projects from {projects_url}")

    try:
        response = requests.get(projects_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        typer.echo(f"      ✗ Error fetching projects: {e}", err=True)
        raise

    soup = BeautifulSoup(response.content, "html.parser")
    projects = []

    # LauzHack 2025 wraps <details> inside <article> with a footer for team/link
    details_elements = soup.find_all("details")
    article_elements = soup.find_all("article")
    has_article_details = any(
        article.find("details") is not None for article in article_elements
    )

    if article_elements and has_article_details:
        typer.echo(
            f"      → Found {len(article_elements)} <article> elements with <details>"
        )
        for idx, element in enumerate(article_elements, 1):
            try:
                project = extract_project_info(element, idx)
                if project:
                    projects.append(project)
            except Exception as e:
                typer.echo(f"      ⚠ Error parsing project {idx}: {e}")
                continue

        return projects

    if details_elements:
        typer.echo(f"      → Found {len(details_elements)} <details> elements")
        for idx, element in enumerate(details_elements, 1):
            try:
                project = extract_project_info(element, idx)
                if project:
                    projects.append(project)
            except Exception as e:
                typer.echo(f"      ⚠ Error parsing project {idx}: {e}")
                continue

        if article_elements:
            typer.echo(
                f"      → Found {len(article_elements)} <article> elements for awards"
            )
            awards_projects = []
            for idx, element in enumerate(article_elements, 1):
                try:
                    project = extract_project_info(element, idx)
                    if project:
                        awards_projects.append(project)
                except Exception as e:
                    typer.echo(
                        f"      ⚠ Error parsing awards project {idx}: {e}")
                    continue

            awards_by_title = {
                _normalize_title(p.get("title")): p
                for p in awards_projects
                if p.get("title")
            }

            for project in projects:
                key = _normalize_title(project.get("title"))
                if not key:
                    continue
                awards_project = awards_by_title.get(key)
                if not awards_project:
                    continue
                if awards_project.get("awards") and not project.get("awards"):
                    project["awards"] = awards_project["awards"]
                    project["categories"] = awards_project.get(
                        "categories", awards_project["awards"]
                    )
                if awards_project.get("url") and not project.get("url"):
                    project["url"] = awards_project["url"]
                if awards_project.get("team") and not project.get("team"):
                    project["team"] = awards_project["team"]

        return projects

    # Fallback to other common selectors
    project_elements = soup.find_all("div", class_="project") or soup.find_all(
        "div", class_="project-card"
    )

    if not project_elements:
        # Try alternative selectors
        project_elements = (
            article_elements
            or soup.find_all("div", class_="card")
            or soup.find_all("section", class_="project")
        )

    typer.echo(f"      → Found {len(project_elements)} project elements")

    for idx, element in enumerate(project_elements, 1):
        try:
            project = extract_project_info(element, idx)
            if project:
                projects.append(project)
        except Exception as e:
            typer.echo(f"      ⚠ Error parsing project {idx}: {e}")
            continue

    return projects


def extract_project_info(
    element: BeautifulSoup, idx: int
) -> Optional[Dict[str, Any]]:
    """
    Extract information from a single project element.

    Args:
        element: BeautifulSoup element containing project info
        idx: Project index for identification

    Returns:
        Dictionary with project information or None if extraction fails
    """
    project = {"id": idx}

    # For <details> elements (LauzHack 2025 structure)
    if element.name == "details":
        # Extract title from <summary>
        summary_elem = element.find("summary")
        if summary_elem:
            project["title"] = summary_elem.get_text(strip=True)

        # Check for awards/prizes in mark tags
        mark_elems = element.find_all("mark")
        if mark_elems:
            awards = [mark.get_text(strip=True) for mark in mark_elems]
            project["awards"] = awards
            project["categories"] = awards  # Same as awards for consistency

        # Extract description (text after summary)
        description_parts = []
        for child in element.children:
            if child.name == "summary":
                continue
            if child.name == "footer":
                break
            if isinstance(child, str):
                text = child.strip()
                if text:
                    description_parts.append(text)
            elif hasattr(child, 'get_text'):
                text = child.get_text(strip=True)
                if text:
                    description_parts.append(text)

        if description_parts:
            project["description"] = " ".join(description_parts)

        # Extract team and link from <footer>
        footer_elem = element.find("footer")
        if footer_elem:
            # Extract project link first
            link_elem = footer_elem.find("a")
            if link_elem and link_elem.get("href"):
                project["url"] = link_elem["href"]
                # Remove the link to get clean team text
                link_elem.decompose()

            # Get text content and split by <br> tags
            # Replace <br> with newlines for proper splitting
            for br in footer_elem.find_all("br"):
                br.replace_with("\n")

            footer_text = footer_elem.get_text(strip=True)
            lines = [line.strip()
                     for line in footer_text.split('\n') if line.strip()]

            if lines:
                # First line usually contains team members (comma-separated)
                team_line = lines[0]
                if team_line and ',' in team_line:
                    project["team"] = [m.strip()
                                       for m in team_line.split(',') if m.strip()]
                elif team_line:
                    project["team"] = [team_line]

    # For <article> elements (LauzHack 2023/2024 structure)
    elif element.name == "article":
        # Articles typically have header > b for title
        header_elem = element.find("header")
        if header_elem:
            title_elem = header_elem.find("b")
            if title_elem:
                project["title"] = title_elem.get_text(strip=True)

            # Check for awards/prizes in mark tags (extract ALL mark tags)
            mark_elems = header_elem.find_all("mark")
            if mark_elems:
                awards = [mark.get_text(strip=True) for mark in mark_elems]
                project["awards"] = awards
                # Same as awards for consistency
                project["categories"] = awards

        details_elem = element.find("details")
        if details_elem:
            summary_elem = details_elem.find("summary")
            summary_text = None
            if summary_elem:
                summary_text = summary_elem.get_text(strip=True)

            description_parts = []
            for child in details_elem.children:
                if getattr(child, "name", None) == "summary":
                    continue
                if isinstance(child, str):
                    text = child.strip()
                    if text:
                        description_parts.append(text)
                elif hasattr(child, "get_text"):
                    text = child.get_text(strip=True)
                    if text:
                        description_parts.append(text)

            if description_parts:
                project["description"] = " ".join(description_parts)
            elif summary_text and "description" not in project:
                project["description"] = summary_text

        # If no header, try other heading tags
        if "title" not in project:
            title_elem = element.find("h2") or element.find(
                "h3") or element.find("h4")
            if title_elem:
                project["title"] = title_elem.get_text(strip=True)

        # Extract description from main content divs or paragraphs
        # Look in divs that are not headers
        content_divs = []
        for div in element.find_all("div"):
            text = div.get_text(strip=True)
            # Skip very short divs or those that look like metadata
            if text and len(text) > 20:
                content_divs.append(text)

        if content_divs:
            project["description"] = " ".join(
                content_divs[:2])  # First 2 content divs

        # Also check paragraphs
        if "description" not in project or len(project["description"]) < 50:
            desc_paragraphs = element.find_all("p")
            if desc_paragraphs:
                description_parts = [p.get_text(strip=True)
                                     for p in desc_paragraphs]
                project["description"] = " ".join(description_parts)

        footer_elem = element.find("footer")
        if footer_elem:
            link_elem = footer_elem.find("a")
            if link_elem and link_elem.get("href"):
                project["url"] = link_elem["href"]
                link_elem.decompose()

            for br in footer_elem.find_all("br"):
                br.replace_with("\n")

            footer_text = footer_elem.get_text(strip=True)
            lines = [line.strip()
                     for line in footer_text.split("\n") if line.strip()]

            if lines:
                team_line = lines[0]
                if team_line and "," in team_line:
                    project["team"] = [
                        m.strip() for m in team_line.split(",") if m.strip()
                    ]
                elif team_line:
                    project["team"] = [team_line]

        # Extract link if not found in footer
        if "url" not in project:
            link_elem = element.find("a")
            if link_elem and link_elem.get("href"):
                href = link_elem["href"]
                if href and not href.startswith("#") and not href.startswith("javascript"):
                    project["url"] = href

        # Try to extract team info (often in small tags or specific divs)
        team_elem = element.find(
            class_="team") or element.find(class_="authors")
        if team_elem:
            team_text = team_elem.get_text(strip=True)
            if ',' in team_text:
                project["team"] = [m.strip()
                                   for m in team_text.split(',') if m.strip()]
            else:
                project["team"] = [team_text] if team_text else []

    else:
        # Fallback: Try standard selectors for other HTML structures
        title_elem = (
            element.find("h1")
            or element.find("h2")
            or element.find("h3")
            or element.find(class_="title")
            or element.find(class_="project-title")
        )
        if title_elem:
            project["title"] = title_elem.get_text(strip=True)

        # Extract description
        desc_elem = (
            element.find("p")
            or element.find(class_="description")
            or element.find(class_="project-description")
        )
        if desc_elem:
            project["description"] = desc_elem.get_text(strip=True)

        # Extract link
        link_elem = element.find("a")
        if link_elem and link_elem.get("href"):
            href = link_elem["href"]
            # Make absolute URL if relative
            if href.startswith("/"):
                base_url = "https://lauzhack.com"
                project["url"] = base_url + href
            else:
                project["url"] = href

        # Extract team members (if available)
        team_elems = element.find_all(class_="team-member") or element.find_all(
            class_="member"
        )
        if team_elems:
            project["team"] = [
                member.get_text(strip=True) for member in team_elems
            ]

        # Extract tags/technologies
        tag_elems = element.find_all(class_="tag") or element.find_all(
            class_="tech"
        )
        if tag_elems:
            project["tags"] = [tag.get_text(strip=True) for tag in tag_elems]

        # Extract image if available
        img_elem = element.find("img")
        if img_elem and img_elem.get("src"):
            project["image_url"] = img_elem["src"]

    return project if "title" in project else None


def fetch_metadata(metadata_url: str) -> Dict[str, Any]:
    """
    Fetch hackathon metadata from LauzHack main page.

    Args:
        metadata_url: URL to the LauzHack main page

    Returns:
        Dictionary containing hackathon metadata

    Raises:
        Exception: If fetching or parsing fails
    """
    typer.echo(f"      → Fetching metadata from {metadata_url}")

    try:
        response = requests.get(metadata_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        typer.echo(f"      ✗ Error fetching metadata: {e}", err=True)
        raise

    soup = BeautifulSoup(response.content, "html.parser")
    metadata = {"source_url": metadata_url}

    # Extract year from URL or title
    year_match = re.search(r"(\d{4})", metadata_url)
    if year_match:
        metadata["year"] = int(year_match.group(1))

    # Set hackathon name
    metadata["name"] = f"LauzHack {metadata.get('year', '')}"

    # Extract description
    desc_elem = (
        soup.find("meta", attrs={"name": "description"})
        or soup.find("meta", property="og:description")
        or soup.find(class_="description")
    )
    if desc_elem:
        if desc_elem.name == "meta":
            metadata["description"] = desc_elem.get("content", "")
        else:
            metadata["description"] = desc_elem.get_text(strip=True)

    # Extract date information
    date_elem = soup.find(class_="date") or soup.find(class_="event-date")
    if date_elem:
        metadata["date"] = date_elem.get_text(strip=True)

    # Extract location
    location_elem = soup.find(class_="location") or soup.find(
        class_="venue"
    )
    if location_elem:
        metadata["location"] = location_elem.get_text(strip=True)
    else:
        metadata["location"] = "EPFL, Lausanne, Switzerland"

    # Extract sponsor information
    sponsor_elements = soup.find_all(class_="sponsor") or soup.find_all(
        class_="partner"
    )
    if sponsor_elements:
        metadata["sponsors"] = [
            elem.get_text(strip=True) for elem in sponsor_elements
        ]

    # Extract prize information
    prize_elements = soup.find_all(class_="prize") or soup.find_all(
        class_="award"
    )
    if prize_elements:
        metadata["prizes"] = [
            elem.get_text(strip=True) for elem in prize_elements
        ]

    # Extract social media links
    social_links = {}
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if "twitter.com" in href or "x.com" in href:
            social_links["twitter"] = href
        elif "facebook.com" in href:
            social_links["facebook"] = href
        elif "instagram.com" in href:
            social_links["instagram"] = href
        elif "linkedin.com" in href:
            social_links["linkedin"] = href
        elif "github.com" in href and "lauzhack" in href.lower():
            social_links["github"] = href

    if social_links:
        metadata["social_links"] = social_links

    return metadata


def process_project_data(projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process and clean project data.

    Args:
        projects: Raw projects data

    Returns:
        Processed projects data
    """
    typer.echo(f"      → Processing {len(projects)} projects")

    processed_projects = []
    seen_keys = set()

    for project in projects:
        # Remove duplicates based on multiple fields
        title = project.get("title", "")
        description = project.get("description", "")
        url = project.get("url", "")
        team = project.get("team", []) or []
        team_key = ",".join([member.strip() for member in team if member]).lower()
        dedupe_key = "|".join(
            [
                _normalize_title(title),
                _normalize_title(description),
                url.strip().lower(),
                team_key,
            ]
        )

        if any(dedupe_key):
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

        # Clean and validate data
        processed_project = {
            "id": project.get("id"),
            "title": title or "Untitled Project",
            # Limit length
            "description": project.get("description", "")[:500],
            "url": project.get("url", ""),
            "team": project.get("team", []),
            "tags": project.get("tags", []),
            "image_url": project.get("image_url", ""),
            "awards": project.get("awards", []),
            "categories": project.get("categories", []),
        }

        # Remove empty fields
        processed_project = {
            k: v for k, v in processed_project.items() if v or v == 0
        }

        processed_projects.append(processed_project)

    return processed_projects


def process_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process and clean metadata.

    Args:
        metadata: Raw metadata

    Returns:
        Processed metadata
    """
    typer.echo(f"      → Processing metadata")

    # Add timestamp
    metadata["extracted_at"] = datetime.now().isoformat()

    # Ensure required fields exist
    if "name" not in metadata:
        metadata["name"] = "LauzHack"

    if "location" not in metadata:
        metadata["location"] = "EPFL, Lausanne, Switzerland"

    # Clean description
    if "description" in metadata:
        metadata["description"] = metadata["description"][:1000]

    return metadata


def merge_project_data(
    projects: List[Dict[str, Any]], metadata: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Merge project data with hackathon metadata.

    Args:
        projects: List of projects
        metadata: Hackathon metadata

    Returns:
        Projects with embedded metadata context
    """
    typer.echo(f"      → Merging {len(projects)} projects with metadata")

    merged_projects = []
    for project in projects:
        merged_project = {
            **project,
            "hackathon_name": metadata.get("name"),
            "hackathon_year": metadata.get("year"),
            "hackathon_location": metadata.get("location"),
        }
        merged_projects.append(merged_project)

    return merged_projects


def save_projects_data(
    projects: List[Dict[str, Any]], output_file: Path
) -> None:
    """
    Save projects data to JSON file.

    Args:
        projects: Projects data to save
        output_file: Path to output JSON file
    """
    typer.echo(
        f"      → Saving {len(projects)} projects to {output_file.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(projects, f, indent=2, ensure_ascii=False)


def save_metadata(metadata: Dict[str, Any], output_file: Path) -> None:
    """
    Save metadata to JSON file.

    Args:
        metadata: Metadata to save
        output_file: Path to output JSON file
    """
    typer.echo(f"      → Saving metadata to {output_file.name}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def extract_year_data(
    projects_url: str,
    metadata_url: str,
    output_projects: Path,
    output_metadata: Path,
    merge_data: bool = True,
) -> Dict[str, Any]:
    """
    Extract data for a single year.

    This is the main orchestration function that coordinates all extraction steps.

    Args:
        projects_url: URL to projects page
        metadata_url: URL to metadata page
        output_projects: Path to save projects JSON
        output_metadata: Path to save metadata JSON
        merge_data: Whether to merge metadata into projects

    Returns:
        Dictionary with extraction statistics
    """
    typer.echo(f"      → Starting extraction")

    # Step 1: Fetch raw data
    projects = fetch_projects_data(projects_url)
    metadata = fetch_metadata(metadata_url)

    # Step 2: Process data
    projects = process_project_data(projects)
    metadata = process_metadata(metadata)

    # Step 3: Merge if requested
    if merge_data:
        projects = merge_project_data(projects, metadata)

    # Step 4: Save data
    save_projects_data(projects, output_projects)
    save_metadata(metadata, output_metadata)

    # Return statistics
    stats = {
        "total_projects": len(projects),
        "metadata_fields": len(metadata.keys()),
        "output_files": [str(output_projects), str(output_metadata)],
    }

    typer.echo(f"      ✓ Extracted {stats['total_projects']} projects")
    return stats
