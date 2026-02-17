"""Configuration for different hackathon providers."""

from dataclasses import dataclass, field


@dataclass
class HackathonConfig:
    """Configuration for a hackathon provider."""

    name: str
    projects_url_template: str
    metadata_url_template: str
    years: list = field(default_factory=list)
    supports_custom_url: bool = False  # Whether provider supports custom URLs


# Hackathon provider configurations
HACKATHON_CONFIGS = {
    "lauzhack": HackathonConfig(
        name="LauzHack",
        projects_url_template="https://{year}.lauzhack.com/projects",
        metadata_url_template="https://{year}.lauzhack.com",
        years=[2025, 2024, 2023],
    ),
    "devpost": HackathonConfig(
        name="Devpost",
        projects_url_template="https://{hackathon_name}.devpost.com/project-gallery",
        metadata_url_template="https://{hackathon_name}.devpost.com/",
        years=[],
        supports_custom_url=True,
    ),
}
