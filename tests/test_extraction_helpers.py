"""
Basic test for LauzHack data extraction.

This test demonstrates how to test a simple helper function that normalizes project titles.
"""

# Import the functions we want to test
from hackathon_analysis.data_extraction.lauzhack_extractor import (
    _normalize_title,
    _build_dedup_key,
    _extract_footer_team_and_url,
    _extract_title_with_awards,
    _extract_descriptions,
)
from bs4 import BeautifulSoup


def test_normalize_title_removes_extra_spaces():
    """
    Test that _normalize_title removes extra spaces and converts to lowercase.

    What this function does:
    - Takes a project title that might have irregular spacing
    - Removes extra spaces (multiple spaces become one)
    - Converts to lowercase for consistent comparison
    - Strips leading/trailing whitespace

    Why we need this:
    - When comparing titles for deduplication, we want "Project  A" 
      and "project a" to be treated as the same title
    """
    # ARRANGE: Set up the input data
    messy_title = "  LauzHack   Project  2025  "
    # ACT: Call the function we're testing
    result = _normalize_title(messy_title)
    expected = "lauzhack project 2025"

    assert result == expected


def test_build_dedup_key_creates_normalized_key():
    """
    Test that _build_dedup_key creates a normalized, case-insensitive key.

    What this function does:
    - Combines title, description, URL, and team members into a single key
    - Normalizes each component (lowercase, extra spaces removed)
    - Joins them with pipe separators
    - Used to identify duplicate projects across years

    Why we need this:
    - If the same project is submitted in 2023 and 2024, we only keep the newest
    - The key helps us detect these duplicates reliably
    """
    # ARRANGE: Create a sample project
    title = "  AI Chat Bot  "
    description = "  A Smart   Chat Application  "
    url = "https://GitHub.com/Project/AI-Chat"
    team = ["Alice Smith", "Bob Jones"]

    # ACT: Build the dedup key
    result = _build_dedup_key(title, description, url, team)

    # ASSERT: Key should be normalized (lowercase, single spaces, consistent format)
    assert "ai chat bot" in result
    assert "a smart chat application" in result
    assert "https://github.com/project/ai-chat" in result
    assert "alice smith" in result
    assert "bob jones" in result


def test_extract_footer_team_and_url_with_valid_footer():
    """
    Test that _extract_footer_team_and_url correctly extracts URL and team members.

    What this function does:
    - Finds the link (URL) in a footer element
    - Extracts team member names from the first line
    - Handles comma-separated team lists
    - Returns tuple of (url, team_list)

    Why we need this:
    - Footer HTML structure varies between years
    - Extracting correctly ensures we capture all project metadata
    """
    # ARRANGE: Create mock HTML footer with URL and team
    html = """
    <footer>
        <a href="https://github.com/team/project">Project Link</a>
        <br>
        Alice Smith, Bob Jones, Charlie Brown
    </footer>
    """
    footer_elem = BeautifulSoup(html, "html.parser").find("footer")

    # ACT: Extract URL and team
    url, team = _extract_footer_team_and_url(footer_elem)

    # ASSERT: Should extract both correctly
    assert url == "https://github.com/team/project"
    assert team == ["Alice Smith", "Bob Jones", "Charlie Brown"]


def test_extract_footer_team_and_url_with_single_member():
    """
    Test extraction when there's only one team member (no commas).

    What this tests:
    - Footer with only one team member and no commas
    - Ensures single members are wrapped in a list
    - Different team member format variations
    """
    # ARRANGE: Create mock HTML with single team member
    html = """
    <footer>
        <a href="https://example.com/project">GitHub</a>
        <br>
        Solo Developer
    </footer>
    """
    footer_elem = BeautifulSoup(html, "html.parser").find("footer")

    # ACT: Extract
    url, team = _extract_footer_team_and_url(footer_elem)

    # ASSERT: Single member should be in a list
    assert url == "https://example.com/project"
    assert team == ["Solo Developer"]


def test_extract_footer_team_and_url_with_no_footer():
    """
    Test extraction when footer is None or empty.

    What this tests:
    - Handling of missing footer elements gracefully
    - Returns None for both url and team
    """
    # ACT: Call with None footer
    url, team = _extract_footer_team_and_url(None)

    # ASSERT: Should return None, None
    assert url is None
    assert team is None


def test_extract_title_with_awards_from_header():
    """
    Test extraction of title and awards from article header element.

    What this function does:
    - Finds title in <b> tag within <header>
    - Extracts award badges from <mark> tags
    - Sets both 'awards' and 'categories' to the same list
    - Handles missing header gracefully

    Why we need this:
    - 2025 LauzHack uses article-based structure with award badges
    - Awards are visual indicators in <mark> tags that we need to extract
    """
    # ARRANGE: Create mock HTML article with header, title, and awards
    html = """
    <article>
        <header>
            <b>AI Chat Assistant</b>
            <mark>Best AI</mark>
            <mark>People's Choice</mark>
        </header>
    </article>
    """
    element = BeautifulSoup(html, "html.parser").find("article")
    project = {}

    # ACT: Extract title and awards
    _extract_title_with_awards(element, project)

    # ASSERT: Should populate title and both awards fields
    assert project["title"] == "AI Chat Assistant"
    assert project["awards"] == ["Best AI", "People's Choice"]
    assert project["categories"] == ["Best AI", "People's Choice"]


def test_extract_title_with_awards_fallback_to_h2():
    """
    Test that title extraction falls back to h2 tag if no header <b> found.

    What this tests:
    - Fallback mechanism for different HTML structures
    - Tries h2, then h3, then h4 if no header <b>
    - Still works with minimal HTML
    """
    # ARRANGE: Create HTML without header, only h2
    html = """
    <article>
        <h2>Alternative Title Format</h2>
    </article>
    """
    element = BeautifulSoup(html, "html.parser").find("article")
    project = {}

    # ACT: Extract title
    _extract_title_with_awards(element, project)

    # ASSERT: Should extract from h2
    assert project["title"] == "Alternative Title Format"
    assert "awards" not in project


def test_extract_descriptions_from_content_divs():
    """
    Test description extraction from content divs.

    What this function does:
    - Tries multiple sources: details element, content divs, paragraphs
    - Combines multiple divs/paragraphs for richer descriptions
    - Returns early once a source is found
    - Filters out small divs (< 20 chars)

    Why we need this:
    - Different hackathon years use different HTML structures
    - Need robust fallback chain to capture descriptions
    """
    # ARRANGE: Create HTML with content divs (no details element)
    html = """
    <article>
        <div>A brief intro</div>
        <div>This is a detailed description about the project. It has enough content to be meaningful and interesting to read.</div>
        <div>More details about features and functionality</div>
    </article>
    """
    element = BeautifulSoup(html, "html.parser").find("article")
    project = {}

    # ACT: Extract descriptions
    _extract_descriptions(element, project)

    # ASSERT: Should combine the two largest divs
    assert "description" in project
    assert "detailed description" in project["description"]
    assert "features and functionality" in project["description"]


def test_extract_descriptions_from_paragraphs():
    """
    Test description extraction from paragraph fallback.

    What this tests:
    - Uses <p> tags when no details/divs available
    - Combines multiple paragraphs
    """
    # ARRANGE: Create HTML with only paragraphs
    html = """
    <article>
        <p>First paragraph of description.</p>
        <p>Second paragraph with more details.</p>
    </article>
    """
    element = BeautifulSoup(html, "html.parser").find("article")
    project = {}

    # ACT: Extract descriptions
    _extract_descriptions(element, project)

    # ASSERT: Should combine paragraphs
    assert "description" in project
    assert "First paragraph" in project["description"]
    assert "Second paragraph" in project["description"]


if __name__ == "__main__":
    # Run this test directly with: python tests/test_extraction_basic.py
    test_normalize_title_removes_extra_spaces()
    test_build_dedup_key_creates_normalized_key()
    test_extract_footer_team_and_url_with_valid_footer()
    test_extract_footer_team_and_url_with_single_member()
    test_extract_footer_team_and_url_with_no_footer()
    test_extract_title_with_awards_from_header()
    test_extract_title_with_awards_fallback_to_h2()
    test_extract_descriptions_from_content_divs()
    test_extract_descriptions_from_paragraphs()
    print("✓ All tests passed!")
