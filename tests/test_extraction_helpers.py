"""
Basic test for LauzHack data extraction.

This test demonstrates how to test a simple helper function that normalizes project titles.
"""

# Import the function we want to test
from hackathon_analysis.data_extraction.lauzhack_extractor import _normalize_title


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

if __name__ == "__main__":
    # Run this test directly with: python tests/test_extraction_basic.py
    test_normalize_title_removes_extra_spaces()
    print("✓ Test passed!")
