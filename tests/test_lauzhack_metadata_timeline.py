"""Tests for LauzHack metadata date range extraction."""

from hackathon_analysis.data_extraction.lauzhack_extractor import _parse_event_date_range


def test_parse_event_date_range_cross_month():
    """Parse a month-crossing event date range."""
    parsed = _parse_event_date_range("November 30 - December 1", 2024)
    assert parsed is not None
    start_date, end_date = parsed
    assert start_date.isoformat() == "2024-11-30"
    assert end_date.isoformat() == "2024-12-01"


def test_parse_event_date_range_same_month():
    """Parse a same-month event date range."""
    parsed = _parse_event_date_range("December 2-3", 2023)
    assert parsed is not None
    start_date, end_date = parsed
    assert start_date.isoformat() == "2023-12-02"
    assert end_date.isoformat() == "2023-12-03"
