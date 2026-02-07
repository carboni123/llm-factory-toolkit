"""Tests for critical untested error paths in provider.py.

These tests target error handling and edge cases that were identified
as gaps in the test coverage audit.
"""

from __future__ import annotations

import pytest

from llm_factory_toolkit.provider import _strip_urls


# =============================================================================
# _strip_urls() function tests
# =============================================================================


def test_strip_urls_removes_markdown_links() -> None:
    """Test that markdown links are stripped correctly."""
    text = "Check out [this link](https://example.com) for more info."
    result = _strip_urls(text)
    assert "https://example.com" not in result
    assert "[this link]" not in result
    assert "Check out for more info." in result


def test_strip_urls_removes_image_links() -> None:
    """Test that markdown image links are stripped."""
    text = "See ![alt text](https://example.com/image.png) here."
    result = _strip_urls(text)
    assert "https://example.com/image.png" not in result
    assert "![alt text]" not in result
    assert "See here." in result


def test_strip_urls_handles_empty_parens() -> None:
    """Test that empty parentheses left after URL removal are cleaned up."""
    text = "[link](url) and [another](url)"
    result = _strip_urls(text)
    assert "()" not in result
    assert "and" in result


def test_strip_urls_normalizes_whitespace() -> None:
    """Test that multiple spaces are normalized."""
    text = "Text    with     multiple  spaces"
    result = _strip_urls(text)
    assert "    " not in result
    assert "Text with multiple spaces" in result


def test_strip_urls_handles_punctuation_spacing() -> None:
    """Test that spaces before punctuation are removed."""
    text = "Hello , world ! How are you ?"
    result = _strip_urls(text)
    assert result == "Hello, world! How are you?"


def test_strip_urls_handles_empty_string() -> None:
    """Test that empty string returns empty string."""
    assert _strip_urls("") == ""


def test_strip_urls_complex_mixed_content() -> None:
    """Test that complex text with multiple patterns is handled correctly."""
    text = """
    This is a [link](https://example.com) and ![image](https://img.png).
    Multiple    spaces     should    be   normalized  .
    Trailing spaces should be removed   .
    """
    result = _strip_urls(text)
    # Should remove URLs
    assert "https://example.com" not in result
    assert "https://img.png" not in result
    # Should normalize spaces
    assert "    " not in result
    # Should fix punctuation spacing
    assert " ." not in result or result.strip().endswith(".")


def test_strip_urls_preserves_regular_text() -> None:
    """Test that regular text without links is preserved."""
    text = "This is normal text with no links."
    result = _strip_urls(text)
    assert result == text


def test_strip_urls_handles_multiple_links_same_line() -> None:
    """Test multiple links on the same line."""
    text = "See [link1](url1) and [link2](url2) and [link3](url3) here."
    result = _strip_urls(text)
    assert "url1" not in result
    assert "url2" not in result
    assert "url3" not in result
    assert "See and and here." in result


def test_strip_urls_handles_newlines_with_trailing_spaces() -> None:
    """Test that newlines with trailing spaces are handled."""
    text = "Line one  \nLine two   \nLine three"
    result = _strip_urls(text)
    # Should not have trailing spaces before newlines
    assert "  \n" not in result
    assert "   \n" not in result
