"""Tests for utility helpers."""

from utils import build_workspace_config, chunk_text, slugify


def test_chunk_text_returns_overlapping_chunks() -> None:
    """Chunking should preserve overlap between successive chunks."""
    text = "abcdefghij" * 30
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) > 1
    assert chunks[0][-10:] == chunks[1][:10]


def test_chunk_text_rejects_invalid_overlap() -> None:
    """Chunking should reject invalid overlap settings."""
    try:
        chunk_text("hello", chunk_size=5, chunk_overlap=5)
    except ValueError as exc:
        assert "smaller than chunk_size" in str(exc)
    else:
        raise AssertionError("Expected ValueError was not raised")


def test_workspace_config_builds_scoped_paths() -> None:
    """Workspace config should derive isolated storage paths."""
    config = build_workspace_config(
        {"paths": {"workspace_root": "data/workspaces"}},
        "Team Handbook",
    )

    assert slugify("Team Handbook") == "team-handbook"
    assert config["workspace"]["slug"] == "team-handbook"
    assert config["paths"]["documents_dir"].endswith("team-handbook/raw/sources")
