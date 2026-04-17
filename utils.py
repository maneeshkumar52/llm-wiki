"""Utility helpers for chunking, configuration, and logging."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


LOGGER_NAME = "llm_wiki"


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(level.upper())
        return logger

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("config.yaml must contain a top-level mapping")

    ollama_config = config.setdefault("ollama", {})
    if not isinstance(ollama_config, dict):
        raise ValueError("config.yaml field 'ollama' must be a mapping")

    env_overrides = {
        "base_url": os.getenv("OLLAMA_BASE_URL"),
        "chat_model": os.getenv("OLLAMA_CHAT_MODEL"),
    }
    for key, value in env_overrides.items():
        if value:
            ollama_config[key] = value

    return config


def slugify(value: str) -> str:
    """Convert a display name into a filesystem-safe slug."""
    lowered = value.strip().lower()
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    lowered = re.sub(r"-+", "-", lowered).strip("-")
    return lowered or "default"


def build_workspace_config(config: dict[str, Any], workspace_name: str) -> dict[str, Any]:
    """Return a config copy with workspace-scoped storage paths."""
    resolved = deepcopy(config)
    paths = resolved.setdefault("paths", {})
    workspace_root = Path(paths.get("workspace_root", "data/workspaces")) / slugify(
        workspace_name
    )

    paths["workspace_root"] = str(workspace_root)
    paths["vector_db_dir"] = str(workspace_root / "chroma")
    paths["documents_dir"] = str(workspace_root / "raw" / "sources")
    paths["wiki_dir"] = str(workspace_root / "wiki")
    paths["exports_dir"] = str(workspace_root / "exports")
    paths["conversations_dir"] = str(workspace_root / ".llm-wiki" / "chats")
    paths["manifest_name"] = "manifest.json"
    paths["collection_name"] = f"wiki_chunks_{slugify(workspace_name)}"
    resolved["workspace"] = {"name": workspace_name, "slug": slugify(workspace_name)}
    return resolved


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def hash_text(value: str) -> str:
    """Create a deterministic short hash for IDs and cache keys."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_hex(value: str) -> str:
    """Return the SHA256 digest for authentication and config helpers."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def format_bytes(size_bytes: int) -> str:
    """Format a byte count for display in the UI."""
    if size_bytes < 1024:
        return f"{size_bytes} B"

    units = ["KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    for unit in units:
        size /= 1024
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"

    return f"{size_bytes} B"


def truncate_text(text: str, max_chars: int = 220) -> str:
    """Shorten long text for compact previews."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1].rstrip()}…"


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split long text into overlapping chunks."""
    cleaned = "\n".join(line.rstrip() for line in text.splitlines()).strip()
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    length = len(cleaned)

    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(cleaned[start:end].strip())
        if end == length:
            break
        start = max(end - chunk_overlap, start + 1)

    return [chunk for chunk in chunks if chunk]