"""Repository-relative locations used by the final QIVER pipeline."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the repository containing both QIVER and the canonical recipe code."""
    return Path(__file__).resolve().parents[2]


def qiver_root() -> Path:
    return Path(__file__).resolve().parents[1]
