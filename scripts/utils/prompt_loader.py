"""Prompt loading helpers for configurable system prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_prompt(path: Optional[str], fallback: str = "") -> str:
    """Loads a prompt from a text file with a fallback option.

    If the path is not provided or the file does not exist, the fallback
    string is returned.

    Args:
        path: The optional path to the prompt file.
        fallback: The string to return if the prompt cannot be loaded.

    Returns:
        The loaded prompt as a string.
    """
    if not path:
        return fallback
    prompt_path = Path(path)
    if not prompt_path.exists():
        return fallback
    text = prompt_path.read_text(encoding="utf-8").strip()
    return text or fallback


__all__ = ["load_prompt"]
