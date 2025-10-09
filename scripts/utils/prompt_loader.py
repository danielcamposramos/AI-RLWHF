"""Prompt loading helpers for configurable system prompts."""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def load_prompt(path: Optional[str], fallback: str = "") -> str:
    if not path:
        return fallback
    prompt_path = Path(path)
    if not prompt_path.exists():
        return fallback
    text = prompt_path.read_text(encoding="utf-8").strip()
    return text or fallback


__all__ = ["load_prompt"]
