"""Lightweight JSONL logger for Multi-Vibe Coding In Chain partners."""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict

def _resolve_log_root() -> Path:
    """Resolves the root directory for logs.

    Uses the AI_RLWHF_LOG_DIR environment variable if set, otherwise defaults
    to a 'logs' directory in the current workspace. Creates the directory
    if it does not exist.

    Returns:
        The resolved log directory path.
    """
    root = Path(os.environ.get("AI_RLWHF_LOG_DIR", "logs"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def log(partner: str, event: str, payload: Dict[str, Any] | None = None) -> Path:
    """Appends a structured JSON record for a partner and event.

    This function creates a timestamped log entry in a daily log file
    named according to the partner.

    Args:
        partner: The name of the AI partner or system component logging the event.
        event: A string describing the event type (e.g., 'start_task', 'finish_task').
        payload: An optional dictionary of serializable data to include in the log.

    Returns:
        The path to the log file that was written to.
    """
    payload = payload or {}
    iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    logfile = _resolve_log_root() / f"chain-{partner}-{_dt.date.today()}.jsonl"
    record = {"ts": iso, "partner": partner, "event": event, **payload}
    with logfile.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return logfile


__all__ = ["log"]
