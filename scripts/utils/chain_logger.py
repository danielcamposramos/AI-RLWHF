"""Lightweight JSONL logger for Multi-Vibe Coding In Chain partners."""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Any, Dict

LOG_ROOT = Path(os.environ.get("AI_RLWHF_LOG_DIR", "logs"))
LOG_ROOT.mkdir(parents=True, exist_ok=True)


def log(partner: str, event: str, payload: Dict[str, Any] | None = None) -> Path:
    """Append a structured JSON record for the given partner and event."""
    payload = payload or {}
    iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")
    logfile = LOG_ROOT / f"chain-{partner}-{_dt.date.today()}.jsonl"
    record = {"ts": iso, "partner": partner, "event": event, **payload}
    with logfile.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return logfile


__all__ = ["log"]
