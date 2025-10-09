"""Python-friendly facade for the multi-teacher aggregator plugin."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parents[1] / "multi-teacher-aggregator"
MODULE_PATH = PLUGIN_DIR / "main.py"

if MODULE_PATH.exists():
    spec = importlib.util.spec_from_file_location("multi_teacher_aggregator_plugin", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules.setdefault("multi_teacher_aggregator_plugin", module)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    multi_teacher_aggregator = getattr(module, "multi_teacher_aggregator")
    aggregate_feedback = getattr(module, "aggregate_feedback")
else:  # pragma: no cover - defensive fallback
    raise ImportError(f"Expected plugin module at {MODULE_PATH}")

__all__ = ["multi_teacher_aggregator", "aggregate_feedback"]
