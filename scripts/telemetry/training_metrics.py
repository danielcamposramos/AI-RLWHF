"""Telemetry helpers for tracking RLWHF training progress."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class TrainingMetricsLogger:
    """Persist batch- and session-level telemetry for honesty training."""

    output_dir: Path
    start_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stream_path = self.output_dir / "training_metrics.jsonl"
        self.summary_path = self.output_dir / "training_summary.json"

    def log_batch(self, batch_index: int, reward_snapshot: Dict[str, Any], hardware_usage: Dict[str, Any]) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_index": batch_index,
            "elapsed_seconds": time.time() - self.start_time,
            "reward_snapshot": reward_snapshot,
            "hardware_usage": hardware_usage,
        }
        with self.stream_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def finalize(self, total_batches: int, final_rewards: Dict[str, Any]) -> None:
        summary = {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_time_seconds": time.time() - self.start_time,
            "total_batches": total_batches,
            "final_rewards": final_rewards,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


__all__ = ["TrainingMetricsLogger"]
