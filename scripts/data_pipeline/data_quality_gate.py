#!/usr/bin/env python3
"""Rule-based RLWHF tuple validator used before launching ms-swift GRPO."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable

REQUIRED_FIELDS = {"prompt", "student_answer", "feedback", "reward", "metadata"}
REQUIRED_METADATA_FIELDS = {"source_ai", "confidence_score", "rubric_dimension"}
ALLOWED_REWARDS = {-2, -1, 0, 1, 2}


def _iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def validate(dataset_path: str) -> bool:
    path = Path(dataset_path)
    if not path.exists():
        print(f"[quality_gate] Missing dataset: {path}")
        return False
    total = 0
    error_counter: Counter[str] = Counter()
    for payload in _iter_jsonl(path):
        total += 1
        missing = REQUIRED_FIELDS - payload.keys()
        if missing:
            error_counter.update(f"missing_field:{field}" for field in missing)
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            error_counter["metadata_not_object"] += 1
            continue
        missing_meta = REQUIRED_METADATA_FIELDS - metadata.keys()
        if missing_meta:
            error_counter.update(f"missing_metadata:{field}" for field in missing_meta)
            continue
        reward = payload.get("reward")
        if reward not in ALLOWED_REWARDS:
            error_counter["invalid_reward"] += 1
    bad = sum(error_counter.values())
    if total == 0:
        print("[quality_gate] Empty dataset")
        return False
    good = total - bad
    print(f"[quality_gate] total={total} good={good} bad={bad}")
    if error_counter:
        print(f"[quality_gate] issues={dict(error_counter)}")
    return bad == 0


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: data_quality_gate.py <dataset.jsonl>")
        return 1
    ok = validate(sys.argv[1])
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
