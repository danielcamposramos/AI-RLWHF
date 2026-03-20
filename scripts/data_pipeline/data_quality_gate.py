#!/usr/bin/env python3
"""Validate / clean RLWHF tuples before they reach GRPO.

Exit-0  -> dataset is good.
Exit-1  -> issues found (details printed).
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, Iterable

REQ_FIELDS = {"prompt", "answer", "feedback", "reward", "metadata"}
META_FIELDS = {"source_ai", "confidence_score", "rubric_dimension"}
DECOMPOSITION_FIELDS = {
    "positive_fragments",
    "negative_fragments",
    "honesty_signals",
    "missing_honesty",
    "overall_honesty",
    "overall_correctness",
}


def _is_number_in_range(value: Any, low: float = 0.0, high: float = 1.0) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return low <= numeric <= high


def _validate_fragment_list(
    fragments: Any,
    *,
    required_fields: Iterable[str],
    score_field: str | None = None,
    boolean_field: str | None = None,
) -> bool:
    if not isinstance(fragments, list):
        return False
    for item in fragments:
        if not isinstance(item, dict):
            return False
        for field in required_fields:
            if field not in item:
                return False
            if not isinstance(item[field], str) or not item[field].strip():
                return False
        if score_field and not _is_number_in_range(item.get(score_field)):
            return False
        if boolean_field and not isinstance(item.get(boolean_field), bool):
            return False
    return True


def validate_decomposition(decomposition: Any) -> bool:
    """Validate the optional contrastive decomposition schema."""
    if decomposition is None:
        return True
    if not isinstance(decomposition, dict):
        return False
    missing = DECOMPOSITION_FIELDS - set(decomposition.keys())
    if missing:
        return False
    if not _validate_fragment_list(
        decomposition.get("positive_fragments"),
        required_fields=("text", "category"),
        score_field="correctness",
    ):
        return False
    if not _validate_fragment_list(
        decomposition.get("negative_fragments"),
        required_fields=("text", "category", "correction"),
        score_field="correctness",
    ):
        return False
    if not _validate_fragment_list(
        decomposition.get("honesty_signals"),
        required_fields=("text",),
        score_field="honesty_score",
        boolean_field="appropriate",
    ):
        return False
    if not _validate_fragment_list(
        decomposition.get("missing_honesty"),
        required_fields=("claim", "reason"),
    ):
        return False
    if not _is_number_in_range(decomposition.get("overall_honesty")):
        return False
    if not _is_number_in_range(decomposition.get("overall_correctness")):
        return False
    return True


def validate(path: str) -> bool:
    good, bad = 0, 0
    report: Dict[str, int] = {}

    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                report["json_error"] = report.get("json_error", 0) + 1
                continue

            missing = REQ_FIELDS - set(obj.keys())
            if missing:
                bad += 1
                # Create a consistent key for reporting
                key = f"missing_{sorted(list(missing))[0]}"
                report[key] = report.get(key, 0) + 1
                continue

            meta = obj.get("metadata", {})
            missing_meta = META_FIELDS - set(meta.keys())
            if missing_meta:
                bad += 1
                report["incomplete_metadata"] = report.get("incomplete_metadata", 0) + 1
                continue

            # reward must be in {-2,-1,0,1,2}
            if obj.get("reward") not in {-2, -1, 0, 1, 2}:
                bad += 1
                report["invalid_reward"] = report.get("invalid_reward", 0) + 1
                continue

            if "decomposition" in obj and not validate_decomposition(obj.get("decomposition")):
                bad += 1
                report["invalid_decomposition"] = report.get("invalid_decomposition", 0) + 1
                continue

            good += 1

    total = good + bad
    if total == 0:
        print("EMPTY dataset")
        return False

    print(f"Quality report for {path}")
    print(f"  Total samples : {total}")
    print(f"  Good samples  : {good/total*100:.1f}% ({good})")
    print(f"  Bad samples   : {bad/total*100:.1f}% ({bad})")
    if report:
        print("  Issues        :", json.dumps(report, indent=2))

    return bad == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data_quality_gate.py <path_to_jsonl_file>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)
