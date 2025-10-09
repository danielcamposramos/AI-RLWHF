#!/usr/bin/env python3
"""
Validate / clean RLWHF tuples before they reach GRPO.
Exit-0  →  dataset is good.
Exit-1  →  issues found (details printed).
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

REQ_FIELDS = {"prompt", "answer", "feedback", "reward", "metadata"}
META_FIELDS = {"source_ai", "confidence_score", "rubric_dimension"}


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