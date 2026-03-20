"""Extract prompt-only GRPO training inputs from scored RLWHF tuples."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.data_pipeline.tlab_dataset_bridge import load_rlwhf_records


def extract_prompts(
    jsonl_path: str | Path,
    output_path: str | Path,
    min_responses: int = 2,
) -> int:
    """Write prompt-only JSONL rows for GRPO training."""
    rows = load_rlwhf_records(jsonl_path)
    counts = Counter(row["prompt"] for row in rows)
    prompts = sorted(prompt for prompt, count in counts.items() if count >= min_responses)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(
                json.dumps(
                    {"prompt": prompt, "response_count": counts[prompt]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(prompts)


__all__ = ["extract_prompts"]
