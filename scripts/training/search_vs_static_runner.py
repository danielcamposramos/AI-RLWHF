"""CLI utility to compare Grok search (online) vs static/offline evaluations."""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from scripts.training.multi_teacher_runner import load_runner_config, run_batch_evaluation


def run_search_vs_static(
    prompts_path: Path,
    answers_path: Path,
    output_dir: Path,
    config_path: Path,
    delay: float,
) -> dict:
    """Runs and compares evaluations with and without internet-enabled search.

    This function performs two batch evaluations: one with internet search
    enabled and one with it disabled. It saves both sets of results and
    a combined summary.

    Args:
        prompts_path: Path to the prompts file.
        answers_path: Path to the student answers file.
        output_dir: Directory to save the output files.
        config_path: Path to the main feature toggles configuration.
        delay: Delay in seconds between evaluation runs.

    Returns:
        A dictionary containing the summary of both online and offline runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_runner_config(config_path)

    online_output = output_dir / "search_enabled.json"
    offline_output = output_dir / "search_disabled.json"

    summary_online = run_batch_evaluation(
        prompts_path,
        answers_path,
        online_output,
        base_config,
        delay=delay,
    )

    offline_config = replace(base_config, enable_internet_teachers=False)
    summary_offline = run_batch_evaluation(
        prompts_path,
        answers_path,
        offline_output,
        offline_config,
        delay=delay,
    )
    combined = {
        "online": summary_online,
        "offline": summary_offline,
        "config": {
            "prompts": str(prompts_path),
            "answers": str(answers_path),
            "feature_toggles": str(config_path),
        },
    }
    (output_dir / "search_vs_static_summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def main() -> None:  # pragma: no cover
    """Main entry point for the CLI script."""
    parser = argparse.ArgumentParser(description="Compare Grok search-enabled vs offline teacher evaluations.")
    parser.add_argument("--prompts", type=Path, default=Path("data/raw/live_bench_50.jsonl"))
    parser.add_argument("--answers", type=Path, default=Path("data/processed/student_answers.txt"))
    parser.add_argument("--output", type=Path, default=Path("experiments/visualizations/search_delta"))
    parser.add_argument("--config", type=Path, default=Path("configs/training/feature_toggles.json"))
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    summary = run_search_vs_static(args.prompts, args.answers, args.output, args.config, args.delay)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
