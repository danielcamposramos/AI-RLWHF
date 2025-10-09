"""Configurable multi-teacher evaluation runner with offline/internet toggles."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from plugins.core.multi_teacher_aggregator import multi_teacher_aggregator
from scripts.utils.config_loader import load_config
from scripts.utils.offline_scoring import load_offline_reference, score_against_reference

CONFIG_PATH = Path("configs/training/feature_toggles.json")


@dataclass
class TeacherSpec:
    name: str
    requires_internet: bool = False
    weight: float = 0.25
    dataset_key: Optional[str] = None


@dataclass
class RunnerConfig:
    enable_internet_teachers: bool = True
    enable_offline_validation: bool = True
    fallback_mode: str = "use_offline"
    offline_dataset_path: Optional[Path] = None
    aggregation_method: str = "weighted_average"
    disagreement_threshold: float = 1.5
    teacher_presets: List[TeacherSpec] = field(default_factory=list)

    @property
    def teacher_weights(self) -> Dict[str, float]:
        return {spec.name: spec.weight for spec in self.teacher_presets}


DEFAULT_PRESETS = [
    TeacherSpec(name="grok-search-evaluator", requires_internet=True, weight=0.4),
    TeacherSpec(name="codex", requires_internet=False, weight=0.2),
    TeacherSpec(name="kimi", requires_internet=False, weight=0.2),
    TeacherSpec(name="glm", requires_internet=False, weight=0.2),
]


def default_runner_config() -> RunnerConfig:
    cfg = RunnerConfig(
        enable_internet_teachers=True,
        enable_offline_validation=True,
        fallback_mode="use_offline",
        offline_dataset_path=Path("data/examples/offline_reference.jsonl"),
        teacher_presets=list(DEFAULT_PRESETS),
    )
    return cfg


def load_runner_config(path: Path | str = CONFIG_PATH) -> RunnerConfig:
    defaults = {
        "enable_internet_teachers": True,
        "enable_offline_validation": True,
        "fallback_mode": "use_offline",
        "offline_dataset_path": "data/examples/offline_reference.jsonl",
        "aggregation_method": "weighted_average",
        "disagreement_threshold": 1.5,
        "teacher_presets": [spec.__dict__ for spec in DEFAULT_PRESETS],
    }
    payload = load_config(path, defaults)
    presets_payload = payload.get("teacher_presets", [])
    teacher_presets: List[TeacherSpec] = []
    for item in presets_payload:
        if isinstance(item, Mapping):
            teacher_presets.append(
                TeacherSpec(
                    name=str(item.get("name", "teacher")),
                    requires_internet=bool(item.get("requires_internet", False)),
                    weight=float(item.get("weight", 0.25)),
                    dataset_key=item.get("dataset_key"),
                )
            )
    config = RunnerConfig(
        enable_internet_teachers=bool(payload.get("enable_internet_teachers", True)),
        enable_offline_validation=bool(payload.get("enable_offline_validation", True)),
        fallback_mode=str(payload.get("fallback_mode", "use_offline")),
        offline_dataset_path=Path(payload["offline_dataset_path"]).expanduser()
        if payload.get("offline_dataset_path")
        else None,
        aggregation_method=str(payload.get("aggregation_method", "weighted_average")),
        disagreement_threshold=float(payload.get("disagreement_threshold", 1.5)),
        teacher_presets=teacher_presets or list(DEFAULT_PRESETS),
    )
    return config


def _simulate_remote_score(prompt: str, student_answer: str) -> float:
    topical_bonus = 0.2 if any(token in prompt.lower() for token in ["latest", "news", "update"]) else 0.0
    length_score = min(len(student_answer) / 500.0, 1.0) * 2 - 1
    jitter = random.uniform(-0.3, 0.3)
    return max(-2.0, min(2.0, length_score + topical_bonus + jitter))


def _simulate_local_score(student_answer: str, bias: float = 0.0) -> float:
    base = min(len(student_answer) / 400.0, 1.0) * 2 - 1
    return max(-2.0, min(2.0, base + bias))


def _load_offline_map(config: RunnerConfig) -> Dict[str, str]:
    if not (config.enable_offline_validation and config.offline_dataset_path):
        return {}
    return load_offline_reference(config.offline_dataset_path)


def run_multi_teacher_loop(prompt: str, student_answer: str, config: RunnerConfig, offline_map: Mapping[str, str]) -> Dict[str, object]:
    teacher_feedback: MutableMapping[str, Dict[str, object]] = {}
    active_teachers: List[str] = []
    for spec in config.teacher_presets:
        using_internet = spec.requires_internet
        if using_internet and not config.enable_internet_teachers:
            if config.enable_offline_validation and offline_map and config.fallback_mode == "use_offline":
                score, feedback = score_against_reference(prompt, student_answer, offline_map)
            elif config.fallback_mode == "skip_missing":
                continue
            else:
                score, feedback = 0.0, "Internet disabled; placeholder neutral score."
        else:
            if using_internet:
                score = _simulate_remote_score(prompt, student_answer)
                feedback = f"Internet-enabled evaluator ({spec.name}) heuristic score."
            else:
                bias = 0.05 if "explain" in prompt.lower() else 0.0
                score = _simulate_local_score(student_answer, bias=bias)
                feedback = f"Local evaluator ({spec.name}) heuristic score."
            if config.enable_offline_validation and offline_map:
                offline_score, offline_feedback = score_against_reference(prompt, student_answer, offline_map)
                feedback = f"{feedback} Offline check: {offline_feedback}"
                if config.fallback_mode == "use_offline":
                    score = (score + offline_score) / 2.0
        teacher_feedback[spec.name] = {"score": score, "feedback": feedback}
        active_teachers.append(spec.name)
    aggregator_payload = multi_teacher_aggregator(
        teacher_feedback=teacher_feedback,
        teacher_weights=config.teacher_weights,
        aggregation_method=config.aggregation_method,
        disagreement_threshold=config.disagreement_threshold,
        active_teachers=active_teachers,
        enable_internet_teachers=config.enable_internet_teachers,
        enable_offline_validation=config.enable_offline_validation,
        fallback_mode=config.fallback_mode,
        prompt=prompt,
        student_answer=student_answer,
    )
    aggregator_payload["teacher_feedback"] = teacher_feedback
    return aggregator_payload


def run_batch_evaluation(
    prompts_file: Path,
    student_answers_file: Path,
    output_file: Path,
    config: RunnerConfig,
    delay: float = 0.0,
) -> Dict[str, object]:
    prompts = _load_lines(prompts_file)
    answers = _load_lines(student_answers_file)
    total = min(len(prompts), len(answers))
    offline_map = _load_offline_map(config)
    outputs: List[Dict[str, object]] = []
    for idx in range(total):
        result = run_multi_teacher_loop(prompts[idx], answers[idx], config, offline_map)
        result["prompt_id"] = idx
        outputs.append(result)
        if delay:
            import time

            time.sleep(delay)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return {"count": total, "output": str(output_file)}


def _load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-teacher aggregation with configurable toggles")
    parser.add_argument("--prompts", type=Path, default=Path("data/raw/prompts.txt"))
    parser.add_argument("--answers", type=Path, default=Path("data/processed/student_answers.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/multi_teacher_results.json"))
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--delay", type=float, default=0.0)
    args = parser.parse_args()
    runner_config = load_runner_config(args.config)
    summary = run_batch_evaluation(args.prompts, args.answers, args.output, runner_config, delay=args.delay)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "TeacherSpec",
    "RunnerConfig",
    "default_runner_config",
    "load_runner_config",
    "run_multi_teacher_loop",
    "run_batch_evaluation",
]
