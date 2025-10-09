"""Configurable multi-teacher evaluation runner with API/local slot toggles."""
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
from scripts.utils.prompt_loader import load_prompt

CONFIG_PATH = Path("configs/training/feature_toggles.json")
ALLOWED_CONNECTIONS = {"api", "transformerlab_local", "ollama"}


@dataclass
class TeacherSlot:
    """Configuration for a single teacher slot in the runner.

    Attributes:
        label: The display name for the teacher slot.
        connection_type: The connection type (api, transformerlab_local, ollama).
        api_profile: The profile to use for API connections.
        transformerlab_profile: The profile for Transformer Lab local models.
        ollama_endpoint: The endpoint for Ollama models.
        model_hint: A hint for which model to use.
        weight: The weight assigned to this teacher's score.
        requires_internet: Explicitly sets if internet is required.
        system_prompt_path: Path to the system prompt file.
        api_context_ratio: Context ratio for API connections.
        ollama_context_ratio: Context ratio for Ollama connections.
        local_context_ratio: Context ratio for local connections.
    """
    label: str
    connection_type: str = "transformerlab_local"
    api_profile: str = ""
    transformerlab_profile: str = ""
    ollama_endpoint: str = "http://localhost:11434"
    model_hint: str = ""
    weight: float = 0.25
    requires_internet: Optional[bool] = None
    system_prompt_path: str = "configs/prompts/teacher/system.md"
    api_context_ratio: float = 0.66
    ollama_context_ratio: float = 1.33
    local_context_ratio: float = 1.0

    def normalized_connection(self) -> str:
        """Returns the sanitized, lowercased connection type.

        Defaults to 'transformerlab_local' if the type is not recognized.

        Returns:
            The normalized connection type string.
        """
        kind = self.connection_type.strip().lower()
        if kind not in ALLOWED_CONNECTIONS:
            kind = "transformerlab_local"
        return kind

    def internet_required(self) -> bool:
        """Determines if the slot requires an internet connection.

        This is determined by the `requires_internet` attribute if set,
        otherwise it defaults to True for 'api' connections.

        Returns:
            True if the slot requires internet, False otherwise.
        """
        if self.requires_internet is not None:
            return bool(self.requires_internet)
        return self.normalized_connection() == "api"

    def context_ratio(self) -> float:
        """Returns the appropriate context ratio for the connection type.

        Returns:
            The context ratio value for the slot's connection type.
        """
        if self.normalized_connection() == "api":
            return self.api_context_ratio
        if self.normalized_connection() == "ollama":
            return self.ollama_context_ratio
        return self.local_context_ratio

    def as_dict(self) -> Dict[str, object]:
        """Converts the teacher slot configuration to a serializable dictionary.

        Returns:
            A dictionary representation of the TeacherSlot instance.
        """
        return {
            "name": self.label,
            "connection_type": self.normalized_connection(),
            "api_profile": self.api_profile,
            "transformerlab_profile": self.transformerlab_profile,
            "ollama_endpoint": self.ollama_endpoint,
            "model_hint": self.model_hint,
            "weight": self.weight,
            "requires_internet": self.internet_required(),
            "system_prompt_path": self.system_prompt_path,
            "api_context_ratio": self.api_context_ratio,
            "ollama_context_ratio": self.ollama_context_ratio,
            "local_context_ratio": self.local_context_ratio,
        }


@dataclass
class RunnerConfig:
    """Configuration for the multi-teacher evaluation runner.

    Attributes:
        teacher_mode: The mode for teacher evaluation ('single' or 'multiple').
        teacher_count: The number of active teachers in 'multiple' mode.
        enable_internet_teachers: Whether to enable teachers requiring internet.
        enable_offline_validation: Whether to use offline reference data.
        fallback_mode: Behavior when a teacher fails ('use_offline', 'skip_missing').
        offline_dataset_path: Path to the offline reference dataset.
        aggregation_method: The method for aggregating scores.
        disagreement_threshold: The threshold for flagging high disagreement.
        teacher_slots: A list of configured teacher slots.
        teacher_prompt_path: Path to the default teacher system prompt.
        teacher_prompt: The loaded default teacher system prompt.
        active_slots: (Property) The list of currently active teacher slots.
        teacher_weights: (Property) A dictionary of weights for active teachers.
        teacher_names: (Property) A list of names for active teachers.
    """
    teacher_mode: str = "multiple"
    teacher_count: int = 4
    enable_internet_teachers: bool = True
    enable_offline_validation: bool = True
    fallback_mode: str = "use_offline"
    offline_dataset_path: Optional[Path] = None
    aggregation_method: str = "weighted_average"
    disagreement_threshold: float = 1.5
    teacher_slots: List[TeacherSlot] = field(default_factory=list)
    teacher_prompt_path: Path = Path("configs/prompts/teacher/system.md")
    teacher_prompt: str = ""

    @property
    def active_slots(self) -> List[TeacherSlot]:
        """Returns the list of currently active teacher slots based on the configuration."""
        slots = self.teacher_slots or DEFAULT_SLOTS
        limit = max(1, self.teacher_count if self.teacher_mode == "multiple" else 1)
        return slots[:limit]

    @property
    def teacher_weights(self) -> Dict[str, float]:
        """Returns a dictionary of weights for the active teachers."""
        return {slot.label: slot.weight for slot in self.active_slots}

    @property
    def teacher_names(self) -> List[str]:
        """Returns a list of names for the active teachers."""
        return [slot.label for slot in self.active_slots]


DEFAULT_SLOTS = [
    TeacherSlot(label="grok-search-evaluator", connection_type="api", api_profile="transformerlab_default", model_hint="grok-4", weight=0.4),
    TeacherSlot(label="codex", connection_type="transformerlab_local", transformerlab_profile="codex-default", weight=0.2),
    TeacherSlot(label="kimi", connection_type="transformerlab_local", transformerlab_profile="kimi-local", weight=0.2),
    TeacherSlot(label="glm", connection_type="transformerlab_local", transformerlab_profile="glm-local", weight=0.2),
]


def default_runner_config() -> RunnerConfig:
    """Creates a default runner configuration.

    Returns:
        A RunnerConfig object with default settings.
    """
    cfg = RunnerConfig(
        teacher_mode="multiple",
        teacher_count=4,
        enable_internet_teachers=True,
        enable_offline_validation=True,
        fallback_mode="use_offline",
        offline_dataset_path=Path("data/examples/offline_reference.jsonl"),
        teacher_slots=list(DEFAULT_SLOTS),
    )
    return cfg


def _coerce_slot(payload: Mapping[str, object]) -> TeacherSlot:
    """Coerces a dictionary into a TeacherSlot object.

    Args:
        payload: A dictionary containing teacher slot data.

    Returns:
        A TeacherSlot object.
    """
    return TeacherSlot(
        label=str(payload.get("label", payload.get("name", "teacher"))).strip() or "teacher",
        connection_type=str(payload.get("connection_type", "transformerlab_local")),
        api_profile=str(payload.get("api_profile", "")),
        transformerlab_profile=str(payload.get("transformerlab_profile", "")),
        ollama_endpoint=str(payload.get("ollama_endpoint", "http://localhost:11434")),
        model_hint=str(payload.get("model_hint", "")),
        weight=float(payload.get("weight", 0.25)),
        requires_internet=payload.get("requires_internet"),
        system_prompt_path=str(payload.get("system_prompt_path", "configs/prompts/teacher/system.md")),
        api_context_ratio=float(payload.get("api_context_ratio", 0.66)),
        ollama_context_ratio=float(payload.get("ollama_context_ratio", 1.33)),
        local_context_ratio=float(payload.get("local_context_ratio", 1.0)),
    )


def load_runner_config(path: Path | str = CONFIG_PATH) -> RunnerConfig:
    """Loads the runner configuration from a file.

    Args:
        path: The path to the configuration file.

    Returns:
        A RunnerConfig object.
    """
    defaults = {
        "teacher_mode": "multiple",
        "teacher_count": 4,
        "enable_internet_teachers": True,
        "enable_offline_validation": True,
        "fallback_mode": "use_offline",
        "offline_dataset_path": "data/examples/offline_reference.jsonl",
        "aggregation_method": "weighted_average",
        "disagreement_threshold": 1.5,
        "teacher_slots": [slot.as_dict() for slot in DEFAULT_SLOTS],
    }
    payload = load_config(path, defaults)
    slots_payload = payload.get("teacher_slots", [])
    teacher_slots = [_coerce_slot(item) for item in slots_payload if isinstance(item, Mapping)]
    teacher_prompt_path = Path(payload.get("teacher_prompt_path", "configs/prompts/teacher/system.md"))
    teacher_prompt = load_prompt(teacher_prompt_path, fallback="")
    config = RunnerConfig(
        teacher_mode=str(payload.get("teacher_mode", "multiple")).lower(),
        teacher_count=int(payload.get("teacher_count", 4)),
        enable_internet_teachers=bool(payload.get("enable_internet_teachers", True)),
        enable_offline_validation=bool(payload.get("enable_offline_validation", True)),
        fallback_mode=str(payload.get("fallback_mode", "use_offline")),
        offline_dataset_path=Path(payload["offline_dataset_path"]).expanduser()
        if payload.get("offline_dataset_path")
        else None,
        aggregation_method=str(payload.get("aggregation_method", "weighted_average")),
        disagreement_threshold=float(payload.get("disagreement_threshold", 1.5)),
        teacher_slots=teacher_slots or list(DEFAULT_SLOTS),
        teacher_prompt_path=teacher_prompt_path,
        teacher_prompt=teacher_prompt,
    )
    return config


def _simulate_remote_score(prompt: str, student_answer: str, system_prompt: str) -> float:
    """Simulates a score from a remote (API-based) teacher.

    This is a placeholder for a real API call.

    Args:
        prompt: The user prompt.
        student_answer: The student's answer.
        system_prompt: The system prompt given to the teacher.

    Returns:
        A simulated score between -2.0 and 2.0.
    """
    topical_bonus = 0.25 if any(token in prompt.lower() for token in ["latest", "today", "news", "update"]) else 0.0
    length_score = min(len(student_answer) / 500.0, 1.0) * 2 - 1
    prompt_bonus = 0.0
    if any(token in system_prompt.lower() for token in ["fact-check", "verify", "search"]):
        prompt_bonus += 0.2
    if any(token in system_prompt.lower() for token in ["creative", "brainstorm"]):
        prompt_bonus -= 0.1
    jitter = random.uniform(-0.3, 0.3)
    return max(-2.0, min(2.0, length_score + topical_bonus + prompt_bonus + jitter))


def _simulate_local_score(student_answer: str, system_prompt: str, bias: float = 0.0) -> float:
    """Simulates a score from a local teacher.

    This is a placeholder for a real local model inference.

    Args:
        student_answer: The student's answer.
        system_prompt: The system prompt given to the teacher.
        bias: An optional bias to apply to the score.

    Returns:
        A simulated score between -2.0 and 2.0.
    """
    base = min(len(student_answer) / 400.0, 1.0) * 2 - 1
    prompt_bonus = 0.0
    if any(token in system_prompt.lower() for token in ["concise", "brevity"]):
        if len(student_answer) > 300:
            prompt_bonus -= 0.5
    return max(-2.0, min(2.0, base + bias + prompt_bonus))


def _load_offline_map(config: RunnerConfig) -> Dict[str, str]:
    """Loads the offline reference map if offline validation is enabled.

    Args:
        config: The runner configuration.

    Returns:
        A dictionary mapping prompts to reference answers, or an empty dictionary.
    """
    if not (config.enable_offline_validation and config.offline_dataset_path):
        return {}
    return load_offline_reference(config.offline_dataset_path)


def _score_offline(prompt: str, student_answer: str, offline_map: Mapping[str, str], fallback_mode: str) -> Tuple[float, str]:
    """Scores an answer using only the offline reference data.

    Args:
        prompt: The user prompt.
        student_answer: The student's answer.
        offline_map: The offline reference data.
        fallback_mode: The configured fallback mode.

    Returns:
        A tuple containing the score and a feedback message.
    """
    score, feedback = score_against_reference(prompt, student_answer, offline_map)
    if fallback_mode == "use_offline":
        return score, f"Offline reference used: {feedback}"
    return 0.0, f"Offline reference available (passive): {feedback}"


def _score_with_slot(prompt: str, student_answer: str, slot: TeacherSlot, config: RunnerConfig, offline_map: Mapping[str, str]) -> Tuple[float, str]:
    """Scores an answer using a specific teacher slot, with fallbacks.

    Args:
        prompt: The user prompt.
        student_answer: The student's answer.
        slot: The teacher slot to use for scoring.
        config: The runner configuration.
        offline_map: The offline reference data.

    Returns:
        A tuple containing the score and a feedback message.
    """
    requires_internet = slot.internet_required()
    system_prompt = load_prompt(slot.system_prompt_path, fallback=config.teacher_prompt)

    if requires_internet and not config.enable_internet_teachers:
        if config.enable_offline_validation and offline_map:
            return _score_offline(prompt, student_answer, offline_map, config.fallback_mode)
        if config.fallback_mode == "skip_missing":
            return 0.0, f"Skipped {slot.label} (internet disabled)."
        return 0.0, f"Internet disabled; {slot.label} produced neutral score."

    if slot.normalized_connection() == "api":
        score = _simulate_remote_score(prompt, student_answer, system_prompt)
        feedback = f"API ({slot.api_profile or 'default'}) heuristic score."
    elif slot.normalized_connection() == "ollama":
        score = _simulate_local_score(student_answer, system_prompt, bias=0.1)
        feedback = f"Ollama endpoint {slot.ollama_endpoint} heuristic score."
    else:
        bias = 0.05 if slot.model_hint.lower().startswith("glm") else 0.0
        score = _simulate_local_score(student_answer, system_prompt, bias=bias)
        feedback = f"Transformer Lab local profile {slot.transformerlab_profile or 'default'} score."

    if config.enable_offline_validation and offline_map:
        offline_score, offline_feedback = score_against_reference(prompt, student_answer, offline_map)
        feedback = f"{feedback} Offline check: {offline_feedback}"
        if config.fallback_mode == "use_offline":
            score = (score + offline_score) / 2.0
    return score, feedback


def run_multi_teacher_loop(prompt: str, student_answer: str, config: RunnerConfig, offline_map: Mapping[str, str]) -> Dict[str, object]:
    """Runs a single evaluation loop for a prompt and answer.

    This function simulates scoring from multiple teachers and aggregates the results.

    Args:
        prompt: The user prompt.
        student_answer: The student's answer.
        config: The runner configuration.
        offline_map: The offline reference data.

    Returns:
        A dictionary containing the aggregated evaluation results.
    """
    teacher_feedback: MutableMapping[str, Dict[str, object]] = {}
    for slot in config.active_slots:
        score, feedback = _score_with_slot(prompt, student_answer, slot, config, offline_map)
        if config.fallback_mode == "skip_missing" and feedback.startswith("Skipped"):
            continue
        teacher_feedback[slot.label] = {
            "score": score,
            "feedback": feedback,
            "connection_type": slot.normalized_connection(),
            "system_prompt_path": slot.system_prompt_path,
            "context_ratio": slot.context_ratio(),
        }

    aggregator_payload = multi_teacher_aggregator(
        teacher_feedback=teacher_feedback,
        teacher_weights=config.teacher_weights,
        aggregation_method=config.aggregation_method,
        disagreement_threshold=config.disagreement_threshold,
        active_teachers=config.teacher_names,
        enable_internet_teachers=config.enable_internet_teachers,
        enable_offline_validation=config.enable_offline_validation,
        fallback_mode=config.fallback_mode,
        teacher_mode=config.teacher_mode,
        teacher_count=config.teacher_count,
        teacher_slots=[slot.as_dict() for slot in config.active_slots],
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
    """Runs a batch evaluation over a set of prompts and answers.

    Args:
        prompts_file: Path to a file containing prompts, one per line.
        student_answers_file: Path to a file containing answers, one per line.
        output_file: Path to write the JSON output of the evaluation.
        config: The runner configuration.
        delay: An optional delay between processing each item.

    Returns:
        A dictionary summarizing the batch run.
    """
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
    """Loads non-empty lines from a text file.

    Args:
        path: The path to the text file.

    Returns:
        A list of strings.
    """
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:  # pragma: no cover - CLI helper
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description="Run multi-teacher aggregation with configurable slots")
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
    "TeacherSlot",
    "RunnerConfig",
    "default_runner_config",
    "load_runner_config",
    "run_multi_teacher_loop",
    "run_batch_evaluation",
]
