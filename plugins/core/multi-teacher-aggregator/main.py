"""Multi-teacher reward aggregation plugin with dynamic slot handling."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency for live Transformer Lab runs
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover - fallback for local tooling/tests
    class _DummyTrainer:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.params: Dict[str, object] = {}

        def job_wrapper(self):  # type: ignore[override]
            def decorator(func):
                return func

            return decorator

        def progress_update(self, *_args, **_kwargs) -> None:
            return None

    tlab_trainer = _DummyTrainer()

MAX_SLOTS = 6
DEFAULT_METHOD = "weighted_average"
DEFAULT_THRESHOLD = 1.5
DEFAULT_FALLBACK_MODE = "use_offline"
DEFAULT_LOG_PATH = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
ALLOWED_CONNECTIONS = {"api", "transformerlab_local", "ollama"}
DEFAULT_SLOT_CONFIGS = [
    {
        "name": "grok-search-evaluator",
        "connection_type": "api",
        "api_profile": "transformerlab_default",
        "model_hint": "grok-4",
        "weight": 0.4,
    },
    {
        "name": "codex",
        "connection_type": "transformerlab_local",
        "weight": 0.2,
    },
    {
        "name": "kimi",
        "connection_type": "transformerlab_local",
        "weight": 0.2,
    },
    {
        "name": "glm",
        "connection_type": "transformerlab_local",
        "weight": 0.2,
    },
]
ScoreDict = Dict[str, float]
FeedbackDict = Dict[str, str]


@dataclass
class SlotConfig:
    name: str
    connection_type: str = "transformerlab_local"
    api_profile: str = ""
    transformerlab_profile: str = ""
    ollama_endpoint: str = ""
    model_hint: str = ""
    weight: float = 0.25

    @property
    def requires_internet(self) -> bool:
        return self.connection_type == "api"

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "connection_type": self.connection_type,
            "api_profile": self.api_profile,
            "transformerlab_profile": self.transformerlab_profile,
            "ollama_endpoint": self.ollama_endpoint,
            "model_hint": self.model_hint,
            "weight": self.weight,
            "requires_internet": self.requires_internet,
        }


def _sanitize_connection(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in ALLOWED_CONNECTIONS:
        normalized = "transformerlab_local"
    return normalized


def _extract_slot_configs(
    teacher_slots: Optional[Sequence[Mapping[str, object]]],
    slot_params: Mapping[str, object],
    teacher_mode: str,
    teacher_count: int,
) -> List[SlotConfig]:
    slots: List[SlotConfig] = []
    iterable: Sequence[Mapping[str, object]]
    if teacher_slots:
        iterable = teacher_slots
    else:
        dynamic_slots: List[Mapping[str, object]] = []
        for idx in range(1, MAX_SLOTS + 1):
            label = str(slot_params.get(f"teacher_slot{idx}_label", "")).strip()
            if not label:
                continue
            dynamic_slots.append(
                {
                    "name": label,
                    "connection_type": slot_params.get(f"teacher_slot{idx}_connection_type", "transformerlab_local"),
                    "api_profile": slot_params.get(f"teacher_slot{idx}_api_profile", ""),
                    "transformerlab_profile": slot_params.get(f"teacher_slot{idx}_transformerlab_profile", ""),
                    "ollama_endpoint": slot_params.get(f"teacher_slot{idx}_ollama_endpoint", ""),
                    "model_hint": slot_params.get(f"teacher_slot{idx}_model_hint", ""),
                    "weight": slot_params.get(f"teacher_slot{idx}_weight", 0.25),
                }
            )
        iterable = dynamic_slots or DEFAULT_SLOT_CONFIGS
    limit = max(1, teacher_count if teacher_mode == "multiple" else 1)
    for slot_data in iterable[:limit]:
        name = str(slot_data.get("name", "")).strip()
        if not name:
            continue
        slots.append(
            SlotConfig(
                name=name,
                connection_type=_sanitize_connection(str(slot_data.get("connection_type", "transformerlab_local"))),
                api_profile=str(slot_data.get("api_profile", "")),
                transformerlab_profile=str(slot_data.get("transformerlab_profile", "")),
                ollama_endpoint=str(slot_data.get("ollama_endpoint", "")),
                model_hint=str(slot_data.get("model_hint", "")),
                weight=float(slot_data.get("weight", 0.25)),
            )
        )
    if not slots:
        slots.append(SlotConfig(name="teacher"))
    return slots


def _sanitize_scores(raw_feedback: Mapping[str, Mapping[str, object]]) -> Tuple[ScoreDict, FeedbackDict]:
    scores: ScoreDict = {}
    feedback: FeedbackDict = {}
    for teacher, payload in raw_feedback.items():
        if not isinstance(payload, Mapping):
            continue
        score = float(payload.get("score", 0))
        score = max(-2.0, min(2.0, score))
        scores[teacher] = score
        feedback[teacher] = str(payload.get("feedback", ""))
    return scores, feedback


def _normalize_weights(weights: Optional[Mapping[str, float]], teachers: Iterable[str], slots: Sequence[SlotConfig]) -> Dict[str, float]:
    resolved: Dict[str, float] = {slot.name: slot.weight for slot in slots}
    if weights:
        resolved.update({k: float(v) for k, v in weights.items()})
    filtered = {t: resolved.get(t, 0.0) for t in teachers}
    total = sum(filtered.values())
    if total <= 0:
        equal_weight = 1.0 / max(len(filtered), 1)
        return {t: equal_weight for t in filtered}
    return {t: val / total for t, val in filtered.items()}


def _filter_scores_by_slots(scores: ScoreDict, notes: FeedbackDict, slots: Sequence[SlotConfig]) -> Tuple[ScoreDict, FeedbackDict]:
    slot_names = {slot.name for slot in slots}
    if not slot_names:
        return scores, notes
    filtered_scores: ScoreDict = {}
    filtered_notes: FeedbackDict = {}
    for teacher, score in scores.items():
        if teacher in slot_names:
            filtered_scores[teacher] = score
            filtered_notes[teacher] = notes.get(teacher, "")
    if not filtered_scores:
        return scores, notes
    return filtered_scores, filtered_notes


def _majority_vote(scores: ScoreDict, weights: Mapping[str, float]) -> float:
    from collections import defaultdict

    vote_counts: MutableMapping[str, float] = defaultdict(float)
    for teacher, score in scores.items():
        if score > 0:
            vote_counts["positive"] += weights.get(teacher, 0.0)
        elif score < 0:
            vote_counts["negative"] += weights.get(teacher, 0.0)
        else:
            vote_counts["neutral"] += weights.get(teacher, 0.0)
    if not vote_counts:
        return 0.0
    majority = max(vote_counts, key=vote_counts.get)
    if majority == "positive":
        return sum(scores[t] * weights.get(t, 0.0) for t in scores if scores[t] > 0)
    if majority == "negative":
        return sum(scores[t] * weights.get(t, 0.0) for t in scores if scores[t] < 0)
    return 0.0


def _confidence_weighted(scores: ScoreDict) -> float:
    confidences = {t: abs(s) for t, s in scores.items()}
    total = sum(confidences.values())
    if total == 0:
        return 0.0
    norm = {t: confidences[t] / total for t in scores}
    return sum(scores[t] * norm.get(t, 0.0) for t in scores)


@dataclass
class AggregationResult:
    aggregated_score: float
    disagreement: float
    high_disagreement: bool
    combined_feedback: str
    individual_scores: ScoreDict

    def to_payload(self) -> Dict[str, object]:
        return {
            "aggregated_score": self.aggregated_score,
            "disagreement": self.disagreement,
            "high_disagreement": self.high_disagreement,
            "combined_feedback": self.combined_feedback,
            "individual_scores": self.individual_scores,
        }


def aggregate_feedback(
    *,
    teacher_feedback: Optional[Mapping[str, Mapping[str, object]]] = None,
    teacher_weights: Optional[Mapping[str, float]] = None,
    aggregation_method: str = DEFAULT_METHOD,
    disagreement_threshold: float = DEFAULT_THRESHOLD,
    prompt: str = "",
    student_answer: str = "",
    log_path: Optional[Path] = DEFAULT_LOG_PATH,
    teacher_slots: Optional[Sequence[Mapping[str, object]]] = None,
    slot_params: Optional[Mapping[str, object]] = None,
    teacher_mode: str = "multiple",
    teacher_count: int = MAX_SLOTS,
    enable_internet_teachers: bool = True,
    enable_offline_validation: bool = True,
    fallback_mode: str = DEFAULT_FALLBACK_MODE,
    progress_callback=None,
) -> AggregationResult:
    teacher_feedback = teacher_feedback or {}
    if not isinstance(teacher_feedback, Mapping):
        teacher_feedback = {}
    slot_param_map = slot_params or {}
    slots = _extract_slot_configs(teacher_slots, slot_param_map, teacher_mode, int(teacher_count))
    scores, notes = _sanitize_scores(teacher_feedback)
    scores, notes = _filter_scores_by_slots(scores, notes, slots)
    weights = _normalize_weights(teacher_weights, scores.keys(), slots) if scores else {}

    method = aggregation_method or DEFAULT_METHOD
    if method == "majority_vote":
        aggregated = _majority_vote(scores, weights)
    elif method == "confidence_weighted":
        aggregated = _confidence_weighted(scores)
    else:
        aggregated = sum(scores[t] * weights.get(t, 0.0) for t in scores)

    disagreement = 0.0
    if len(scores) > 1:
        values = list(scores.values())
        disagreement = max(values) - min(values)
    high_disagreement = disagreement > disagreement_threshold

    combined_feedback = [f"Aggregated score: {aggregated:.2f}"]
    combined_feedback.append(f"Disagreement: {disagreement:.2f} (threshold: {disagreement_threshold:.2f})")
    if notes:
        combined_feedback.append("Individual feedback:")
        combined_feedback.extend(f"- {teacher}: {notes[teacher]}" for teacher in sorted(notes))
    feedback_text = "\n".join(combined_feedback)

    if log_path:
        log_entry = {
            "prompt": prompt,
            "student_answer": student_answer,
            "aggregated_score": aggregated,
            "individual_scores": scores,
            "disagreement": disagreement,
            "high_disagreement": high_disagreement,
            "disagreement_threshold": disagreement_threshold,
            "combined_feedback": feedback_text,
            "config": {
                "teacher_mode": teacher_mode,
                "teacher_slots": [slot.to_dict() for slot in slots],
                "enable_internet_teachers": enable_internet_teachers,
                "enable_offline_validation": enable_offline_validation,
                "fallback_mode": fallback_mode,
            },
        }
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    if progress_callback:
        progress_callback(100)

    return AggregationResult(
        aggregated_score=aggregated,
        disagreement=disagreement,
        high_disagreement=high_disagreement,
        combined_feedback=feedback_text,
        individual_scores=scores,
    )


@tlab_trainer.job_wrapper()
def multi_teacher_aggregator(**overrides):
    """Entry point compatible with Transformer Lab and direct invocation."""
    params: Dict[str, object] = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    params.update(overrides)

    progress_cb = getattr(tlab_trainer, "progress_update", None)
    if callable(progress_cb):
        progress_cb(25)

    teacher_mode = str(params.get("teacher_mode", "multiple")).lower()
    teacher_count = int(params.get("teacher_count", MAX_SLOTS))
    slot_payload = params.get("teacher_slots")
    if isinstance(slot_payload, str):
        try:
            slot_payload = json.loads(slot_payload)
        except Exception:  # pragma: no cover - invalid payload fallback
            slot_payload = None

    result = aggregate_feedback(
        teacher_feedback=params.get("teacher_feedback"),
        teacher_weights=params.get("teacher_weights"),
        aggregation_method=params.get("aggregation_method", DEFAULT_METHOD),
        disagreement_threshold=float(params.get("disagreement_threshold", DEFAULT_THRESHOLD)),
        prompt=str(params.get("prompt", "")),
        student_answer=str(params.get("student_answer", "")),
        log_path=params.get("log_path", DEFAULT_LOG_PATH),
        teacher_slots=slot_payload,
        slot_params=params,
        teacher_mode=teacher_mode,
        teacher_count=teacher_count,
        enable_internet_teachers=bool(params.get("enable_internet_teachers", True)),
        enable_offline_validation=bool(params.get("enable_offline_validation", True)),
        fallback_mode=str(params.get("fallback_mode", DEFAULT_FALLBACK_MODE)),
        progress_callback=None,
    )

    if callable(progress_cb):
        progress_cb(100)

    payload = result.to_payload()
    return payload


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    demo_feedback = {
        "grok-search-evaluator": {"score": 2, "feedback": "Accurate and sourced"},
        "codex": {"score": 1, "feedback": "Needs more detail"},
        "kimi": {"score": -1, "feedback": "Minor hallucination"},
    }
    summary = multi_teacher_aggregator(teacher_feedback=demo_feedback)
    print(json.dumps(summary, indent=2))
