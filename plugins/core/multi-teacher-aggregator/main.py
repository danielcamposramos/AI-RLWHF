"""Multi-teacher reward aggregation plugin and helper utilities."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

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

DEFAULT_WEIGHTS: Dict[str, float] = {
    "grok": 0.3,
    "codex": 0.3,
    "kimi": 0.2,
    "glm": 0.2,
}
DEFAULT_METHOD = "weighted_average"
DEFAULT_THRESHOLD = 1.5
DEFAULT_LOG_PATH = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
DEFAULT_FALLBACK_MODE = "use_offline"
ScoreDict = Dict[str, float]
FeedbackDict = Dict[str, str]


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


def _normalize_weights(weights: Optional[Mapping[str, float]], teachers: Iterable[str]) -> Dict[str, float]:
    resolved = dict(DEFAULT_WEIGHTS)
    if weights:
        resolved.update({k: float(v) for k, v in weights.items()})
    filtered = {t: resolved.get(t, 0.0) for t in teachers}
    total = sum(filtered.values())
    if total <= 0:
        equal_weight = 1.0 / max(len(filtered), 1)
        return {t: equal_weight for t in filtered}
    return {t: val / total for t, val in filtered.items()}


def _weighted_average(scores: ScoreDict, weights: Mapping[str, float]) -> float:
    return sum(scores[t] * weights.get(t, 0.0) for t in scores)


def _majority_vote(scores: ScoreDict, weights: Mapping[str, float]) -> float:
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
        return _weighted_average({t: s for t, s in scores.items() if s > 0}, weights)
    if majority == "negative":
        return _weighted_average({t: s for t, s in scores.items() if s < 0}, weights)
    return 0.0


def _confidence_weighted(scores: ScoreDict) -> float:
    confidences = {t: abs(s) for t, s in scores.items()}
    total = sum(confidences.values())
    if total == 0:
        return 0.0
    norm = {t: (abs(scores[t]) / total) for t in scores}
    return _weighted_average(scores, norm)


def _filter_scores_by_active(
    scores: ScoreDict,
    notes: FeedbackDict,
    active_teachers: Optional[Sequence[str]],
    enable_internet_teachers: bool,
) -> Tuple[ScoreDict, FeedbackDict]:
    if not active_teachers:
        return scores, notes
    filtered_scores: ScoreDict = {}
    filtered_notes: FeedbackDict = {}
    active = {teacher.strip(): True for teacher in active_teachers if teacher.strip()}
    for teacher, score in scores.items():
        if teacher in active:
            filtered_scores[teacher] = score
            filtered_notes[teacher] = notes.get(teacher, "")
        elif enable_internet_teachers:
            filtered_scores.setdefault(teacher, score)
            filtered_notes.setdefault(teacher, notes.get(teacher, ""))
    return filtered_scores, filtered_notes


def aggregate_feedback(
    *,
    teacher_feedback: Optional[Mapping[str, Mapping[str, object]]] = None,
    teacher_weights: Optional[Mapping[str, float]] = None,
    aggregation_method: str = DEFAULT_METHOD,
    disagreement_threshold: float = DEFAULT_THRESHOLD,
    prompt: str = "",
    student_answer: str = "",
    log_path: Optional[Path] = DEFAULT_LOG_PATH,
    active_teachers: Optional[Sequence[str]] = None,
    enable_internet_teachers: bool = True,
    enable_offline_validation: bool = True,
    fallback_mode: str = DEFAULT_FALLBACK_MODE,
    progress_callback=None,
) -> AggregationResult:
    teacher_feedback = teacher_feedback or {}
    if not isinstance(teacher_feedback, Mapping):
        teacher_feedback = {}
    scores, notes = _sanitize_scores(teacher_feedback)
    scores, notes = _filter_scores_by_active(scores, notes, active_teachers, enable_internet_teachers)
    weights = _normalize_weights(teacher_weights, scores.keys()) if scores else {}

    method = aggregation_method or DEFAULT_METHOD
    if method == "majority_vote":
        aggregated = _majority_vote(scores, weights)
    elif method == "confidence_weighted":
        aggregated = _confidence_weighted(scores)
    else:
        aggregated = _weighted_average(scores, weights)

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
                "active_teachers": list(active_teachers or scores.keys()),
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

    result = aggregate_feedback(
        teacher_feedback=params.get("teacher_feedback"),
        teacher_weights=params.get("teacher_weights"),
        aggregation_method=params.get("aggregation_method", DEFAULT_METHOD),
        disagreement_threshold=float(params.get("disagreement_threshold", DEFAULT_THRESHOLD)),
        prompt=str(params.get("prompt", "")),
        student_answer=str(params.get("student_answer", "")),
        log_path=params.get("log_path", DEFAULT_LOG_PATH),
        active_teachers=[
            teacher.strip()
            for teacher in str(params.get("active_teachers", "")).split(",")
            if teacher.strip()
        ],
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
        "grok": {"score": 2, "feedback": "Accurate and sourced"},
        "codex": {"score": 1, "feedback": "Needs more detail"},
        "kimi": {"score": -1, "feedback": "Minor hallucination"},
    }
    summary = multi_teacher_aggregator(teacher_feedback=demo_feedback)
    print(json.dumps(summary, indent=2))
