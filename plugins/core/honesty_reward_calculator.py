"""Honesty-aware reward helpers shared across ms-swift integrations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

DEFAULT_RUBRIC_PATH = Path("docs/rlwhf-framework.md")
DEFAULT_SCORE_MAP: Mapping[int, float] = {
    2: 1.0,
    1: 0.5,
    0: 0.0,
    -1: -0.5,
    -2: -1.0,
}


@dataclass
class RewardBreakdown:
    teacher_score: float
    confidence_score: float
    uncertainty_bonus: float
    hallucination_penalty: float
    metadata_adjustment: float

    @property
    def total(self) -> float:
        return float(self.teacher_score + self.confidence_score + self.uncertainty_bonus + self.hallucination_penalty + self.metadata_adjustment)


class HonestyRewardCalculator:
    """Convert RLWHF rubric scores into smooth GRPO/DPO compatible rewards."""

    def __init__(
        self,
        rubric_config_path: Path | str = DEFAULT_RUBRIC_PATH,
        score_map: Optional[Mapping[int, float]] = None,
    ) -> None:
        self.rubric_config_path = Path(rubric_config_path)
        self.score_map = dict(score_map or DEFAULT_SCORE_MAP)
        self.keywords = self._load_rubric_keywords(self.rubric_config_path)

    def _load_rubric_keywords(self, path: Path) -> Mapping[str, Iterable[str]]:
        if not path.exists():
            return {"hallucination": ("hallucination", "fabrication"), "uncertainty": ("uncertain", "unsure")}
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        vocab = {
            "hallucination": tuple({token.strip() for token in ("hallucination", "fabricated", "made up") if token in text}),
            "uncertainty": tuple({token.strip() for token in ("uncertain", "not sure", "unsure") if token in text}),
            "honesty": tuple({token.strip() for token in ("honest", "transparent", "confidence") if token in text}),
        }
        return {key: value or default for key, value, default in (
            ("hallucination", vocab.get("hallucination"), ("hallucination", "fabricated")),
            ("uncertainty", vocab.get("uncertainty"), ("uncertain", "unsure")),
            ("honesty", vocab.get("honesty"), ("honest", "transparency")),
        )}

    def calculate_reward(
        self,
        rubric_score: int,
        confidence: Optional[float] = None,
        feedback: str | None = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> RewardBreakdown:
        base_score = self.score_map.get(rubric_score, 0.0)
        confidence_adj = self._confidence_adjustment(base_score, confidence)
        uncertainty_bonus = self._uncertainty_bonus(feedback or "")
        hallucination_penalty = self._hallucination_penalty(feedback or "")
        metadata_adj = self._metadata_adjustment(metadata or {})
        return RewardBreakdown(
            teacher_score=base_score,
            confidence_score=confidence_adj,
            uncertainty_bonus=uncertainty_bonus,
            hallucination_penalty=hallucination_penalty,
            metadata_adjustment=metadata_adj,
        )

    def _confidence_adjustment(self, base_score: float, confidence: Optional[float]) -> float:
        if confidence is None:
            return 0.0
        centered = confidence - 0.5
        if base_score >= 0:
            return centered * 0.4
        return -centered * 0.6

    def _uncertainty_bonus(self, feedback: str) -> float:
        text = feedback.lower()
        if not text:
            return 0.0
        if any(keyword in text for keyword in self.keywords["uncertainty"]):
            return 0.15
        return 0.0

    def _hallucination_penalty(self, feedback: str) -> float:
        text = feedback.lower()
        if not text:
            return 0.0
        if any(keyword in text for keyword in self.keywords["hallucination"]):
            return -0.3
        return 0.0

    def _metadata_adjustment(self, metadata: Mapping[str, object]) -> float:
        bonus = 0.0
        rubric_dimension = str(metadata.get("rubric_dimension", "")).lower()
        if rubric_dimension in {"honesty", "transparency"}:
            bonus += 0.05
        if metadata.get("confidence_score") is not None:
            bonus += self._confidence_adjustment(0.0, float(metadata["confidence_score"]))
        return bonus

    def summarize(self, breakdown: RewardBreakdown) -> Dict[str, float]:
        return {
            "teacher_score": breakdown.teacher_score,
            "confidence_score": breakdown.confidence_score,
            "uncertainty_bonus": breakdown.uncertainty_bonus,
            "hallucination_penalty": breakdown.hallucination_penalty,
            "metadata_adjustment": breakdown.metadata_adjustment,
            "total": breakdown.total,
        }


def reward_from_tuple(tuple_payload: Mapping[str, object], calculator: Optional[HonestyRewardCalculator] = None) -> Dict[str, float]:
    calculator = calculator or HonestyRewardCalculator()
    rubric_score = int(tuple_payload.get("reward", 0))
    metadata = tuple_payload.get("metadata", {}) if isinstance(tuple_payload.get("metadata"), Mapping) else {}
    breakdown = calculator.calculate_reward(
        rubric_score=rubric_score,
        confidence=float(metadata.get("confidence_score", 0.5)) if metadata else None,
        feedback=str(tuple_payload.get("feedback", "")),
        metadata=metadata,
    )
    payload = calculator.summarize(breakdown)
    payload["normalized_reward"] = math.tanh(payload["total"])
    return payload


__all__ = ["HonestyRewardCalculator", "RewardBreakdown", "reward_from_tuple"]
