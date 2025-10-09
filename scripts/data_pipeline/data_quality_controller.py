"""Higher-level quality scoring utilities for RLWHF tuples."""
from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Dict, Iterable, List, Mapping, Tuple

TupleDict = Mapping[str, object]


class DataQualityController:
    """Compute aggregate quality scores to gate RLWHF datasets."""

    QUALITY_THRESHOLDS = {
        "min_reward_variance": 0.15,
        "max_duplicate_ratio": 0.1,
    }

    REQUIRED_METADATA_FIELDS = {"source_ai", "confidence_score", "rubric_dimension"}

    def validate_tuples(self, tuples: Iterable[TupleDict]) -> Tuple[bool, Dict[str, object]]:
        tuples_list = list(tuples)
        total = len(tuples_list)
        if total == 0:
            return False, {"issues": ["empty_dataset"]}
        rewards = [float(entry.get("reward", 0)) for entry in tuples_list if isinstance(entry.get("reward"), (int, float))]
        reward_variance = self._safe_variance(rewards)
        duplicates = self._count_duplicates(tuples_list)
        duplicate_ratio = duplicates / total if total else 0.0
        metadata_missing = sum(
            1 for entry in tuples_list if not self.REQUIRED_METADATA_FIELDS <= set(self._metadata(entry).keys())
        )
        quality_score = self._calculate_quality_score(total, reward_variance, duplicate_ratio, metadata_missing)
        issues = []
        if reward_variance < self.QUALITY_THRESHOLDS["min_reward_variance"]:
            issues.append("low_reward_variance")
        if duplicate_ratio > self.QUALITY_THRESHOLDS["max_duplicate_ratio"]:
            issues.append("high_duplicate_ratio")
        if metadata_missing:
            issues.append("metadata_missing")
        return quality_score >= 0.8 and not issues, {
            "total": total,
            "quality_score": round(quality_score, 3),
            "reward_variance": round(reward_variance, 4),
            "duplicate_ratio": round(duplicate_ratio, 4),
            "metadata_missing": metadata_missing,
            "issues": issues,
        }

    def _metadata(self, entry: TupleDict) -> Mapping[str, object]:
        metadata = entry.get("metadata")
        if isinstance(metadata, Mapping):
            return metadata
        return {}

    def _count_duplicates(self, tuples_list: List[TupleDict]) -> int:
        serialized = Counter()
        for entry in tuples_list:
            serialized[repr(sorted(entry.items()))] += 1
        return sum(count - 1 for count in serialized.values() if count > 1)

    def _calculate_quality_score(self, total: int, reward_var: float, duplicate_ratio: float, metadata_missing: int) -> float:
        metadata_penalty = metadata_missing / max(total, 1)
        components = [
            min(1.0, reward_var / max(self.QUALITY_THRESHOLDS["min_reward_variance"], 1e-6)),
            max(0.0, 1.0 - duplicate_ratio / max(self.QUALITY_THRESHOLDS["max_duplicate_ratio"], 1e-6)),
            max(0.0, 1.0 - metadata_penalty),
        ]
        return mean(components)

    def _safe_variance(self, values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        avg = mean(values)
        return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


__all__ = ["DataQualityController"]
