"""Metrics for contrastive honesty learning runs."""
from __future__ import annotations

import math
from typing import Any, Iterable, Sequence


def _as_vector(vector: Any) -> list[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(item) for item in vector]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def summarize_axis_losses(loss_payload: dict[str, Any]) -> dict[str, float]:
    """Normalize per-axis losses into simple floats."""
    result: dict[str, float] = {}
    for key in ("correctness", "honesty", "contrast", "total"):
        value = loss_payload.get(key, 0.0)
        if hasattr(value, "item"):
            value = value.item()
        result[f"loss_{key}"] = float(value)
    return result


def embedding_cluster_quality(embeddings: Sequence[Any], labels: Sequence[Any]) -> dict[str, float]:
    """Measure basic intra-cluster and inter-cluster separation."""
    vectors = [_as_vector(item) for item in embeddings]
    if not vectors or len(vectors) != len(labels):
        return {
            "intra_cluster_cosine": 0.0,
            "inter_cluster_cosine": 0.0,
            "embedding_cluster_separation": 0.0,
        }

    intra_scores: list[float] = []
    inter_scores: list[float] = []
    for idx, left in enumerate(vectors):
        for jdx in range(idx + 1, len(vectors)):
            score = _cosine(left, vectors[jdx])
            if labels[idx] == labels[jdx]:
                intra_scores.append(score)
            else:
                inter_scores.append(score)
    intra = sum(intra_scores) / len(intra_scores) if intra_scores else 0.0
    inter = sum(inter_scores) / len(inter_scores) if inter_scores else 0.0
    return {
        "intra_cluster_cosine": intra,
        "inter_cluster_cosine": inter,
        "embedding_cluster_separation": intra - inter,
    }


def honesty_calibration(records: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Correlation between overall honesty and overall correctness."""
    honesty: list[float] = []
    correctness: list[float] = []
    for record in records:
        decomp = record.get("decomposition") or {}
        if not isinstance(decomp, dict):
            continue
        try:
            honesty.append(float(decomp.get("overall_honesty")))
            correctness.append(float(decomp.get("overall_correctness")))
        except (TypeError, ValueError):
            continue
    if len(honesty) < 2 or len(correctness) < 2:
        return {"honesty_correctness_correlation": 0.0}

    mean_h = sum(honesty) / len(honesty)
    mean_c = sum(correctness) / len(correctness)
    numerator = sum((h - mean_h) * (c - mean_c) for h, c in zip(honesty, correctness))
    denom_h = math.sqrt(sum((h - mean_h) ** 2 for h in honesty))
    denom_c = math.sqrt(sum((c - mean_c) ** 2 for c in correctness))
    if denom_h <= 0.0 or denom_c <= 0.0:
        return {"honesty_correctness_correlation": 0.0}
    return {"honesty_correctness_correlation": numerator / (denom_h * denom_c)}


def hard_negative_difficulty_distribution(similarities: Sequence[float]) -> dict[str, float]:
    """Summarize similarity of selected hard negatives."""
    values = [float(item) for item in similarities]
    if not values:
        return {
            "hard_negative_similarity_avg": 0.0,
            "hard_negative_similarity_min": 0.0,
            "hard_negative_similarity_max": 0.0,
        }
    return {
        "hard_negative_similarity_avg": sum(values) / len(values),
        "hard_negative_similarity_min": min(values),
        "hard_negative_similarity_max": max(values),
    }


__all__ = [
    "embedding_cluster_quality",
    "hard_negative_difficulty_distribution",
    "honesty_calibration",
    "summarize_axis_losses",
]
