"""Triplet mining utilities for contrastive honesty learning."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass
class Triplet:
    """Contrastive triplet extracted from teacher-scored RLWHF tuples."""

    anchor: str
    positive: str
    negative: str
    axis: str
    margin: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_vector(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    raise TypeError(f"Unsupported embedding type: {type(value)!r}")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    norm_left = math.sqrt(sum(float(a) * float(a) for a in left))
    norm_right = math.sqrt(sum(float(b) * float(b) for b in right))
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def mine_hard_negatives(
    anchor_embed: Any,
    negative_pool: Sequence[Any],
    k: int = 7,
    min_similarity: float = 0.0,
) -> list[Any]:
    """Select the most similar negatives for a given anchor embedding."""
    anchor = _as_vector(anchor_embed)
    ranked: list[tuple[float, Any]] = []
    for candidate in negative_pool:
        vector = _as_vector(candidate)
        score = _cosine_similarity(anchor, vector)
        if score >= float(min_similarity):
            ranked.append((score, candidate))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in ranked[: max(1, int(k))]]


def _text_length_ok(text: Any, min_length: int) -> bool:
    return isinstance(text, str) and len(text.strip()) >= int(min_length)


def _load_reward(record: dict[str, Any]) -> float:
    try:
        return float(record.get("reward", 0))
    except (TypeError, ValueError):
        return 0.0


def _answer_text(record: dict[str, Any]) -> str:
    return str(record.get("answer") or record.get("student_answer") or "").strip()


def mine_triplets(
    tuples: list[dict[str, Any]],
    *,
    min_reward_gap: float = 1.0,
    max_triplets_per_prompt: int = 10,
    fragment_min_length: int = 10,
) -> list[Triplet]:
    """Mine correctness, honesty, and cross-response triplets from tuples."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in tuples:
        prompt = str(record.get("prompt") or "").strip()
        if prompt:
            grouped[prompt].append(record)

    triplets: list[Triplet] = []
    seen: set[tuple[str, str, str, str]] = set()

    for prompt, responses in grouped.items():
        ranked = sorted(responses, key=_load_reward, reverse=True)

        # Axis 3: better response vs worse response to the same prompt.
        prompt_triplets = 0
        for index, better in enumerate(ranked):
            better_answer = _answer_text(better)
            better_reward = _load_reward(better)
            if not better_answer:
                continue
            for worse in ranked[index + 1 :]:
                worse_answer = _answer_text(worse)
                worse_reward = _load_reward(worse)
                if not worse_answer:
                    continue
                reward_gap = better_reward - worse_reward
                if reward_gap < float(min_reward_gap):
                    continue
                key = (prompt, better_answer, worse_answer, "contrast")
                if key in seen:
                    continue
                seen.add(key)
                triplets.append(
                    Triplet(
                        anchor=prompt,
                        positive=better_answer,
                        negative=worse_answer,
                        axis="contrast",
                        margin=reward_gap,
                    )
                )
                prompt_triplets += 1
                if prompt_triplets >= int(max_triplets_per_prompt):
                    break
            if prompt_triplets >= int(max_triplets_per_prompt):
                break

        # Axis 1 + 2: fragment-level triplets from decomposition.
        for response in responses:
            decomposition = response.get("decomposition") or {}
            if not isinstance(decomposition, dict):
                continue

            positive_fragments = [
                fragment
                for fragment in decomposition.get("positive_fragments", [])
                if _text_length_ok(fragment.get("text"), fragment_min_length)
            ]
            negative_fragments = [
                fragment
                for fragment in decomposition.get("negative_fragments", [])
                if _text_length_ok(fragment.get("text"), fragment_min_length)
            ]
            honesty_signals = [
                fragment
                for fragment in decomposition.get("honesty_signals", [])
                if _text_length_ok(fragment.get("text"), fragment_min_length)
                and bool(fragment.get("appropriate"))
            ]
            missing_honesty = [
                fragment
                for fragment in decomposition.get("missing_honesty", [])
                if _text_length_ok(fragment.get("claim"), fragment_min_length)
            ]

            for pos in positive_fragments:
                for neg in negative_fragments:
                    key = (prompt, str(pos["text"]).strip(), str(neg["text"]).strip(), "correctness")
                    if key in seen:
                        continue
                    seen.add(key)
                    triplets.append(
                        Triplet(
                            anchor=prompt,
                            positive=str(pos["text"]).strip(),
                            negative=str(neg["text"]).strip(),
                            axis="correctness",
                        )
                    )

            response_anchor = _answer_text(response) or prompt
            for honest in honesty_signals:
                for missing in missing_honesty:
                    key = (
                        response_anchor,
                        str(honest["text"]).strip(),
                        str(missing["claim"]).strip(),
                        "honesty",
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    triplets.append(
                        Triplet(
                            anchor=response_anchor,
                            positive=str(honest["text"]).strip(),
                            negative=str(missing["claim"]).strip(),
                            axis="honesty",
                        )
                    )
    return triplets


def load_tuples(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL tuples from disk."""
    records: list[dict[str, Any]] = []
    source = Path(path)
    if not source.exists():
        return records
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


__all__ = ["Triplet", "load_tuples", "mine_hard_negatives", "mine_triplets"]
