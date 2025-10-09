"""Consensus utilities for multi-specialist RLWHF chains."""
from __future__ import annotations

from collections import Counter
from typing import Dict


class ConsensusBuilder:
    """Aggregate specialist responses and report agreement metrics."""

    def build_consensus(self, responses: Dict[str, str]) -> Dict[str, object]:
        agreement_level = self._calculate_agreement(responses)
        consensus_points = self._extract_shared_tokens(responses)
        conflicting = self._identify_conflicts(responses, consensus_points)
        final = self._choose_recommendation(consensus_points, responses)
        return {
            "agreement_level": agreement_level,
            "consensus_points": consensus_points,
            "conflicting_viewpoints": conflicting,
            "final_recommendation": final,
        }

    def _calculate_agreement(self, responses: Dict[str, str]) -> float:
        if not responses:
            return 0.0
        normalized = [resp.strip().lower() for resp in responses.values()]
        most_common = Counter(normalized).most_common(1)[0][1]
        return most_common / len(responses)

    def _extract_shared_tokens(self, responses: Dict[str, str]) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for response in responses.values():
            tokens = {token.strip().lower() for token in response.split() if token}
            counter.update(tokens)
        threshold = max(1, len(responses) // 2)
        return {token: count for token, count in counter.items() if count >= threshold}

    def _identify_conflicts(self, responses: Dict[str, str], consensus_points: Dict[str, int]) -> Dict[str, str]:
        conflicts = {}
        for specialist, response in responses.items():
            if not consensus_points:
                conflicts[specialist] = response
                continue
            if not any(token in response.lower() for token in consensus_points.keys()):
                conflicts[specialist] = response
        return conflicts

    def _choose_recommendation(self, consensus_points: Dict[str, int], responses: Dict[str, str]) -> str:
        if not consensus_points and responses:
            return next(iter(responses.values()))
        sorted_tokens = sorted(consensus_points.items(), key=lambda item: (-item[1], item[0]))
        return " ".join(token for token, _ in sorted_tokens[:10])


__all__ = ["ConsensusBuilder"]
