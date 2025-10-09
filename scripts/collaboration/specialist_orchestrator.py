"""Multi-specialist collaboration orchestration utilities."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional

from scripts.collaboration.consensus_builder import ConsensusBuilder


SpecialistCallback = Callable[[str], str]


@dataclass
class InteractionRecord:
    specialist: str
    prompt: str
    response: str
    timestamp: str


@dataclass
class SpecialistOrchestrator:
    """Coordinate multi-agent prompt chains."""

    specialists: List[str] = field(default_factory=lambda: ["codex", "grok", "qwen", "glm"])
    callbacks: Mapping[str, SpecialistCallback] = field(default_factory=dict)
    consensus_builder: ConsensusBuilder = field(default_factory=ConsensusBuilder)

    def initiate_chain(self, prompt: str, first_specialist: Optional[str] = None) -> List[InteractionRecord]:
        order = self._resolve_order(first_specialist)
        log: List[InteractionRecord] = []
        current_prompt = prompt
        for specialist in order:
            response = self._call_specialist(specialist, current_prompt)
            log.append(
                InteractionRecord(
                    specialist=specialist,
                    prompt=current_prompt,
                    response=response,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            current_prompt = response
        return log

    def build_consensus(self, log: Iterable[InteractionRecord]) -> Dict[str, object]:
        payload = {entry.specialist: entry.response for entry in log}
        return self.consensus_builder.build_consensus(payload)

    def _resolve_order(self, first_specialist: Optional[str]) -> List[str]:
        if first_specialist and first_specialist in self.specialists:
            head = [first_specialist]
            remainder = [spec for spec in self.specialists if spec != first_specialist]
            return head + remainder
        return list(self.specialists)

    def _call_specialist(self, specialist: str, prompt: str) -> str:
        if specialist in self.callbacks:
            return self.callbacks[specialist](prompt)
        return f"[{specialist} response to]: {prompt}"


__all__ = ["SpecialistOrchestrator", "InteractionRecord"]
