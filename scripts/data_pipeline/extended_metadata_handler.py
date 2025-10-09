"""Extend metadata payloads used across RLWHF tuple generation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Mapping

from plugins.core.hardware_detector import HardwareDetector


class ExtendedMetadataHandler:
    """Attach longitudinal metadata for downstream analytics."""

    EXTENDED_FIELDS = ("iteration_count", "consensus_score", "hardware_profile", "update_timestamp")

    def __init__(self) -> None:
        self.detector = HardwareDetector()

    def extend_metadata(self, metadata: Mapping[str, object]) -> Dict[str, object]:
        payload = dict(metadata)
        payload["iteration_count"] = int(payload.get("iteration_count", 0)) + 1
        payload.setdefault("consensus_score", float(payload.get("consensus_score", 1.0)))
        payload["hardware_profile"] = payload.get("hardware_profile") or self._resolve_hardware_label()
        payload["update_timestamp"] = datetime.now(timezone.utc).isoformat()
        payload.setdefault("confidence_score", float(payload.get("confidence_score", 0.5)))
        payload.setdefault("rubric_dimension", payload.get("rubric_dimension", "unspecified"))
        return payload

    def _resolve_hardware_label(self) -> str:
        profile = self.detector.hardware_profile
        if profile.get("cuda_available") and profile.get("cuda_device_count", 0) >= 4:
            return "gpu_high"
        if profile.get("cuda_available"):
            return "gpu_low"
        if profile.get("mps_available"):
            return "mps"
        if profile.get("npu_available"):
            return "npu"
        return "cpu"

    def validate_extended(self, metadata: Mapping[str, object]) -> bool:
        return all(field in metadata for field in self.EXTENDED_FIELDS)


__all__ = ["ExtendedMetadataHandler"]
