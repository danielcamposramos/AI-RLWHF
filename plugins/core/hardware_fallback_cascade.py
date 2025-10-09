"""Apply hardware-aware fallback presets for ms-swift launches."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from plugins.core.hardware_detector import HardwareDetector

DEFAULT_FALLBACK_PATH = Path("configs/training/hardware_fallback.json")


class HardwareFallbackCascade:
    """Resolve fallback arguments from shared configuration presets."""

    def __init__(self, fallback_config: Path | str = DEFAULT_FALLBACK_PATH) -> None:
        self.fallback_config = Path(fallback_config)
        self.detector = HardwareDetector()
        self.fallbacks = self._load_fallbacks()

    def _load_fallbacks(self) -> Dict[str, Dict[str, Any]]:
        if not self.fallback_config.exists():
            return {}
        return json.loads(self.fallback_config.read_text(encoding="utf-8"))

    def get_cascaded_config(self, default_key: str | None = None) -> Dict[str, Any]:
        profile = self.detector.hardware_profile
        if profile.get("cuda_available") and profile.get("cuda_device_count", 0) >= 4:
            return dict(self.fallbacks.get("ascend_npu", {}))
        if profile.get("cuda_available"):
            return dict(self.fallbacks.get("gpu", self.fallbacks.get("cpu", {})))
        if profile.get("mps_available"):
            return dict(self.fallbacks.get("mps", {}))
        if profile.get("npu_available"):
            return dict(self.fallbacks.get("ascend_npu", {}))
        key = default_key or "cpu"
        return dict(self.fallbacks.get(key, {}))

    def apply_to_wrapper(self, wrapper) -> None:
        config = self.get_cascaded_config()
        wrapper.cfg.setdefault("grpo_args", {}).update(config)


__all__ = ["HardwareFallbackCascade"]
