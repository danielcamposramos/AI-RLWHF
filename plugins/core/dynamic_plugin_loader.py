"""Dynamic plugin selection utilities informed by hardware capabilities."""
from __future__ import annotations

from typing import Dict, Mapping

from plugins.core.hardware_detector import HardwareDetector


class DynamicPluginLoader:
    """Return recommended plugin variants for the detected hardware class."""

    VARIANTS = {
        "gpu_high": ["full_grpo_trainer", "real_time_evaluator"],
        "gpu_low": ["quantized_grpo", "batch_evaluator"],
        "mps": ["lightweight_dpo", "sampling_evaluator"],
        "cpu": ["heuristic_evaluator", "offline_analyzer"],
    }

    def __init__(self, detector: HardwareDetector | None = None) -> None:
        self.detector = detector or HardwareDetector()

    def determine_variant(self) -> str:
        profile = self.detector.hardware_profile
        if profile.get("cuda_available") and profile.get("cuda_device_count", 0) >= 4:
            return "gpu_high"
        if profile.get("cuda_available"):
            return "gpu_low"
        if profile.get("mps_available"):
            return "mps"
        return "cpu"

    def load_optimal_plugins(self) -> Dict[str, Mapping[str, object]]:
        variant = self.determine_variant()
        plugins = self.VARIANTS.get(variant, [])
        return {plugin: self._plugin_config(plugin, variant) for plugin in plugins}

    def _plugin_config(self, plugin_name: str, variant: str) -> Mapping[str, object]:
        return {
            "name": plugin_name,
            "variant": variant,
            "batch_size": self.detector.get_recommended_batch_size(),
        }


__all__ = ["DynamicPluginLoader"]
