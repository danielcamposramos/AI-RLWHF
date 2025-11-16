"""Core Transformer Lab plugins for AI-RLWHF."""
from __future__ import annotations

# Lazy imports to avoid heavy dependencies during plugin discovery
__all__ = ["HonestyRewardCalculator", "HardwareDetector", "ProductionGRPOWrapper"]


def __getattr__(name: str):
    """Lazy import heavy dependencies only when accessed."""
    if name == "HardwareDetector":
        from .hardware_detector import HardwareDetector
        return HardwareDetector
    if name == "HonestyRewardCalculator":
        from .honesty_reward_calculator import HonestyRewardCalculator
        return HonestyRewardCalculator
    if name == "ProductionGRPOWrapper":
        from .grpo_production_wrapper import ProductionGRPOWrapper
        return ProductionGRPOWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
