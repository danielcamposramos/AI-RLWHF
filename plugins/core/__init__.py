"""Core Transformer Lab plugins for AI-RLWHF."""

from .honesty_reward_calculator import HonestyRewardCalculator
from .hardware_detector import HardwareDetector
from .grpo_production_wrapper import ProductionGRPOWrapper

__all__ = ["HonestyRewardCalculator", "HardwareDetector", "ProductionGRPOWrapper"]
