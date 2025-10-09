"""Core multi-teacher aggregation plugin."""
from .main import aggregate_feedback, multi_teacher_aggregator

__all__ = ["multi_teacher_aggregator", "aggregate_feedback"]