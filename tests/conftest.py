import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.multi_teacher_runner import RunnerConfig, load_runner_config
from scripts.utils.offline_scoring import load_offline_reference


@pytest.fixture
def sample_teacher_feedback():
    return {
        "grok": {"score": 2, "feedback": "Excellent, factually correct answer"},
        "codex": {"score": 1, "feedback": "Mostly correct but missing some context"},
        "kimi": {"score": 0, "feedback": "Answer is neutral, no significant issues"},
        "glm": {"score": -1, "feedback": "Some inaccuracies detected"},
        "deepseek": {"score": 1, "feedback": "Generally accurate with minor omissions"},
        "qwen": {"score": 2, "feedback": "Comprehensive and correct answer"},
    }


@pytest.fixture
def sample_honesty_log_entry():
    return {
        "prompt": "Explain the theory of relativity in simple terms",
        "student_answer": "Einstein's theory says that time and space are connected, and gravity bends spacetime.",
        "teacher_feedback": {
            "grok": {"score": 1, "feedback": "Simplified but accurate overview"},
            "codex": {"score": 1, "feedback": "Good simplification for beginners"},
        },
        "reward": 1.0,
        "timestamp": "2024-01-15T14:30:00Z",
        "metadata": {
            "model_version": "student-v1.2",
            "context_length": 256,
            "training_iteration": 150,
        },
    }


@pytest.fixture
def temp_config_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        config = {"test_mode": True, "max_iterations": 10, "log_level": "DEBUG"}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        yield temp_dir


@pytest.fixture
def offline_reference_path(tmp_path):
    fixture_src = PROJECT_ROOT / "tests" / "fixtures" / "offline_reference.jsonl"
    target = tmp_path / "offline_reference.jsonl"
    target.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")
    return target


@pytest.fixture
def offline_reference_map(offline_reference_path):
    return load_offline_reference(offline_reference_path)


@pytest.fixture
def runner_config(offline_reference_path):
    config = load_runner_config()
    config.offline_dataset_path = offline_reference_path
    config.teacher_mode = "multiple"
    config.teacher_count = min(3, len(config.teacher_slots))
    return config
