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
    """Provides a sample dictionary of teacher feedback for testing.

    Returns:
        A dictionary mapping teacher names to their scores and feedback.
    """
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
    """Provides a sample honesty log entry for testing.

    Returns:
        A dictionary representing a single log entry.
    """
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
    """Creates a temporary directory with a sample configuration file.

    Yields:
        The path to the temporary directory.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "test_config.json"
        config = {"test_mode": True, "max_iterations": 10, "log_level": "DEBUG"}
        config_path.write_text(json.dumps(config), encoding="utf-8")
        yield temp_dir


@pytest.fixture
def offline_reference_path(tmp_path):
    """Copies the offline reference fixture to a temporary path.

    Args:
        tmp_path: The pytest temporary path fixture.

    Returns:
        The path to the temporary offline reference file.
    """
    fixture_src = PROJECT_ROOT / "tests" / "fixtures" / "offline_reference.jsonl"
    target = tmp_path / "offline_reference.jsonl"
    target.write_text(fixture_src.read_text(encoding="utf-8"), encoding="utf-8")
    return target


@pytest.fixture
def offline_reference_map(offline_reference_path):
    """Loads the offline reference map from the fixture.

    Args:
        offline_reference_path: A fixture providing the path to the offline reference file.

    Returns:
        A dictionary containing the offline reference data.
    """
    return load_offline_reference(offline_reference_path)


@pytest.fixture
def runner_config(offline_reference_path):
    """Provides a default runner configuration for testing.

    Args:
        offline_reference_path: A fixture providing the path to the offline reference file.

    Returns:
        A RunnerConfig object configured for testing.
    """
    config = load_runner_config()
    config.offline_dataset_path = offline_reference_path
    config.teacher_mode = "multiple"
    config.teacher_count = min(3, len(config.teacher_slots))
    return config
