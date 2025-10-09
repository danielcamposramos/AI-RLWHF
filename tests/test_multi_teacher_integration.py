import json
import os
import time
from pathlib import Path

import pytest

from plugins.core.multi_teacher_aggregator import multi_teacher_aggregator
from scripts.training.multi_teacher_runner import load_runner_config, run_batch_evaluation, run_multi_teacher_loop
from scripts.training.unsloth_standby_runner import UnslothStandbyOptimizer
from scripts.utils.offline_scoring import score_against_reference


@pytest.fixture
def temp_log(tmp_path):
    log_file = tmp_path / "aggregation.jsonl"
    yield log_file


def test_teacher_weighting_schemes(sample_teacher_feedback, temp_log):
    """Tests the weighted average aggregation scheme.

    Args:
        sample_teacher_feedback: A fixture providing sample teacher feedback.
        temp_log: A fixture for a temporary log file path.
    """
    weights = {"grok": 0.33, "codex": 0.33, "kimi": 0.34}
    result = multi_teacher_aggregator(
        teacher_feedback=sample_teacher_feedback,
        teacher_weights=weights,
        aggregation_method="weighted_average",
        log_path=temp_log,
        teacher_mode="multiple",
        teacher_slots=[{"name": name} for name in sample_teacher_feedback.keys()],
        teacher_count=3,
    )
    assert "aggregated_score" in result
    assert -2.0 <= result["aggregated_score"] <= 2.0
    assert temp_log.exists()
    assert "teacher_slots" in result


def test_confidence_weighting(sample_teacher_feedback, temp_log):
    """Tests the confidence weighted aggregation scheme.

    Args:
        sample_teacher_feedback: A fixture providing sample teacher feedback.
        temp_log: A fixture for a temporary log file path.
    """
    result = multi_teacher_aggregator(
        teacher_feedback=sample_teacher_feedback,
        aggregation_method="confidence_weighted",
        log_path=temp_log,
        teacher_slots=[{"name": "grok"}, {"name": "codex"}, {"name": "kimi"}],
    )
    assert isinstance(result["aggregated_score"], float)
    rows = [json.loads(line) for line in temp_log.read_text().splitlines() if line.strip()]
    assert rows[0]["config"]["teacher_mode"] == "multiple"
    assert rows[0]["config"]["teacher_slots"]


def test_memory_efficiency_metrics():
    """Tests the memory monitoring functionality of the UnslothStandbyOptimizer."""
    optimizer = UnslothStandbyOptimizer()
    snapshot = optimizer.monitor_memory_usage()
    assert "cpu_memory_percent" in snapshot
    assert snapshot["cpu_memory_percent"] >= 0
    assert "cpu_available_gb" in snapshot


def test_generate_telemetry(tmp_path):
    """Tests the generation of telemetry reports.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    optimizer = UnslothStandbyOptimizer()
    optimizer.performance_metrics = {i: {"loss": 1.0 / (i + 1)} for i in range(5)}
    optimizer.memory_logs = [
        {"cpu_memory_percent": 10 + i, "gpu_0_memory_percent": 20 + i}
        for i in range(5)
    ]
    summary = optimizer.generate_telemetry_report(tmp_path)
    assert summary.total_iterations == 5
    assert (tmp_path / "telemetry_summary.json").exists()


def test_performance_benchmarks(temp_log):
    """Runs a simple performance benchmark on the aggregator.

    Args:
        temp_log: A fixture for a temporary log file path.
    """
    test_feedback = {f"teacher_{i}": {"score": 1, "feedback": "ok"} for i in range(10)}
    start = time.time()
    result = multi_teacher_aggregator(
        teacher_feedback=test_feedback,
        aggregation_method="weighted_average",
        log_path=temp_log,
        teacher_slots=[{"name": f"teacher_{i}"} for i in range(10)],
    )
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert "aggregated_score" in result
    assert "teacher_slots" in result


def test_log_schema(temp_log):
    """Tests that the output log schema is correct.

    Args:
        temp_log: A fixture for a temporary log file path.
    """
    payload = multi_teacher_aggregator(
        teacher_feedback={"grok": {"score": 2, "feedback": "Accurate"}},
        log_path=temp_log,
        prompt="What is RL?",
        student_answer="Reinforcement learning is...",
    )
    assert payload["individual_scores"]["grok"] == 2.0
    rows = [json.loads(line) for line in temp_log.read_text().splitlines() if line.strip()]
    assert rows[0]["prompt"] == "What is RL?"
    assert rows[0]["config"]["teacher_slots"]


def test_error_handling_graceful(temp_log):
    """Tests graceful error handling with invalid input.

    Args:
        temp_log: A fixture for a temporary log file path.
    """
    result = multi_teacher_aggregator(
        teacher_feedback="invalid",  # type: ignore[arg-type]
        log_path=temp_log,
    )
    assert result["aggregated_score"] == 0.0
    assert "teacher_slots" in result


def test_single_teacher_mode(sample_teacher_feedback, temp_log):
    """Tests the aggregator in single teacher mode.

    Args:
        sample_teacher_feedback: A fixture providing sample teacher feedback.
        temp_log: A fixture for a temporary log file path.
    """
    single = {"grok": sample_teacher_feedback["grok"]}
    result = multi_teacher_aggregator(
        teacher_feedback=single,
        teacher_mode="single",
        teacher_count=1,
        teacher_slots=[{"name": "grok", "connection_type": "api"}],
        log_path=temp_log,
    )
    assert list(result["individual_scores"].keys()) == ["grok"]
    rows = [json.loads(line) for line in temp_log.read_text().splitlines() if line.strip()]
    assert rows[0]["config"]["teacher_slots"][0]["connection_type"] in {"api", "transformerlab_local", "ollama"}


def test_batch_runner(tmp_path, runner_config):
    """Tests the batch evaluation runner.

    Args:
        tmp_path: The pytest temporary path fixture.
        runner_config: A fixture providing a runner configuration.
    """
    prompts = tmp_path / "prompts.txt"
    answers = tmp_path / "answers.txt"
    prompts.write_text("What is AI?\nExplain RLHF\n", encoding="utf-8")
    answers.write_text("Artificial intelligence is...\nRLHF mixes RL and feedback\n", encoding="utf-8")
    output_file = tmp_path / "results.json"
    summary = run_batch_evaluation(prompts, answers, output_file, runner_config)
    assert summary["count"] == 2
    assert output_file.exists()
    default_log = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
    if default_log.exists():
        default_log.unlink()


def test_chain_logger(tmp_path):
    """Tests the chain logger utility.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    log_dir = tmp_path / "logs"
    os.environ["AI_RLWHF_LOG_DIR"] = str(log_dir)
    from scripts.utils.chain_logger import log

    log("kimi", "task_start", {"task": "demo"})
    files = list(log_dir.glob("chain-kimi-*.jsonl"))
    assert files
    content = files[0].read_text(encoding="utf-8").strip()
    assert "task" in content


def test_offline_scoring_against_reference(offline_reference_map):
    """Tests the offline scoring utility against a reference map.

    Args:
        offline_reference_map: A fixture providing a sample offline reference map.
    """
    prompt = "Explain RLHF."
    answer = "RLHF uses human feedback rewards to refine models after supervised fine tuning."
    score, feedback = score_against_reference(prompt, answer, offline_reference_map)
    assert score >= 0
    assert "reference" in feedback or "Offline" in feedback


def test_runner_respects_internet_toggle(runner_config, offline_reference_map):
    """Tests that the runner correctly handles the internet toggle.

    Args:
        runner_config: A fixture providing a runner configuration.
        offline_reference_map: A fixture providing a sample offline reference map.
    """
    runner_config.enable_internet_teachers = False
    runner_config.teacher_mode = "multiple"
    runner_config.teacher_count = 2
    result = run_multi_teacher_loop(
        "Explain RLHF.",
        "RLHF mixes rewards plus preference tuning.",
        runner_config,
        offline_reference_map,
    )
    assert "grok-search-evaluator" in result["teacher_feedback"]
    assert result["aggregated_score"] <= 2
    assert "teacher_slots" in result
    default_log = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
    if default_log.exists():
        default_log.unlink()


def test_single_teacher_runner_path(tmp_path, offline_reference_map):
    """Tests the runner in single teacher mode.

    Args:
        tmp_path: The pytest temporary path fixture.
        offline_reference_map: A fixture providing a sample offline reference map.
    """
    config = load_runner_config()
    config.teacher_mode = "single"
    config.teacher_count = 1
    prompt = "What is reinforcement learning?"
    student = "Reinforcement learning optimizes a policy based on rewards."
    result = run_multi_teacher_loop(prompt, student, config, offline_reference_map)
    assert len(result["teacher_feedback"]) == 1
