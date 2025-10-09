import json
import os
import time
from pathlib import Path

import pytest

from plugins.core.multi_teacher_aggregator import multi_teacher_aggregator
from scripts.training.multi_teacher_runner import run_batch_evaluation, run_multi_teacher_loop
from scripts.training.unsloth_standby_runner import UnslothStandbyOptimizer
from scripts.utils.offline_scoring import score_against_reference


@pytest.fixture
def temp_log(tmp_path):
    log_file = tmp_path / "aggregation.jsonl"
    yield log_file


def test_teacher_weighting_schemes(sample_teacher_feedback, temp_log):
    weights = {"grok": 0.33, "codex": 0.33, "kimi": 0.34}
    result = multi_teacher_aggregator(
        teacher_feedback=sample_teacher_feedback,
        teacher_weights=weights,
        aggregation_method="weighted_average",
        log_path=temp_log,
    )
    assert "aggregated_score" in result
    assert -2.0 <= result["aggregated_score"] <= 2.0
    assert temp_log.exists()


def test_confidence_weighting(sample_teacher_feedback, temp_log):
    result = multi_teacher_aggregator(
        teacher_feedback=sample_teacher_feedback,
        aggregation_method="confidence_weighted",
        log_path=temp_log,
    )
    assert isinstance(result["aggregated_score"], float)


def test_memory_efficiency_metrics():
    optimizer = UnslothStandbyOptimizer()
    snapshot = optimizer.monitor_memory_usage()
    assert "cpu_memory_percent" in snapshot
    assert snapshot["cpu_memory_percent"] >= 0
    assert "cpu_available_gb" in snapshot


def test_generate_telemetry(tmp_path):
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
    test_feedback = {f"teacher_{i}": {"score": 1, "feedback": "ok"} for i in range(10)}
    start = time.time()
    result = multi_teacher_aggregator(
        teacher_feedback=test_feedback,
        aggregation_method="weighted_average",
        log_path=temp_log,
    )
    elapsed = time.time() - start
    assert elapsed < 1.0
    assert "aggregated_score" in result


def test_log_schema(temp_log):
    payload = multi_teacher_aggregator(
        teacher_feedback={"grok": {"score": 2, "feedback": "Accurate"}},
        log_path=temp_log,
        prompt="What is RL?",
        student_answer="Reinforcement learning is...",
    )
    assert payload["individual_scores"]["grok"] == 2.0
    rows = [json.loads(line) for line in temp_log.read_text().splitlines() if line.strip()]
    assert rows[0]["prompt"] == "What is RL?"


def test_error_handling_graceful(temp_log):
    result = multi_teacher_aggregator(
        teacher_feedback="invalid",  # type: ignore[arg-type]
        log_path=temp_log,
    )
    assert result["aggregated_score"] == 0.0


def test_batch_runner(tmp_path, runner_config):
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
    log_dir = tmp_path / "logs"
    os.environ["AI_RLWHF_LOG_DIR"] = str(log_dir)
    from scripts.utils.chain_logger import log

    log("kimi", "task_start", {"task": "demo"})
    files = list(log_dir.glob("chain-kimi-*.jsonl"))
    assert files
    content = files[0].read_text(encoding="utf-8").strip()
    assert "task" in content


def test_offline_scoring_against_reference(offline_reference_map):
    prompt = "Explain RLHF."
    answer = "RLHF uses human feedback rewards to refine models after supervised fine tuning."
    score, feedback = score_against_reference(prompt, answer, offline_reference_map)
    assert score >= 0
    assert "reference" in feedback or "Offline" in feedback


def test_runner_respects_internet_toggle(runner_config, offline_reference_map):
    runner_config.enable_internet_teachers = False
    result = run_multi_teacher_loop(
        "Explain RLHF.",
        "RLHF mixes rewards plus preference tuning.",
        runner_config,
        offline_reference_map,
    )
    assert "grok-search-evaluator" in result["teacher_feedback"]
    assert result["aggregated_score"] <= 2
    default_log = Path("data/processed/honesty_logs/multi_teacher_aggregation.jsonl")
    if default_log.exists():
        default_log.unlink()
