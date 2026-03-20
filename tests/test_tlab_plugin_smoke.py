"""Optional smoke test for the TransformerLab trainer plugin."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from plugins.core.grpo_rlwhf_wrapper.main import LocalTLabTrainerAdapter, run_direct_grpo_training


@pytest.mark.skipif(
    not os.environ.get("AI_RLWHF_RUN_TLAB_SMOKE"),
    reason="Set AI_RLWHF_RUN_TLAB_SMOKE=1 to enable the live GRPO smoke test.",
)
def test_one_step_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("trl")
    pytest.importorskip("peft")

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the live GRPO smoke test")

    dataset_path = tmp_path / "smoke.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "What does honesty mean?", "answer": "Honesty means telling the truth.", "reward": 1})
        + "\n",
        encoding="utf-8",
    )
    params = {
        "model_name": "HuggingFaceTB/SmolLM-135M",
        "output_dir": str(tmp_path / "out"),
        "adaptor_output_dir": str(tmp_path / "adaptor"),
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "learning_rate": 1e-5,
        "max_completion_length": 64,
        "max_prompt_length": 64,
        "num_generations": 2,
        "logging_steps": 1,
        "save_steps": 1,
        "max_steps": 1,
        "contrastive_enabled": False,
    }
    runtime = LocalTLabTrainerAdapter(params, dataset_path)
    result = run_direct_grpo_training(runtime)
    assert result["status"] == "trained"
    assert Path(result["output_dir"]).exists()
