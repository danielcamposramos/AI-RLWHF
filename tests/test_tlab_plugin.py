"""Tests for the TransformerLab GRPO + contrastive honesty trainer plugin."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from datasets import load_from_disk

from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from plugins.core.grpo_rlwhf_wrapper.main import (
    build_contrastive_callback,
    build_decomposition_reward_fn,
    build_grpo_config_kwargs,
    build_honesty_reward_fn,
    create_grpo_config,
)
from scripts.data_pipeline.tlab_dataset_bridge import convert_rlwhf_to_tlab


PLUGIN_DIR = Path("plugins/core/grpo_rlwhf_wrapper")


def test_index_json_valid():
    manifest = json.loads((PLUGIN_DIR / "index.json").read_text(encoding="utf-8"))
    required = {
        "name",
        "uniqueId",
        "description",
        "plugin-format",
        "type",
        "version",
        "files",
        "setup-script",
        "training_template_format",
        "parameters",
    }
    assert required <= set(manifest.keys())
    assert manifest["plugin-format"] == "python"
    assert manifest["type"] == "trainer"
    assert "main.py" in manifest["files"]
    assert "setup.sh" in manifest["files"]
    assert "LlamaForCausalLM" in manifest["model_architectures"]


def test_setup_sh_exists():
    setup_path = PLUGIN_DIR / "setup.sh"
    assert setup_path.exists()
    mode = setup_path.stat().st_mode
    assert mode & stat.S_IXUSR


def test_reward_function_returns_floats():
    reward_fn = build_honesty_reward_fn()
    rewards = reward_fn(["I think the answer is Paris.", "Everyone knows this with certainty."])
    assert rewards and len(rewards) == 2
    assert all(isinstance(value, float) for value in rewards)


def test_decomposition_reward_uses_honesty():
    reward_fn = build_decomposition_reward_fn()
    completions = ["I am not sure, but Paris is likely the answer."]
    high = reward_fn(
        completions,
        decomposition=[{"overall_honesty": 0.9, "overall_correctness": 0.6}],
        reward=[1.0],
    )[0]
    low = reward_fn(
        completions,
        decomposition=[{"overall_honesty": 0.1, "overall_correctness": 0.6}],
        reward=[1.0],
    )[0]
    assert high > low


def test_contrastive_callback_initializes():
    callback = build_contrastive_callback(
        {
            "loss_weights": {"correctness": 0.25, "honesty": 0.40, "contrast": 0.15},
            "temperature": {"initial": 0.1, "final": 0.03, "schedule": "exponential"},
        }
    )
    assert callback is not None


def test_dataset_bridge_preserves_decomposition(tmp_path):
    dataset_path = tmp_path / "sample.jsonl"
    dataset_path.write_text(
        json.dumps(
            {
                "prompt": "What is honesty?",
                "answer": "Honesty is truthful reporting.",
                "reward": 1,
                "decomposition": {"overall_honesty": 0.9, "overall_correctness": 0.8},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "hf_dataset"
    convert_rlwhf_to_tlab(dataset_path, output_dir)
    dataset = load_from_disk(str(output_dir))
    assert dataset[0]["prompt"] == "What is honesty?"
    assert dataset[0]["decomposition"]["overall_honesty"] == 0.9


def test_grpo_config_valid(monkeypatch):
    from plugins.core.grpo_rlwhf_wrapper import main as module

    class DummyGRPOConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(module, "GRPOConfig", DummyGRPOConfig)
    kwargs = build_grpo_config_kwargs(
        output_dir="tmp-output",
        params={
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": 5e-6,
            "max_completion_length": 128,
            "max_prompt_length": 128,
            "num_generations": 4,
        },
        report_to=["tensorboard"],
    )
    config = create_grpo_config(
        {
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
            "learning_rate": 5e-6,
            "max_completion_length": 128,
            "max_prompt_length": 128,
            "num_generations": 4,
        },
        output_dir="tmp-output",
        report_to=["tensorboard"],
    )
    assert kwargs["num_generations"] == 4
    assert kwargs["report_to"] == ["tensorboard"]
    assert config.kwargs["output_dir"] == "tmp-output"


def test_production_wrapper_direct_fallback_is_structured(tmp_path):
    config_path = tmp_path / "test_grpo_config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_name": "HuggingFaceTB/SmolLM-135M",
                "per_device_train_batch_size": 1,
                "num_train_epochs": 1,
            }
        ),
        encoding="utf-8",
    )
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        json.dumps({"prompt": "Say hello honestly.", "answer": "Hello.", "reward": 1}) + "\n",
        encoding="utf-8",
    )
    wrapper = ProductionGRPOWrapper(str(config_path))
    result = wrapper.launch(str(dataset_path), str(tmp_path / "out"))
    assert result["status"] in {"trained", "dependency_unavailable"}
