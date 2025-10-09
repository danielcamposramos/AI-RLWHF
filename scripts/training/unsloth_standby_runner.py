"""Memory-aware training helper inspired by the Unsloth Standby guide."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional

try:  # Optional heavy deps; code falls back when unavailable.
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore

try:
    import GPUtil  # type: ignore
except Exception:  # pragma: no cover
    GPUtil = None  # type: ignore

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:  # pragma: no cover - optional
    from unsloth import FastLanguageModel  # type: ignore
except Exception:  # pragma: no cover
    FastLanguageModel = None  # type: ignore

try:  # pragma: no cover - optional
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:  # pragma: no cover - optional
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover
    class _DummyTrainer:  # pylint: disable=too-few-public-methods
        def job_wrapper(self):
            def decorator(func):
                return func

            return decorator

    tlab_trainer = _DummyTrainer()  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from plugins.core.grpo_rlwhf_wrapper import load_launch_bundle
from scripts.utils.config_loader import load_config

CONFIG_PATH = Path("configs/training/unsloth_standby.json")
GRPO_CONFIG_PATH = Path("configs/transformer-lab/grpo_config.yaml")
TELEMETRY_DIR = Path("experiments/telemetry")
MS_SWIFT_PLAN_PATH = Path("workspace/plans/ms_swift_grpo_launch.json")


@dataclass
class TelemetrySummary:
    """A summary of telemetry data collected during a training run.

    Attributes:
        total_iterations: The total number of training iterations performed.
        average_loss: The average loss recorded across all iterations.
        peak_gpu_percent: The peak GPU memory usage percentage observed.
        average_gpu_percent: The average GPU memory usage percentage.
    """

    total_iterations: int
    average_loss: float
    peak_gpu_percent: float
    average_gpu_percent: float

    def to_json(self) -> Dict[str, float]:
        """Convert the telemetry summary to a JSON-serializable dictionary."""
        return {
            "total_iterations": float(self.total_iterations),
            "average_loss": float(self.average_loss),
            "peak_gpu_percent": float(self.peak_gpu_percent),
            "average_gpu_percent": float(self.average_gpu_percent),
        }


class UnslothStandbyOptimizer:
    """Manages memory-efficient training inspired by Unsloth Standby."""

    def __init__(self, config_path: Path | str = CONFIG_PATH, grpo_config_path: Path | str = GRPO_CONFIG_PATH) -> None:
        """Initialize the optimizer and load configuration defaults."""
        self.config = self._load_config(Path(config_path))
        self.grpo_config_path = Path(grpo_config_path)
        self.memory_logs: List[Dict[str, float]] = []
        self.performance_metrics: Dict[int, Dict[str, float]] = {}

    @staticmethod
    def _load_config(path: Path) -> Dict[str, object]:
        """Load the training configuration with default fallbacks."""
        defaults: Dict[str, object] = {
            "gpu_memory_utilization": 0.95,
            "max_seq_length": 8192,
            "dtype": "bfloat16",
            "use_gradient_checkpointing": True,
            "optimizer": "adamw_8bit",
            "learning_rate": 2e-4,
            "warmup_steps": 10,
            "max_steps": 100,
            "logging_steps": 5,
            "save_steps": 25,
            "evaluation_strategy": "steps",
            "enable_internet_teachers": True,
            "enable_offline_validation": True,
            "fallback_mode": "use_offline",
            "offline_dataset_path": "data/examples/offline_reference.jsonl",
            "models": {
                "student": "unsloth/tinyllama",
                "teacher_ensemble": [
                    "grok-search-evaluator",
                    "codex",
                    "kimi",
                    "glm",
                    "deepseek",
                    "qwen",
                ],
            },
        }
        return load_config(path, defaults)

    def setup_environment(self) -> None:
        """Set environment variables and PyTorch flags for standby optimization."""
        os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
        if torch is not None:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

    def monitor_memory_usage(self) -> Dict[str, float]:
        """Record CPU and GPU snapshots for telemetry export."""
        snapshot: Dict[str, float] = {
            "timestamp": datetime.now(UTC).timestamp(),
            "cpu_memory_percent": 0.0,
            "cpu_available_gb": 0.0,
        }
        if psutil is not None:
            vm = psutil.virtual_memory()
            snapshot["cpu_memory_percent"] = float(vm.percent)
            snapshot["cpu_available_gb"] = float(vm.available) / 1024**3
        if GPUtil is not None:
            gpus = GPUtil.getGPUs()  # type: ignore[attr-defined]
            for idx, gpu in enumerate(gpus):
                snapshot[f"gpu_{idx}_memory_percent"] = float(gpu.memoryUtil * 100)
        self.memory_logs.append(snapshot)
        return snapshot

    def optimize_model_loading(self, model_name: str, model_type: str = "student"):
        """Load a causal language model while preserving memory efficiency."""
        if FastLanguageModel is not None:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=int(self.config.get("max_seq_length", 2048)),
                dtype=self.config.get("dtype", "bfloat16"),
                load_in_4bit=True,
                device_map="auto",
            )
            if self.config.get("use_gradient_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            return model, tokenizer
        if AutoModelForCausalLM is not None and AutoTokenizer is not None:
            model = AutoModelForCausalLM.from_pretrained(model_name)  # type: ignore[arg-type]
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            return model, tokenizer
        return DummyModel(), DummyTokenizer()

    def run_training_step(self, student_model, teacher_feedback: Mapping[str, Mapping[str, float]], iteration: int):
        """Simulate a single training step and log telemetry-friendly metrics."""
        self.monitor_memory_usage()
        if torch is None or not hasattr(student_model, "__call__"):
            loss = 0.0
        else:
            aggregated_reward = sum(float(v.get("score", 0)) for v in teacher_feedback.values())
            loss_tensor = torch.tensor(1.0 - aggregated_reward * 0.01)  # type: ignore[assignment]
            loss = float(loss_tensor.item())
        self.performance_metrics[iteration] = {"loss": loss}
        return loss

    def generate_telemetry_report(self, output_dir: Path = TELEMETRY_DIR) -> TelemetrySummary:
        """Save telemetry summary, CSV trace, and optional memory plot."""
        output_dir.mkdir(parents=True, exist_ok=True)
        peak_gpu = 0.0
        gpu_values: List[float] = []
        for entry in self.memory_logs:
            gpu_stats = [value for key, value in entry.items() if key.endswith("memory_percent")]
            if gpu_stats:
                peak_gpu = max(peak_gpu, max(gpu_stats))
                gpu_values.extend(gpu_stats)
        losses = [metrics.get("loss", 0.0) for metrics in self.performance_metrics.values()]
        average_loss = sum(losses) / len(losses) if losses else 0.0
        avg_gpu = sum(gpu_values) / len(gpu_values) if gpu_values else 0.0
        summary = TelemetrySummary(
            total_iterations=len(self.performance_metrics),
            average_loss=average_loss,
            peak_gpu_percent=peak_gpu,
            average_gpu_percent=avg_gpu,
        )
        (output_dir / "telemetry_summary.json").write_text(json.dumps(summary.to_json(), indent=2), encoding="utf-8")
        if pd is not None and plt is not None and self.memory_logs:
            frame = pd.DataFrame(self.memory_logs)
            frame.to_csv(output_dir / "memory_trace.csv", index=False)
            plt.figure(figsize=(10, 4))
            frame.get("cpu_memory_percent", pd.Series()).plot(label="CPU")
            if "gpu_0_memory_percent" in frame:
                frame["gpu_0_memory_percent"].plot(label="GPU0")
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_dir / "memory_plot.png")
            plt.close()
        toggle_payload = {
            "enable_internet_teachers": bool(self.config.get("enable_internet_teachers", True)),
            "enable_offline_validation": bool(self.config.get("enable_offline_validation", True)),
            "fallback_mode": self.config.get("fallback_mode", "use_offline"),
        }
        telemetry_json = summary.to_json()
        telemetry_json.update(toggle_payload)
        (output_dir / "telemetry_summary.json").write_text(json.dumps(telemetry_json, indent=2), encoding="utf-8")
        return summary

    def export_ms_swift_plan(self, hardware_profile: str | None = None) -> Dict[str, object]:
        """Generate and persist a launch bundle for the ms-swift GRPO runner."""
        profile = hardware_profile or str(self.config.get("hardware_profile", "single_gpu"))
        try:
            bundle = load_launch_bundle(hardware_profile=profile, config_path=self.grpo_config_path)
        except Exception as exc:  # pragma: no cover - dependency/runtime issues
            plan_payload = {"error": str(exc), "hardware_profile": profile}
        else:
            plan_payload = {
                "hardware_profile": profile,
                "env": bundle.env,
                "args": bundle.args,
                "command": bundle.command,
            }
        MS_SWIFT_PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
        MS_SWIFT_PLAN_PATH.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")
        return plan_payload


class DummyModel:  # pragma: no cover - test-friendly stub
    """A placeholder model used when Unsloth/transformers dependencies are absent."""

    def __call__(self, **_kwargs):
        class _Output:
            loss = 0.0

        return _Output()


class DummyTokenizer:  # pragma: no cover - placeholder stub
    """A placeholder tokenizer for environments without transformers."""
    pass


@tlab_trainer.job_wrapper()  # type: ignore[attr-defined]
def run_unsloth_optimized_training():
    """Entry point for running an Unsloth-optimized telemetry loop."""
    optimizer = UnslothStandbyOptimizer()
    optimizer.setup_environment()
    model, tokenizer = optimizer.optimize_model_loading(str(optimizer.config["models"]["student"]))  # type: ignore[index]
    for iteration in range(int(optimizer.config.get("max_steps", 1))):
        feedback = {"teacher": {"score": 1.0, "feedback": "synthetic"}}
        optimizer.run_training_step(model, feedback, iteration)
    optimizer.generate_telemetry_report()
    optimizer.export_ms_swift_plan()
    return True


if __name__ == "__main__":  # pragma: no cover
    runner = UnslothStandbyOptimizer()
    runner.setup_environment()
    model, _ = runner.optimize_model_loading("unsloth/tinyllama")
    runner.run_training_step(model, {"codec": {"score": 1.5}}, 0)
    print(runner.generate_telemetry_report().to_json())
