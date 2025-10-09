"""Production oriented GRPO launcher that adapts to local hardware."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from plugins.core.honesty_reward_calculator import HonestyRewardCalculator, reward_from_tuple
from plugins.core.hardware_detector import HardwareDetector
from plugins.core.hardware_fallback_cascade import HardwareFallbackCascade
from scripts.telemetry.training_metrics import TrainingMetricsLogger
from scripts.utils.config_loader import load_config

try:  # Prefer native ms-swift entrypoint when present
    from swift.llm.train import run_grpo  # type: ignore
except Exception:  # pragma: no cover - fallback when ms-swift missing
    run_grpo = None  # type: ignore

DEFAULT_CONFIG_PATH = Path("configs/transformer-lab/grpo_config.yaml")


class ProductionGRPOWrapper:
    """Enhance GRPO launches with adaptive batch sizing and telemetry."""

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.cfg = load_config(self.config_path, default={})
        self.detector = HardwareDetector()
        self.reward_calculator = HonestyRewardCalculator()
        self.fallback_cascade = HardwareFallbackCascade()

    def launch(self, dataset_path: str, output_dir: str, simulate_low_mem: bool | None = None) -> Dict[str, Any]:
        dataset = Path(dataset_path)
        if not dataset.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset}")
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)

        hardware_profile = self.detector.hardware_profile
        cascade_profile = self.fallback_cascade.get_cascaded_config()
        self.cfg.setdefault("grpo_args", {}).setdefault("dataset_path", dataset_path)
        self.cfg["grpo_args"]["output_dir"] = str(output)
        self.cfg["grpo_args"]["reward_module"] = "plugins.core.custom_honesty_rm"
        self.cfg["grpo_args"]["per_device_batch_size"] = self._calc_safe_batch(
            requested=int(self.cfg["grpo_args"].get("per_device_batch_size", 4)),
            simulate_low_mem=bool(simulate_low_mem or os.environ.get("SIMULATE_LOW_MEM")),
        )
        self.cfg["grpo_args"].update({f"hardware_{key}": value for key, value in cascade_profile.items()})

        telemetry = TrainingMetricsLogger(output / "telemetry")
        summary = {
            "hardware_profile": hardware_profile,
            "grpo_args": self.cfg["grpo_args"],
        }

        if run_grpo is not None:
            tmp_config = self._write_tmp_config()
            run_grpo(dataset=str(dataset), output_dir=str(output), config_json=tmp_config, reward_func=self._reward_hook)  # type: ignore[misc]
        else:
            command = self._build_command()
            self._persist_offline_launch(command, output)

        telemetry.finalize(total_batches=0, final_rewards={"status": "submitted"})
        summary_path = output / "production_wrapper_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _calc_safe_batch(self, requested: int, simulate_low_mem: bool) -> int:
        if simulate_low_mem or not self.detector.hardware_profile.get("cuda_available"):
            return max(1, min(requested, 2))
        total_memory = self.detector.hardware_profile.get("total_memory_gb", 0.0)
        if total_memory >= 40:
            return max(requested, 8)
        if total_memory >= 24:
            return min(requested, 6)
        if total_memory >= 16:
            return min(requested, 4)
        return min(requested, 2)

    def _write_tmp_config(self) -> str:
        tmp_file = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
        json.dump(self.cfg.get("grpo_args", {}), tmp_file, indent=2)
        tmp_file.close()
        return tmp_file.name

    def _build_command(self) -> str:
        from plugins.core.grpo_rlwhf_wrapper import GRPORLWHFWrapper  # inline import to avoid cycles

        wrapper = GRPORLWHFWrapper(self.config_path)
        launch = wrapper.build_launch_bundle(hardware_profile=self.cfg["grpo_args"].get("hardware_profile", "single_gpu"))
        env_exports = " ".join(f'{key}="{value}"' for key, value in launch.env.items())
        return f"{env_exports} {launch.command}"

    def _reward_hook(self, tuple_payload: Mapping[str, Any]) -> float:
        breakdown = reward_from_tuple(tuple_payload, self.reward_calculator)
        return breakdown["normalized_reward"]

    def _persist_offline_launch(self, command: str, output_dir: Path) -> None:
        script_path = output_dir / "offline_launch.sh"
        script_path.write_text(command + "\n", encoding="utf-8")
        subprocess.run(command, shell=True, check=False)


__all__ = ["ProductionGRPOWrapper"]
