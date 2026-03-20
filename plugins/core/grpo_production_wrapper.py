# AI-RLWHF – production wrapper for ms-swift or direct TRL GRPO training.
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict

VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor/ms-swift-sub"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from swift.llm.app.app import app_main  # type: ignore
except ImportError as exc:  # pragma: no cover - optional runtime
    app_main = None  # type: ignore
    _ms_swift_import_error = exc

from plugins.core.grpo_rlwhf_wrapper.main import LocalTLabTrainerAdapter, run_direct_grpo_training
from plugins.core.hardware_detector import HardwareDetector
from plugins.core.honesty_reward_calculator import HonestyRewardCalculator
from scripts.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("grpo_prod_wrapper")


class ProductionGRPOWrapper:
    """Launch GRPO training through ms-swift or the direct TRL bridge."""

    def __init__(self, config_path: str):
        self.cfg = load_config(config_path, default={})
        self.hw = HardwareDetector()
        self.reward_fn = HonestyRewardCalculator()

    def _cfg_value(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in self.cfg:
                return self.cfg[key]
        ms_swift_cfg = self.cfg.get("ms_swift", {})
        grpo_args = ms_swift_cfg.get("grpo_args", {})
        for key in keys:
            if key in grpo_args:
                return grpo_args[key]
        return default

    def _build_args(self, dataset_jsonl: str, output_dir: str) -> list[str]:
        args = [
            "--model_type",
            str(self._cfg_value("model_type", default="qwen2-0_5b-instruct")),
            "--sft_type",
            "lora",
            "--rlhf_type",
            "grpo",
            "--dataset",
            dataset_jsonl,
            "--output_dir",
            output_dir,
            "--max_length",
            str(self._cfg_value("max_length", "max_prompt_length", default=2048)),
            "--per_device_train_batch_size",
            str(self._cfg_value("per_device_train_batch_size", "per_device_batch_size", default=1)),
            "--gradient_accumulation_steps",
            str(self._cfg_value("gradient_accumulation_steps", default=16)),
            "--learning_rate",
            str(self._cfg_value("learning_rate", default=1e-5)),
            "--num_train_epochs",
            str(self._cfg_value("num_train_epochs", default=1)),
            "--save_steps",
            str(self._cfg_value("save_steps", default=500)),
            "--eval_steps",
            str(self._cfg_value("eval_steps", default=500)),
            "--logging_steps",
            str(self._cfg_value("logging_steps", default=10)),
            "--beta",
            str(self._cfg_value("beta", default=0.04)),
            "--num_generations",
            str(self._cfg_value("num_generations", default=2)),
        ]
        if bool(self._cfg_value("fp16", default=True)):
            args.append("--fp16")
        if bool(self._cfg_value("use_flash_attn", default=False)):
            args.append("--use_flash_attn")
        return args

    def combine_losses(
        self,
        grpo_loss: float,
        contrastive_loss: float | None = None,
        contrastive_weight: float | None = None,
    ) -> Dict[str, float]:
        weight = float(
            contrastive_weight
            if contrastive_weight is not None
            else self.cfg.get("contrastive_loss_weight", 0.20)
        )
        grpo_component = float(grpo_loss)
        contrastive_component = float(contrastive_loss) if contrastive_loss is not None else 0.0
        total = grpo_component + (weight * contrastive_component)
        return {
            "grpo_loss": grpo_component,
            "contrastive_loss": contrastive_component,
            "contrastive_weight": weight,
            "total_loss": total,
        }

    def _launch_ms_swift(self, dataset_jsonl: str, output_dir: str) -> Dict[str, Any]:
        args = self._build_args(dataset_jsonl, output_dir)
        if app_main is None:  # pragma: no cover - dependency-specific path
            raise RuntimeError("ms-swift is unavailable") from _ms_swift_import_error
        old_argv = list(sys.argv)
        try:
            sys.argv = ["swift"] + args
            app_main()
        finally:
            sys.argv = old_argv
        return {"status": "trained", "backend": "ms-swift", "args": args, "output_dir": output_dir}

    def _launch_trl_direct(
        self,
        dataset_jsonl: str,
        output_dir: str,
        contrastive_payload: Dict[str, Any] | None = None,
        telemetry=None,
    ) -> Dict[str, Any]:
        params = {
            "model_name": self._cfg_value(
                "model_name",
                default="HuggingFaceTB/SmolLM-135M",
            ),
            "output_dir": output_dir,
            "adaptor_output_dir": output_dir,
            "num_train_epochs": int(self._cfg_value("num_train_epochs", default=1)),
            "per_device_train_batch_size": int(
                self._cfg_value("per_device_train_batch_size", "per_device_batch_size", default=1)
            ),
            "learning_rate": float(self._cfg_value("learning_rate", default=1e-5)),
            "max_completion_length": int(self._cfg_value("max_completion_length", default=512)),
            "max_prompt_length": int(self._cfg_value("max_prompt_length", "max_length", default=512)),
            "num_generations": int(self._cfg_value("num_generations", default=2)),
            "lora_rank": int(self._cfg_value("lora_rank", default=16)),
            "lora_alpha": int(self._cfg_value("lora_alpha", default=32)),
            "contrastive_enabled": bool(self._cfg_value("contrastive_enabled", default=True)),
            "contrastive_correctness_weight": float(
                self._cfg_value("contrastive_correctness_weight", default=0.25)
            ),
            "contrastive_honesty_weight": float(
                self._cfg_value("contrastive_honesty_weight", default=0.40)
            ),
            "contrastive_contrast_weight": float(
                self._cfg_value("contrastive_contrast_weight", default=0.15)
            ),
            "contrastive_grpo_weight": float(
                self._cfg_value("contrastive_grpo_weight", default=0.20)
            ),
            "temperature_initial": float(self._cfg_value("temperature_initial", default=0.1)),
            "temperature_final": float(self._cfg_value("temperature_final", default=0.03)),
            "logging_steps": int(self._cfg_value("logging_steps", default=10)),
            "save_steps": int(self._cfg_value("save_steps", default=50)),
            "gradient_checkpointing": bool(self._cfg_value("gradient_checkpointing", default=True)),
            "max_steps": int(self._cfg_value("max_steps", default=-1)),
        }
        runtime = LocalTLabTrainerAdapter(params, dataset_jsonl)
        return run_direct_grpo_training(
            runtime,
            contrastive_payload=contrastive_payload,
            telemetry=telemetry,
            allow_missing_dependencies=True,
        )

    def launch(
        self,
        dataset_jsonl: str,
        output_dir: str,
        *,
        contrastive_payload: Dict[str, Any] | None = None,
        telemetry=None,
    ):
        log.info("Preparing GRPO launch for dataset=%s output_dir=%s", dataset_jsonl, output_dir)

        combined_losses = None
        if contrastive_payload:
            combined_losses = self.combine_losses(
                grpo_loss=float(self.cfg.get("grpo_loss_baseline", 0.0)),
                contrastive_loss=contrastive_payload.get("loss_preview"),
                contrastive_weight=contrastive_payload.get("loss_weight"),
            )
            log.info("Contrastive payload attached: %s", contrastive_payload)
            log.info("Combined loss preview: %s", combined_losses)

        try:
            if app_main is not None:
                result = self._launch_ms_swift(dataset_jsonl, output_dir)
            else:
                result = self._launch_trl_direct(
                    dataset_jsonl,
                    output_dir,
                    contrastive_payload=contrastive_payload,
                    telemetry=telemetry,
                )
            if combined_losses:
                result["combined_losses"] = combined_losses
            return result
        except Exception:
            log.exception("GRPO launch failed")
            raise


if __name__ == "__main__":  # pragma: no cover - CLI convenience
    import fire

    fire.Fire(ProductionGRPOWrapper)
