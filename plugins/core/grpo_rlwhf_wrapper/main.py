"""TransformerLab GRPO + contrastive honesty trainer plugin."""
from __future__ import annotations

import inspect
import json
import logging
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # TransformerLab runtime
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover - local/offline fallback
    class _DummyTrainer:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.params = SimpleNamespace()
            self.report_to: list[str] = []

        def job_wrapper(self):
            def decorator(func):
                return func

            return decorator

        def progress_update(self, *_args, **_kwargs) -> None:
            return None

        def log_metric(self, *_args, **_kwargs) -> None:
            return None

        def create_progress_callback(self, **_kwargs):
            return None

        def load_dataset(self, dataset_types: Sequence[str] | None = None):
            from scripts.data_pipeline.tlab_dataset_bridge import load_dataset_source

            source = getattr(self.params, "dataset_path", None)
            if not source:
                return {"train": []}
            dataset = load_dataset_source(source)
            keys = list(dataset_types or ["train"])
            return {keys[0]: dataset}

    tlab_trainer = _DummyTrainer()  # type: ignore

try:  # Optional training dependencies
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from transformers import (  # type: ignore
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
except Exception:  # pragma: no cover
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore
    TrainerCallback = None  # type: ignore

try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
except Exception:  # pragma: no cover
    LoraConfig = None  # type: ignore
    TaskType = None  # type: ignore
    get_peft_model = None  # type: ignore

try:
    from trl import GRPOConfig, GRPOTrainer  # type: ignore
except Exception:  # pragma: no cover
    GRPOConfig = None  # type: ignore
    GRPOTrainer = None  # type: ignore

from scripts.data_pipeline.tlab_dataset_bridge import ensure_prompt_dataset, load_dataset_source
from scripts.utils.config_loader import load_config

DEFAULT_CONFIG_PATH = Path("configs/transformer-lab/grpo_config.yaml")

LOG = logging.getLogger("grpo_rlwhf_wrapper")


@dataclass
class LaunchBundle:
    """Launch metadata for legacy ms-swift and standby workflows."""

    env: Dict[str, str]
    args: Dict[str, Any]
    command: str


class GRPORLWHFWrapper:
    """Legacy launch-bundle builder retained for existing scripts."""

    def __init__(self, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path, default={})

    def build_launch_bundle(self, hardware_profile: str = "single_gpu") -> LaunchBundle:
        ms_swift_cfg = self.config.get("ms_swift", {})
        hardware_profiles = self.config.get("hardware_profiles", {})
        profile = hardware_profiles.get(hardware_profile, hardware_profiles.get("single_gpu", {}))
        env = dict(ms_swift_cfg.get("launch_env", {}))
        env.setdefault("UNSLOTH_VLLM_STANDBY", "1")
        args: Dict[str, Any] = dict(ms_swift_cfg.get("grpo_args", {}))
        args.setdefault("reward_module", ms_swift_cfg.get("reward_module", "plugins.core.custom_honesty_rm"))
        args.setdefault("hardware_profile", hardware_profile)
        args.update({f"hardware_{key}": value for key, value in profile.items()})
        command = self._render_command(ms_swift_cfg, args)
        return LaunchBundle(env=env, args=args, command=command)

    def _render_command(self, ms_swift_cfg: Mapping[str, Any], args: Mapping[str, Any]) -> str:
        entry = ms_swift_cfg.get("trainer_entry", "swift.llm.train.run_grpo")
        serialized_args = " ".join(
            f"--{key.replace('_', '-')}" f" {shlex.quote(_stringify(value))}"
            for key, value in args.items()
        )
        return f"python3 -m {entry} {serialized_args}"


class LocalTLabTrainerAdapter:  # pragma: no cover - exercised via wrapper tests
    """Small runtime adapter that mimics TransformerLab outside the host app."""

    def __init__(self, params: Mapping[str, Any], dataset_source: str | Path) -> None:
        self.params = SimpleNamespace(**dict(params))
        self.report_to: list[str] = []
        self._dataset_source = str(dataset_source)
        self.progress_events: list[int] = []
        self.metrics: list[tuple[str, float, int]] = []

    def progress_update(self, percent: int) -> None:
        self.progress_events.append(int(percent))

    def log_metric(self, name: str, value: float, step: int) -> None:
        self.metrics.append((name, float(value), int(step)))

    def create_progress_callback(self, **_kwargs):
        return None

    def load_dataset(self, dataset_types: Sequence[str] | None = None):
        dataset = load_dataset_source(self._dataset_source)
        keys = list(dataset_types or ["train"])
        return {keys[0]: dataset}


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _get_param(params: Any, name: str, default: Any = None) -> Any:
    if isinstance(params, Mapping):
        return params.get(name, default)
    return getattr(params, name, default)


def _value_for_index(value: Any, index: int) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return None
        if index < len(value):
            return value[index]
        return value[-1]
    return value


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        if "content" in completion:
            return str(completion.get("content") or "")
        if "text" in completion:
            return str(completion.get("text") or "")
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes, bytearray)):
        parts = [_extract_completion_text(item) for item in completion]
        return " ".join(part for part in parts if part)
    return str(completion or "")


def _dependency_state() -> dict[str, bool]:
    return {
        "torch": torch is not None,
        "transformers": AutoModelForCausalLM is not None and AutoTokenizer is not None,
        "peft": get_peft_model is not None and LoraConfig is not None,
        "trl": GRPOTrainer is not None and GRPOConfig is not None,
    }


def _missing_training_dependencies() -> list[str]:
    return [name for name, available in _dependency_state().items() if not available]


def build_honesty_reward_fn():
    """Baseline GRPO reward based on honesty language and format quality."""

    def honesty_reward(completions, **_kwargs):
        rewards: list[float] = []
        honesty_phrases = (
            "i'm not sure",
            "i do not know",
            "i don't know",
            "i'm uncertain",
            "i believe",
            "i think",
            "it is possible",
            "this might be",
            "approximately",
            "roughly",
            "to the best of my knowledge",
            "i may be wrong",
        )
        overconfidence_phrases = (
            "i am 100% certain",
            "there is no doubt",
            "it is absolutely",
            "without question",
            "i guarantee",
            "everyone knows",
        )
        for completion in completions:
            text = _extract_completion_text(completion).strip()
            lowered = text.lower()
            reward = 0.0
            honesty_count = sum(1 for phrase in honesty_phrases if phrase in lowered)
            reward += min(honesty_count * 0.3, 0.9)
            reward -= sum(0.5 for phrase in overconfidence_phrases if phrase in lowered)
            if len(text) < 10 and honesty_count == 0:
                reward -= 1.0
            rewards.append(float(reward))
        return rewards

    return honesty_reward


def build_decomposition_reward_fn():
    """Prefer decomposition-aware scores while falling back to heuristics."""
    heuristic_reward = build_honesty_reward_fn()

    def decomposition_reward(completions, **kwargs):
        rewards = heuristic_reward(completions, **kwargs)
        decomposition_values = kwargs.get("decomposition") or kwargs.get("decompositions")
        reward_values = kwargs.get("reward") or kwargs.get("rewards")

        for idx, _completion in enumerate(completions):
            decomposition = _value_for_index(decomposition_values, idx)
            if isinstance(decomposition, str):
                try:
                    decomposition = json.loads(decomposition)
                except json.JSONDecodeError:
                    decomposition = None
            if not isinstance(decomposition, Mapping):
                continue
            honesty = _to_float(decomposition.get("overall_honesty"))
            correctness = _to_float(decomposition.get("overall_correctness"))
            dataset_reward = _to_float(_value_for_index(reward_values, idx))
            rewards[idx] = float((0.4 * honesty) + (0.25 * correctness) + (0.35 * dataset_reward))
        return rewards

    return decomposition_reward


def build_contrastive_callback(contrastive_config: Mapping[str, Any]):
    """Create a lightweight callback that logs contrastive metrics from trainer logs.

    TODO: subclass GRPOTrainer to inject the auxiliary contrastive loss directly
    into the training step. The callback is the first integration milestone.
    """
    if TrainerCallback is None or torch is None:
        return None

    try:
        from plugins.core.contrastive_loss import ContrastiveHonestyLoss
        from plugins.core.embedding_projector import EmbeddingProjector
    except Exception:  # pragma: no cover - optional modules missing
        return None

    class ContrastiveHonestyCallback(TrainerCallback):
        def __init__(self, config: Mapping[str, Any]) -> None:
            self.config = dict(config)
            self.loss_module = ContrastiveHonestyLoss(config)
            self.projector_cls = EmbeddingProjector
            self.total_steps = 0

        def on_train_begin(self, args, state, control, **_kwargs):
            self.total_steps = int(state.max_steps or getattr(args, "max_steps", 0) or 0)
            return control

        def on_log(self, args, state, control, logs=None, **_kwargs):
            if logs:
                for key in ("loss_correctness", "loss_honesty", "loss_contrast", "loss_total"):
                    if key in logs:
                        tlab_trainer.log_metric(key, float(logs[key]), int(state.global_step))
            return control

    return ContrastiveHonestyCallback(contrastive_config)


def build_grpo_config_kwargs(
    *,
    output_dir: str | Path,
    params: Any,
    report_to: Sequence[str] | None = None,
) -> dict[str, Any]:
    report_targets = list(report_to or [])
    use_bf16 = bool(
        torch is not None
        and torch.cuda.is_available()
        and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    use_fp16 = bool(torch is not None and torch.cuda.is_available() and not use_bf16)
    return {
        "output_dir": str(output_dir),
        "num_train_epochs": int(_get_param(params, "num_train_epochs", 1)),
        "per_device_train_batch_size": int(_get_param(params, "per_device_train_batch_size", 4)),
        "learning_rate": float(_get_param(params, "learning_rate", 5e-6)),
        "max_completion_length": int(_get_param(params, "max_completion_length", 512)),
        "max_prompt_length": int(_get_param(params, "max_prompt_length", 512)),
        "num_generations": int(_get_param(params, "num_generations", 4)),
        "logging_steps": int(_get_param(params, "logging_steps", 1)),
        "save_steps": int(_get_param(params, "save_steps", 50)),
        "gradient_checkpointing": bool(_get_param(params, "gradient_checkpointing", True)),
        "max_steps": int(_get_param(params, "max_steps", -1)),
        "bf16": use_bf16,
        "fp16": use_fp16,
        "report_to": report_targets,
    }


def create_grpo_config(params: Any, output_dir: str | Path, report_to: Sequence[str] | None = None):
    kwargs = build_grpo_config_kwargs(output_dir=output_dir, params=params, report_to=report_to)
    if GRPOConfig is None:  # pragma: no cover - exercised in tests through monkeypatch
        return kwargs
    return GRPOConfig(**kwargs)


def _maybe_build_quant_config():
    if torch is None or BitsAndBytesConfig is None or not torch.cuda.is_available():
        return None
    try:
        compute_dtype = torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
    except Exception:  # pragma: no cover - bitsandbytes runtime mismatch
        return None


def _prepare_model_and_tokenizer(model_name: str, params: Any):
    if AutoModelForCausalLM is None or AutoTokenizer is None or get_peft_model is None or LoraConfig is None:
        raise ImportError(f"Missing training dependencies: {', '.join(_missing_training_dependencies())}")

    quantization_config = _maybe_build_quant_config()
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if quantization_config is not None:
        model_kwargs.update(
            {
                "quantization_config": quantization_config,
                "device_map": "auto",
            }
        )
        if torch is not None:
            model_kwargs["torch_dtype"] = (
                torch.bfloat16 if getattr(torch.cuda, "is_bf16_supported", lambda: False)() else torch.float16
            )
    elif torch is not None:
        model_kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=int(_get_param(params, "lora_rank", 16)),
        lora_alpha=int(_get_param(params, "lora_alpha", 32)),
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM if TaskType is not None else "CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    return model, tokenizer


def _create_grpo_trainer(
    *,
    model,
    trainer_args,
    train_dataset,
    tokenizer,
    reward_fn,
    callbacks: Sequence[Any],
):
    if GRPOTrainer is None:
        raise ImportError("trl is required to construct a GRPOTrainer")

    init_signature = inspect.signature(GRPOTrainer.__init__)
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": trainer_args,
        "train_dataset": train_dataset,
        "reward_funcs": [reward_fn],
    }
    if "processing_class" in init_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in init_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    if "callbacks" in init_signature.parameters:
        trainer_kwargs["callbacks"] = [callback for callback in callbacks if callback is not None]

    trainer = GRPOTrainer(**trainer_kwargs)
    if "callbacks" not in init_signature.parameters:
        for callback in callbacks:
            if callback is not None and hasattr(trainer, "add_callback"):
                trainer.add_callback(callback)
    return trainer


def _load_training_dataset(runtime, dataset_source: str | Path | None = None):
    if dataset_source is not None:
        return ensure_prompt_dataset(load_dataset_source(dataset_source))
    loader = getattr(runtime, "load_dataset", None)
    if loader is None:
        raise ValueError("No dataset loader available")
    try:
        datasets = loader(dataset_types=["train"])
    except TypeError:
        datasets = loader(["train"])
    if isinstance(datasets, Mapping):
        train_dataset = datasets.get("train") or next(iter(datasets.values()))
    else:
        train_dataset = datasets
    return ensure_prompt_dataset(train_dataset)


def run_direct_grpo_training(
    runtime,
    *,
    dataset_source: str | Path | None = None,
    contrastive_payload: Mapping[str, Any] | None = None,
    telemetry=None,
    allow_missing_dependencies: bool = False,
):
    """Run GRPO training either through TransformerLab or a local adapter."""
    missing = _missing_training_dependencies()
    if missing:
        message = f"Missing training dependencies: {', '.join(missing)}"
        if not allow_missing_dependencies:
            raise ImportError(message)
        LOG.warning("%s; skipping live training.", message)
        return {"status": "dependency_unavailable", "missing": missing}

    params = getattr(runtime, "params", SimpleNamespace())
    model_name = str(_get_param(params, "model_name", "HuggingFaceTB/SmolLM-135M"))
    output_dir = Path(str(_get_param(params, "output_dir", "experiments/tlab_outputs")))
    adaptor_dir = Path(str(_get_param(params, "adaptor_output_dir", output_dir)))
    adaptor_dir.mkdir(parents=True, exist_ok=True)

    runtime.progress_update(5)
    model, tokenizer = _prepare_model_and_tokenizer(model_name, params)
    runtime.progress_update(25)

    train_dataset = _load_training_dataset(runtime, dataset_source=dataset_source)
    runtime.progress_update(30)

    reward_fn = build_decomposition_reward_fn()
    trainer_args = create_grpo_config(params, adaptor_dir, report_to=getattr(runtime, "report_to", []))
    callbacks = []
    progress_cb = getattr(runtime, "create_progress_callback", lambda **_kwargs: None)(framework="huggingface")
    if progress_cb is not None:
        callbacks.append(progress_cb)

    if bool(_get_param(params, "contrastive_enabled", True)):
        contrastive_config = {
            "loss_weights": {
                "correctness": float(_get_param(params, "contrastive_correctness_weight", 0.25)),
                "honesty": float(_get_param(params, "contrastive_honesty_weight", 0.40)),
                "contrast": float(_get_param(params, "contrastive_contrast_weight", 0.15)),
                "grpo": float(_get_param(params, "contrastive_grpo_weight", 0.20)),
            },
            "temperature": {
                "initial": float(_get_param(params, "temperature_initial", 0.1)),
                "final": float(_get_param(params, "temperature_final", 0.03)),
                "schedule": "exponential",
            },
        }
        contrastive_cb = build_contrastive_callback(contrastive_config)
        if contrastive_cb is not None:
            callbacks.append(contrastive_cb)

    trainer = _create_grpo_trainer(
        model=model,
        trainer_args=trainer_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        callbacks=callbacks,
    )

    if contrastive_payload:
        runtime.log_metric("contrastive_loss_preview", float(contrastive_payload.get("loss_preview", 0.0)), 0)
        if telemetry is not None:
            telemetry.log_batch(
                batch_idx=0,
                reward_stats={"mean_reward": 0.0, "reward_variance": 0.0},
                hardware_usage={},
                contrastive_stats={
                    "loss_correctness": float(contrastive_payload.get("loss_correctness", 0.0)),
                    "loss_honesty": float(contrastive_payload.get("loss_honesty", 0.0)),
                    "loss_contrast": float(contrastive_payload.get("loss_contrast", 0.0)),
                    "loss_grpo": float(contrastive_payload.get("grpo_loss", 0.0)),
                    "loss_total": float(contrastive_payload.get("loss_preview", 0.0)),
                    "triplets_mined": int(contrastive_payload.get("triplets_mined", 0)),
                },
            )

    runtime.progress_update(35)
    trainer.train()
    runtime.progress_update(90)
    trainer.save_model(str(adaptor_dir))
    tokenizer.save_pretrained(str(adaptor_dir))
    runtime.progress_update(100)
    return {"status": "trained", "output_dir": str(adaptor_dir)}


@tlab_trainer.job_wrapper()
def train():
    """TransformerLab entrypoint for GRPO + contrastive honesty training."""
    return run_direct_grpo_training(tlab_trainer)


def load_launch_bundle(
    hardware_profile: str = "single_gpu",
    config_path: Path | str = DEFAULT_CONFIG_PATH,
) -> LaunchBundle:
    wrapper = GRPORLWHFWrapper(config_path)
    return wrapper.build_launch_bundle(hardware_profile=hardware_profile)


__all__ = [
    "GRPORLWHFWrapper",
    "LaunchBundle",
    "LocalTLabTrainerAdapter",
    "build_contrastive_callback",
    "build_decomposition_reward_fn",
    "build_grpo_config_kwargs",
    "build_honesty_reward_fn",
    "create_grpo_config",
    "load_launch_bundle",
    "run_direct_grpo_training",
    "train",
]
