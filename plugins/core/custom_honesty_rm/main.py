"""Custom honesty reward model plugin compatible with ms-swift GRPO pipelines."""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # Transformer Lab friendly decorator when available.
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover - offline usage
    class _DummyTrainer:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.params: Dict[str, Any] = {}

        def job_wrapper(self):
            def decorator(func):
                return func

            return decorator

        def progress_update(self, *_args, **_kwargs) -> None:
            return None

    tlab_trainer = _DummyTrainer()  # type: ignore

try:  # Optional torch helpers (quantization, device checks).
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional runtime
    torch = None  # type: ignore

UNCERTAINTY_TOKENS = {"unsure", "uncertain", "unknown", "not sure", "cannot confirm"}
HALLUCINATION_TOKENS = {"hallucinate", "fabricated", "invented", "made up", "nonsense"}
POSITIVE_TOKENS = {"accurate", "correct", "grounded", "verified", "substantiated"}


@dataclass
class RewardModelConfig:
    dataset_path: Path = Path("data/processed/honesty_logs/grpo_ready.jsonl")
    output_dir: Path = Path("models/reward/custom_honesty_rm")
    rubric_path: Path = Path("docs/rlwhf-framework.md")
    smoothing_factor: float = 0.15
    honesty_weight: float = 0.5
    uncertainty_bonus: float = 0.35
    disagreement_weight: float = 0.15
    score_floor: float = -2.0
    score_ceiling: float = 2.0
    quantization: str = "qlora"
    history_window: int = 8
    dtype: str = "bfloat16"
    hardware_target: str = "auto"


@dataclass
class HonestyRewardArtifact:
    config: RewardModelConfig
    statistics: Mapping[str, float]
    token_weights: Mapping[str, float]
    version: str = "0.1.0"

    def to_json(self) -> str:
        payload = {
            "config": _stringify_values(asdict(self.config)),
            "statistics": dict(self.statistics),
            "token_weights": dict(self.token_weights),
            "version": self.version,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)


def _stringify_values(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _stringify_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify_values(item) for item in value]
    return value


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    if not path.exists():
        return samples
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            samples.append(data)
    return samples


def _normalize_reward(value: Any, config: RewardModelConfig) -> float:
    try:
        reward = float(value)
    except (TypeError, ValueError):
        reward = 0.0
    span = config.score_ceiling - config.score_floor or 1.0
    reward = (reward - config.score_floor) / span
    return max(0.0, min(1.0, reward))


def _tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def _aggregate_statistics(
    samples: Iterable[Mapping[str, Any]],
    config: RewardModelConfig,
) -> Dict[str, float]:
    total = 0.0
    count = 0
    honest_hits = 0.0
    uncertainty_hits = 0.0
    hallucination_hits = 0.0
    for record in samples:
        reward = _normalize_reward(record.get("reward"), config)
        teacher_feedback = str(record.get("teacher_feedback", "")).lower()
        student_answer = str(record.get("student_answer", "")).lower()
        tokens = set(_tokenize(teacher_feedback) + _tokenize(student_answer))
        total += reward
        count += 1
        if tokens & POSITIVE_TOKENS:
            honest_hits += 1
        if tokens & UNCERTAINTY_TOKENS:
            uncertainty_hits += 1
        if tokens & HALLUCINATION_TOKENS:
            hallucination_hits += 1
    if count == 0:
        return {
            "avg_reward": 0.0,
            "honesty_ratio": 0.0,
            "uncertainty_ratio": 0.0,
            "hallucination_ratio": 0.0,
            "samples": 0.0,
        }
    return {
        "avg_reward": total / count,
        "honesty_ratio": honest_hits / count,
        "uncertainty_ratio": uncertainty_hits / count,
        "hallucination_ratio": hallucination_hits / max(1, count),
        "samples": float(count),
    }


def _derive_token_weights(stats: Mapping[str, float], config: RewardModelConfig) -> Dict[str, float]:
    weights = {
        "honesty": config.honesty_weight,
        "uncertainty": config.uncertainty_bonus,
        "hallucination": -config.honesty_weight,
    }
    # Smooth weights based on observed dataset bias.
    hallo = stats.get("hallucination_ratio", 0.0)
    if hallo > 0.2:
        weights["hallucination"] *= 1.5
    if stats.get("uncertainty_ratio", 0.0) < 0.05:
        weights["uncertainty"] *= 0.75
    return weights


class HonestyRewardModel:
    """Heuristic reward scorer that emulates ms-swift reward adapters."""

    def __init__(self, artifact: HonestyRewardArtifact) -> None:
        self.artifact = artifact

    def score(self, prompt: str, response: str, critique: str = "") -> Dict[str, float]:
        config = self.artifact.config
        weights = self.artifact.token_weights
        base_reward = self.artifact.statistics.get("avg_reward", 0.5)
        adjusted = base_reward
        tokens = set(_tokenize(response) + _tokenize(critique))
        if tokens & POSITIVE_TOKENS:
            adjusted += weights.get("honesty", 0.0)
        if tokens & UNCERTAINTY_TOKENS:
            adjusted += weights.get("uncertainty", 0.0)
        if tokens & HALLUCINATION_TOKENS:
            adjusted += weights.get("hallucination", -0.2)
        adjusted = max(0.0, min(1.0, adjusted))
        # Re-project to RLWHF scale (-2..2).
        span = config.score_ceiling - config.score_floor or 1.0
        scaled = adjusted * span + config.score_floor
        entropy = -adjusted * math.log(adjusted + 1e-9) - (1 - adjusted) * math.log(1 - adjusted + 1e-9)
        return {
            "score": float(round(scaled, 3)),
            "normalized_score": float(round(adjusted, 4)),
            "uncertainty": float(round(entropy, 4)),
        }

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / "honesty_reward_model.json"
        artifact_path.write_text(self.artifact.to_json(), encoding="utf-8")
        return artifact_path


def collect_reward_config(overrides: Optional[Mapping[str, Any]] = None) -> RewardModelConfig:
    params: Dict[str, Any] = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    if overrides:
        params.update(overrides)
    dataset_path = Path(params.get("dataset_path", RewardModelConfig().dataset_path))
    output_dir = Path(params.get("output_dir", RewardModelConfig().output_dir))
    config = RewardModelConfig(
        dataset_path=dataset_path,
        output_dir=output_dir,
        rubric_path=Path(params.get("rubric_path", RewardModelConfig().rubric_path)),
        smoothing_factor=float(params.get("smoothing_factor", RewardModelConfig().smoothing_factor)),
        honesty_weight=float(params.get("honesty_weight", RewardModelConfig().honesty_weight)),
        uncertainty_bonus=float(params.get("uncertainty_bonus", RewardModelConfig().uncertainty_bonus)),
        disagreement_weight=float(params.get("disagreement_weight", RewardModelConfig().disagreement_weight)),
        score_floor=float(params.get("score_floor", RewardModelConfig().score_floor)),
        score_ceiling=float(params.get("score_ceiling", RewardModelConfig().score_ceiling)),
        quantization=str(params.get("quantization", RewardModelConfig().quantization)),
        history_window=int(params.get("history_window", RewardModelConfig().history_window)),
        dtype=str(params.get("dtype", RewardModelConfig().dtype)),
        hardware_target=str(params.get("hardware_target", RewardModelConfig().hardware_target)),
    )
    return config


def build_reward_artifact(config: RewardModelConfig) -> HonestyRewardArtifact:
    samples = _read_jsonl(config.dataset_path)
    stats = _aggregate_statistics(samples, config)
    token_weights = _derive_token_weights(stats, config)
    artifact = HonestyRewardArtifact(config=config, statistics=stats, token_weights=token_weights)
    return artifact


def score_with_custom_rm(prompt: str, response: str, critique: str = "", artifact_path: Optional[Path] = None) -> Dict[str, float]:
    artifact = load_reward_artifact(artifact_path) if artifact_path else build_reward_artifact(RewardModelConfig())
    model = HonestyRewardModel(artifact)
    return model.score(prompt=prompt, response=response, critique=critique)


def _emit_artifact(artifact: HonestyRewardArtifact, config: RewardModelConfig) -> Dict[str, Any]:
    model = HonestyRewardModel(artifact)
    artifact_path = model.save(config.output_dir)
    metadata_path = config.output_dir / "metadata.json"
    metadata_payload = {
        "quantization": config.quantization,
        "dtype": config.dtype,
        "history_window": config.history_window,
        "hardware_target": config.hardware_target,
    }
    metadata_path.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
    return {"artifact_path": str(artifact_path), "metadata_path": str(metadata_path), "statistics": artifact.statistics}


def _maybe_set_device(config: RewardModelConfig) -> None:
    if torch is None:
        return
    if config.hardware_target == "auto":
        if torch.cuda.is_available():
            torch.set_default_device("cuda")  # type: ignore[attr-defined]
        elif torch.backends.mps.is_available():  # type: ignore[attr-defined]
            torch.set_default_device("mps")  # type: ignore[attr-defined]
        else:
            torch.set_default_device("cpu")  # type: ignore[attr-defined]


@tlab_trainer.job_wrapper()
def custom_honesty_reward_entrypoint(**overrides):
    progress_cb = getattr(tlab_trainer, "progress_update", None)
    if callable(progress_cb):
        progress_cb(5)
    config = collect_reward_config(overrides)
    _maybe_set_device(config)
    artifact = build_reward_artifact(config)
    if callable(progress_cb):
        progress_cb(60)
    payload = _emit_artifact(artifact, config)
    if callable(progress_cb):
        progress_cb(100)
    return payload


def load_reward_artifact(path: Path | None) -> HonestyRewardArtifact:
    if path is None:
        raise ValueError("Provide a path to an honesty reward artifact.")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config_payload = payload["config"]
    config = RewardModelConfig(
        dataset_path=Path(config_payload["dataset_path"]),
        output_dir=Path(config_payload["output_dir"]),
        rubric_path=Path(config_payload["rubric_path"]),
        smoothing_factor=config_payload["smoothing_factor"],
        honesty_weight=config_payload["honesty_weight"],
        uncertainty_bonus=config_payload["uncertainty_bonus"],
        disagreement_weight=config_payload["disagreement_weight"],
        score_floor=config_payload["score_floor"],
        score_ceiling=config_payload["score_ceiling"],
        quantization=config_payload["quantization"],
        history_window=config_payload["history_window"],
        dtype=config_payload["dtype"],
        hardware_target=config_payload["hardware_target"],
    )
    return HonestyRewardArtifact(config=config, statistics=payload["statistics"], token_weights=payload["token_weights"], version=payload.get("version", "0.1.0"))


def main() -> None:  # pragma: no cover - CLI helper
    overrides: Dict[str, Any] = {}
    dataset_path = os.environ.get("CUSTOM_RM_DATASET")
    if dataset_path:
        overrides["dataset_path"] = dataset_path
    custom_honesty_reward_entrypoint(**overrides)


if __name__ == "__main__":  # pragma: no cover
    main()
