import json
import sys
from pathlib import Path
from typing import Any

try:  # pragma: no cover - CLI helper only
    import fire
except Exception:  # pragma: no cover
    fire = None  # type: ignore

# Ensure the script can find the custom modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data_pipeline.data_quality_gate import validate
from scripts.data_pipeline.triplet_miner import load_tuples, mine_triplets
from scripts.telemetry.training_metrics import TrainingMetrics
from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from plugins.core.hardware_fallback_cascade import HardwareFallbackCascade

try:  # pragma: no cover - optional dependency for preflight validation
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


def _load_json(path: str | Path, default: dict[str, Any]) -> dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return dict(default)
    with cfg_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_contrastive_payload(dataset_path: str, config: dict[str, Any]) -> dict[str, Any]:
    tuples = load_tuples(dataset_path)
    mining = dict(config.get("triplet_mining", {}))
    triplets = mine_triplets(
        tuples,
        min_reward_gap=float(mining.get("min_reward_gap", 1)),
        max_triplets_per_prompt=int(mining.get("max_triplets_per_prompt", 10)),
        fragment_min_length=int(mining.get("fragment_min_length", 10)),
    )
    payload: dict[str, Any] = {
        "enabled": bool(config.get("contrastive_enabled", False)),
        "triplets_mined": len(triplets),
        "loss_weight": float(config.get("loss_weights", {}).get("grpo", 0.20)),
        "loss_preview": 0.0,
        "loss_correctness": 0.0,
        "loss_honesty": 0.0,
        "loss_contrast": 0.0,
    }
    if not payload["enabled"] or not triplets or torch is None:
        return payload

    try:
        from plugins.core.contrastive_loss import ContrastiveHonestyLoss
    except Exception:
        return payload

    batch_size = min(4, len(triplets))
    embedding_dim = int(config.get("embedding", {}).get("dim", 256))
    embeddings = {
        "question": torch.randn(batch_size, embedding_dim),
        "positive_fragments": torch.randn(batch_size, embedding_dim),
        "negative_fragments": torch.randn(batch_size, 2, embedding_dim),
        "response": torch.randn(batch_size, embedding_dim),
        "honesty_signals": torch.randn(batch_size, embedding_dim),
        "missing_honesty": torch.randn(batch_size, 2, embedding_dim),
        "better_response": torch.randn(batch_size, embedding_dim),
        "worse_response": torch.randn(batch_size, embedding_dim),
    }
    loss_module = ContrastiveHonestyLoss(config)
    losses = loss_module(embeddings, step=0, total_steps=max(1, len(triplets)))
    payload.update(
        {
            "loss_preview": float(losses["total"].item()),
            "loss_correctness": float(losses["correctness"].item()),
            "loss_honesty": float(losses["honesty"].item()),
            "loss_contrast": float(losses["contrast"].item()),
        }
    )
    return payload


def launch_full_cycle(
    dataset_path: str,
    output_dir: str,
    config_path: str = "configs/transformer-lab/grpo_config.yaml",
    contrastive_config_path: str = "configs/training/contrastive_config.json",
):
    """
    Launches a full RLWHF training and evaluation cycle.

    This master script orchestrates the entire process:
    1. Validates the input dataset using the data quality gate.
    2. Initializes the production GRPO wrapper with the specified config.
    3. Applies the hardware fallback cascade to adapt the config to the environment.
    4. Launches the training process.

    Args:
        dataset_path: Path to the JSONL file containing the training data.
        output_dir: Path to the directory where training artifacts will be saved.
        config_path: Path to the GRPO configuration YAML file.
    """
    print("--- Starting AI-RLWHF Master Launch Cycle ---")
    telemetry = TrainingMetrics(output_dir=output_dir)

    # Step 1: Validate dataset quality
    print(f"1. Validating data quality for: {dataset_path}")
    if not validate(dataset_path):
        print("\nERROR: Dataset failed quality gate. Please fix the issues and retry.", file=sys.stderr)
        sys.exit(1)
    print("   Dataset quality check passed.")

    # Step 2: Initialize Production GRPO Wrapper
    print(f"2. Initializing GRPO wrapper with config: {config_path}")
    if not Path(config_path).exists():
        print(f"\nERROR: Config file not found at {config_path}", file=sys.stderr)
        # Create a default config if it doesn't exist to make it runnable out of the box
        default_cfg = {
            "per_device_batch_size": 4,
            "max_length": 2048
        }
        Path(config_path).parent.mkdir(exist_ok=True, parents=True)
        with open(config_path, 'w') as f:
            import json
            json.dump(default_cfg, f)
        print(f"   Created a default config file at {config_path}")

    wrapper = ProductionGRPOWrapper(config_path)
    print("   Wrapper initialized.")

    # Step 3: Apply Hardware Fallback Cascade
    print("3. Applying hardware fallback cascade...")
    cascade = HardwareFallbackCascade()
    cascade.apply_to_wrapper(wrapper)
    print("   Hardware-aware configuration applied.")

    print(f"4. Preparing contrastive configuration from: {contrastive_config_path}")
    contrastive_cfg = _load_json(
        contrastive_config_path,
        default={
            "contrastive_enabled": False,
            "loss_weights": {"grpo": 0.20},
            "triplet_mining": {},
            "embedding": {"dim": 256},
        },
    )
    contrastive_payload = _build_contrastive_payload(dataset_path, contrastive_cfg)
    print(
        f"   Contrastive enabled: {contrastive_payload['enabled']} "
        f"(triplets mined: {contrastive_payload['triplets_mined']})"
    )

    # Step 4: Launch Training
    print(f"5. Launching training process. Output will be in: {output_dir}")
    wrapper.launch(dataset_path, output_dir, contrastive_payload=contrastive_payload, telemetry=telemetry)
    telemetry.finalize(
        total_batches=1,
        final_rewards={"mean_reward": 0.0, "reward_variance": 0.0},
        contrastive_summary=contrastive_payload if contrastive_payload.get("enabled") else None,
    )
    print("\n--- AI-RLWHF Master Launch Cycle Complete ---")
    print(f"Cycle complete; artifacts saved in {output_dir}")

if __name__ == "__main__":
    if fire is None:
        raise SystemExit("fire is required for CLI usage: pip install fire")
    fire.Fire({"launch": launch_full_cycle})
