"""Unified command entry point for running the RLWHF ms-swift pipeline."""
from __future__ import annotations

import fire

from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from scripts.data_pipeline.data_quality_gate import validate


def launch_full_cycle(
    dataset_path: str,
    output_dir: str,
    config_path: str = "configs/transformer-lab/grpo_config.yaml",
    simulate_low_mem: bool = False,
) -> None:
    """Run quality gate then invoke the adaptive GRPO launcher."""
    if not validate(dataset_path):
        raise ValueError(f"Dataset failed quality gate: {dataset_path}")
    wrapper = ProductionGRPOWrapper(config_path)
    summary = wrapper.launch(dataset_path, output_dir, simulate_low_mem=simulate_low_mem)
    print(f"[master_rlwhf_launcher] Completed launch -> {summary}")


def main() -> None:
    fire.Fire({"launch": launch_full_cycle})


if __name__ == "__main__":  # pragma: no cover
    main()
