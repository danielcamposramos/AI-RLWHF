import fire
import sys
from pathlib import Path

# Ensure the script can find the custom modules
sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.data_pipeline.data_quality_gate import validate
from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from plugins.core.hardware_fallback_cascade import HardwareFallbackCascade

def launch_full_cycle(dataset_path: str, output_dir: str, config_path: str = "configs/transformer-lab/grpo_config.yaml"):
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

    # Step 4: Launch Training
    print(f"4. Launching training process. Output will be in: {output_dir}")
    wrapper.launch(dataset_path, output_dir)
    print("\n--- AI-RLWHF Master Launch Cycle Complete ---")
    print(f"Cycle complete; artifacts saved in {output_dir}")

if __name__ == "__main__":
    fire.Fire({"launch": launch_full_cycle})