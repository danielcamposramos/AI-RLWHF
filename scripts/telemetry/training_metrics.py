import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

class TrainingMetrics:
    """
    Enhanced telemetry for RLWHF training sessions.

    This class provides methods to log detailed, batch-level metrics during
    a training run and to save a final summary of the entire session.
    """

    def __init__(self, output_dir: str):
        """
        Initializes the telemetry logger.

        Args:
            output_dir: The directory where the telemetry files will be saved.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self.start_time = time.time()
        self.hardware_profile = {} # Should be populated from HardwareDetector

    def set_hardware_profile(self, profile: Dict[str, Any]):
        """Sets the hardware profile for this training session."""
        self.hardware_profile = profile

    def log_batch(self, batch_idx: int, reward_stats: dict, hardware_usage: dict):
        """
        Logs metrics for a single training batch.

        Args:
            batch_idx: The index of the current batch.
            reward_stats: A dictionary of reward statistics for the batch.
            hardware_usage: A dictionary of hardware usage metrics (e.g., GPU memory).
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "batch_idx": batch_idx,
            "elapsed_seconds": time.time() - self.start_time,
            "reward_stats": reward_stats,
            "hardware_usage": hardware_usage
        }

        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def finalize(self, total_batches: int, final_rewards: dict):
        """
        Finalizes the training session and saves a summary report.

        Args:
            total_batches: The total number of batches processed in the session.
            final_rewards: A dictionary containing the final aggregated reward statistics.
        """
        summary = {
            "session_complete": True,
            "total_batches": total_batches,
            "total_time_seconds": time.time() - self.start_time,
            "final_rewards": final_rewards,
            "hardware_profile": self.hardware_profile
        }

        summary_file = self.output_dir / "session_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Training session summary saved to: {summary_file}")

if __name__ == '__main__':
    # Example Usage
    telemetry = TrainingMetrics(output_dir="experiments/telemetry/example_run")

    # Mock hardware profile
    mock_hardware_profile = {
        "system": "Linux",
        "cpu_count": 8,
        "cuda_available": True,
        "gpu_details": [{"name": "NVIDIA GeForce RTX 3090", "memory_gb": 24}]
    }
    telemetry.set_hardware_profile(mock_hardware_profile)

    # Simulate a few training batches
    for i in range(5):
        time.sleep(0.1)
        telemetry.log_batch(
            batch_idx=i,
            reward_stats={"mean_reward": 2.5 - i * 0.1, "reward_variance": 0.5},
            hardware_usage={"gpu_mem_used_gb": 10 + i * 0.2}
        )

    # Finalize the session
    telemetry.finalize(
        total_batches=5,
        final_rewards={"mean_reward": 2.3, "reward_variance": 0.6}
    )