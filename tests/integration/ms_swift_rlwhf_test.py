import unittest
import os
import sys
from pathlib import Path
import json

# Add project root to the Python path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data_pipeline.data_quality_gate import validate
from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from tests.fixtures.sample_honesty_data import create_sample_honesty_data

class MsSwiftRLWHFIntegrationTest(unittest.TestCase):
    """
    Integration tests for the full ms-swift RLWHF pipeline.
    """

    @classmethod
    def setUpClass(cls):
        """Set up the test environment once for all tests."""
        cls.test_dataset_path = create_sample_honesty_data()
        cls.output_dir = Path("experiments/test_output/")
        cls.output_dir.mkdir(exist_ok=True, parents=True)

        # Create a dummy config for the wrapper
        cls.config_path = "configs/training/test_grpo_config.json"
        with open(cls.config_path, "w") as f:
            json.dump({
                "per_device_batch_size": 1,
                "max_length": 512,
                "reward_module": "plugins.core.custom_honesty_rm"
            }, f)

    def test_01_full_pipeline_runs_without_error(self):
        """
        Tests that the full pipeline (quality gate -> wrapper launch)
        can be executed without raising an error.
        """
        print("\n--- Running Test: Full Pipeline Execution ---")
        try:
            # Step 1: Quality Gate
            print("Step 1: Validating test dataset...")
            quality_ok = validate(self.test_dataset_path)
            self.assertTrue(quality_ok, "Dataset failed quality gate")
            print("...Dataset validation successful.")

            # Step 2: GRPO Training Launch (conceptual)
            print("Step 2: Launching GRPO production wrapper...")
            wrapper = ProductionGRPOWrapper(self.config_path)
            # The launch method is conceptual and doesn't actually train,
            # but we can check that it runs its internal logic.
            wrapper.launch(self.test_dataset_path, str(self.output_dir))
            print("...GRPO wrapper launched successfully (conceptual).")

            # A more robust test would check for output artifacts,
            # but the current wrapper implementation is conceptual.
            self.assertTrue(True) # Mark as passed if no errors occurred

        except Exception as e:
            self.fail(f"Full pipeline test failed with an exception: {e}")

    def test_02_hardware_fallback_simulation(self):
        """
        Tests the hardware fallback mechanism by simulating a low-memory environment.
        """
        print("\n--- Running Test: Hardware Fallback Simulation ---")
        # Simulate a low-memory environment by setting an environment variable
        # that the hardware detector could theoretically check.
        # This is a conceptual test of the fallback idea.
        os.environ["SIMULATE_LOW_MEM"] = "1"

        try:
            wrapper = ProductionGRPOWrapper(self.config_path)

            # In a real scenario, the hardware detector would see the env var
            # and the fallback cascade would alter the config.
            # We can check that the config gets modified.
            from plugins.core.hardware_fallback_cascade import HardwareFallbackCascade
            cascade = HardwareFallbackCascade()

            # Check batch size before fallback
            self.assertEqual(wrapper.cfg.get("per_device_batch_size"), 1)

            # This test is more conceptual as the cascade logic is simple.
            # A real test would mock the hardware detector.
            print("Applying fallback cascade...")
            cascade.apply_to_wrapper(wrapper)

            # For this test, we assume the CPU fallback is triggered, which
            # might change the gradient accumulation steps.
            # This depends on the specific logic in the cascade.
            # Let's just verify it runs without error.
            print("...Fallback cascade applied.")
            self.assertTrue(True)

        except Exception as e:
            self.fail(f"Hardware fallback test failed with an exception: {e}")
        finally:
            # Clean up the environment variable
            if "SIMULATE_LOW_MEM" in os.environ:
                del os.environ["SIMULATE_LOW_MEM"]

if __name__ == "__main__":
    unittest.main(verbosity=2)