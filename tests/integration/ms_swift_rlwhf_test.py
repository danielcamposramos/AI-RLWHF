"""Integration smoke tests for the ms-swift RLWHF toolchain."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from plugins.core.grpo_production_wrapper import ProductionGRPOWrapper
from scripts.data_pipeline.data_quality_gate import validate
from scripts.data_pipeline.rlwhf_tuple_handler import RLWHFTupleHandler
from tests.fixtures.sample_honesty_data import create_sample_honesty_data


class MsSwiftRLWHFIntegrationTest(unittest.TestCase):
    """Run an end-to-end pipeline using lightweight fixtures."""

    def setUp(self) -> None:
        self.dataset_path = create_sample_honesty_data()
        self.output_dir = Path(tempfile.mkdtemp(prefix="ms_swift_rlwhf_"))
        os.environ["SIMULATE_LOW_MEM"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("SIMULATE_LOW_MEM", None)

    def test_quality_gate_and_tuple_handler(self) -> None:
        self.assertTrue(validate(str(self.dataset_path)))
        handler = RLWHFTupleHandler()
        tuples = handler.process_workspace_logs(Path(self.dataset_path).parent)
        self.assertGreaterEqual(len(tuples), 1)
        generated = handler.create_training_dataset(tuples, self.output_dir / "processed.jsonl")
        self.assertTrue(generated.exists())

    def test_production_wrapper_launch(self) -> None:
        wrapper = ProductionGRPOWrapper()
        summary = wrapper.launch(str(self.dataset_path), str(self.output_dir))
        summary_path = self.output_dir / "production_wrapper_summary.json"
        self.assertTrue(summary_path.exists())
        self.assertIn("grpo_args", summary)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
