# MS-Swift Integration Checklist

This document provides a step-by-step checklist to ensure the `ms-swift` integration is set up and functioning correctly within the AI-RLWHF environment.

## Phase 1: Setup ✅
- [ ] Run the main setup script: `bash scripts/setup/setup_ms_swift_integration.sh`
- [ ] Verify that all dependencies from `requirements-ms-swift.txt` were installed successfully.
- [ ] Confirm that sample data has been created at `data/test/honesty_logs_sample.jsonl`.
- [ ] Check that the `vendor/ms-swift-sub` directory has been created and populated.

## Phase 2: Hardware Detection ✅
- [ ] Test the hardware detector script to ensure it correctly profiles your system's hardware. Run the following command and verify the output:
  ```bash
  python -c "from plugins.core.hardware_detector import HardwareDetector; hd = HardwareDetector(); print(hd.hardware_profile)"
  ```

## Phase 3: Data Pipeline ✅
- [ ] Validate the sample data using the data quality gate to ensure it passes the checks:
  ```bash
  python scripts/data_pipeline/data_quality_gate.py data/test/honesty_logs_sample.jsonl
  ```
- [ ] (Manual) Inspect the code for the metadata extension in `scripts/data_pipeline/extended_metadata_handler.py` to understand how tracking data is added.

## Phase 4: Training Integration ✅
- [ ] Run the integration test suite to verify that all components of the pipeline work together correctly:
  ```bash
  python -m unittest tests/integration/ms_swift_rlwhf_test.py
  ```
- [ ] Perform a full test run using the master launcher script with the sample data:
  ```bash
  python scripts/training/master_rlwhf_launcher.py launch --dataset_path data/test/honesty_logs_sample.jsonl --output_dir experiments/test_run/
  ```

## Phase 5: Production Readiness ✅
- [ ] If you have access to a GitHub repository, verify that the CI pipeline defined in `.github/workflows/ms_swift_rlwhf_ci.yml` passes successfully.
- [ ] (Advanced) Test hardware fallback scenarios by modifying the `hardware_detector.py` script to simulate different environments.
- [ ] After a training run, inspect the output directory (e.g., `experiments/test_run/`) to validate that telemetry logs have been created.

## Quick Start
For a quick start after cloning the repository, run the following commands:

```bash
# Set up the environment and dependencies
bash scripts/setup/setup_ms_swift_integration.sh

# Run a full training cycle with sample data
python scripts/training/master_rlwhf_launcher.py launch --dataset_path data/test/honesty_logs_sample.jsonl --output_dir experiments/my_first_rlwhf/
```
This will set up the environment, run the data quality checks, apply hardware-aware configurations, and launch the training process, providing a complete end-to-end test of the system.