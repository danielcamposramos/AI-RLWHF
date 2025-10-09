# MS-Swift Integration Checklist

## Phase 1 · Setup
- [ ] Run `scripts/setup/vendor_ms_swift.py` to clone the ms-swift vendor tree.
- [ ] Install dependencies with `pip install -r requirements-ms-swift.txt` or `scripts/setup/setup_ms_swift_integration.sh`.
- [ ] Confirm seeded dataset exists at `data/test/honesty_logs_sample.jsonl`.

## Phase 2 · Hardware Detection
- [ ] Inspect detected hardware:  
  `python -c "from plugins.core.hardware_detector import HardwareDetector;print(HardwareDetector().hardware_profile)"`
- [ ] Adjust `configs/training/hardware_fallback.json` if custom presets are required.

## Phase 3 · Data Pipeline
- [ ] Validate tuples: `python scripts/data_pipeline/data_quality_gate.py data/test/honesty_logs_sample.jsonl`
- [ ] Generate processed tuples:  
  `python -c "from scripts.data_pipeline.rlwhf_tuple_handler import RLWHFTupleHandler; RLWHFTupleHandler().create_training_dataset([], 'data/processed/honesty_logs/processed.jsonl')"`

## Phase 4 · Training
- [ ] Launch pipeline via `python scripts/training/master_rlwhf_launcher.py launch --dataset_path data/test/honesty_logs_sample.jsonl --output_dir experiments/rlwhf_smoke/`
- [ ] Monitor telemetry artifacts in `experiments/rlwhf_smoke/telemetry`.

## Phase 5 · Testing & CI
- [ ] Run integration tests: `python -m unittest tests/integration/ms_swift_rlwhf_test.py`
- [ ] Ensure GitHub Actions workflow `.github/workflows/ms_swift_rlwhf_ci.yml` passes.

## Quick Start
```bash
python scripts/setup/setup_ms_swift_integration.sh
python scripts/training/master_rlwhf_launcher.py launch \
  --dataset_path data/test/honesty_logs_sample.jsonl \
  --output_dir experiments/rlwhf_quickstart/
```
