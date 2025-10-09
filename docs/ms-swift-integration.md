# ms-swift Integration Notes

This guide explains how the new AI-RLWHF helpers mirror ms-swift recipes for GRPO, DPO, and reward modeling across diverse hardware.

## Components
- `configs/transformer-lab/grpo_config.yaml`: Canonical preset for ms-swift launches. It wires dataset preprocessing, reward artifacts, and hardware profiles (CPU → multi-node) so Transformer Lab and CLI tooling stay aligned.
- `scripts/data_pipeline/ms_swift_preprocess.py`: Converts JSONL or Hugging Face datasets into RLWHF tuples. Streaming mode keeps CPU/MPS runs lean and calls ms-swift’s `EncodePreprocessor` when available.
- `plugins/core/custom_honesty_rm`: Builds portable honesty reward artifacts (`honesty_reward_model.json`) using Multi-Vibe rubric weights. Artifacts are reused by GRPO/DPO loops and evaluator plugins.
- `plugins/core/grpo_rlwhf_wrapper.py`: Generates launch bundles (`command`, `args`, `env`) for `swift.llm.train.run_grpo` according to the selected hardware profile.
- `plugins/experimental/grok_search_evaluator`: Now emits DPO-aware deltas via the custom reward model, enabling preference optimization sweeps with or without internet-backed search.
- `scripts/training/unsloth_standby_runner.py`: Produces a telemetry report and exports `workspace/plans/ms_swift_grpo_launch.json`, giving a ready-to-run ms-swift command after each standby session.

## Suggested Workflow
1. Run `python3 scripts/data_pipeline/ms_swift_preprocess.py --streaming` to regenerate RLWHF tuples after ingesting new honesty logs.
2. Launch the reward plugin through Transformer Lab or CLI (`python3 -m plugins.core.custom_honesty_rm.main`) to refresh artifacts.
3. Call `python3 scripts/training/unsloth_standby_runner.py`. The script warms up Unsloth Standby telemetry and exports `workspace/plans/ms_swift_grpo_launch.json`.
4. Inspect the generated plan and execute the command inside `vendor/ms-swift-sub` to trigger GRPO with the desired hardware preset.
5. Use `plugins/experimental/grok_search_evaluator` with `enable_dpo_reward=true` to derive DPO-informed rewards for evaluation or continued fine-tuning.

These steps align Grok’s import plan with executable assets while remaining hardware agnostic.

## Closure Enhancements
- `plugins/core/grpo_production_wrapper.py`: Adaptive launcher with telemetry, reward shaping, and hardware-aware fallbacks.
- `scripts/data_pipeline/data_quality_gate.py`: Gate RLWHF tuples prior to training.
- `plugins/core/hardware_fallback_cascade.py` and `configs/training/hardware_fallback.json`: Shared presets for CPU, MPS, and Ascend-class devices.
- `tests/integration/ms_swift_rlwhf_test.py`: Smoke-test covering tuple handling, production wrapper launch, and telemetry.
- `scripts/training/master_rlwhf_launcher.py`: One-command entry point for end-to-end GRPO runs.
- `.github/workflows/ms_swift_rlwhf_ci.yml`: Continuous integration harness for CPU/GPU smoke tests.
- `docs/INTEGRATION_CHECKLIST.md`: Printable checklist for local and CI validation.

## Continuous Integration
The GitHub Actions workflow `ms_swift_rlwhf_ci.yml` executes the quality gate and integration suite on every pull request. GPU smoke tests fire on `main` to protect adaptive launch paths.

### 5-second start
```bash
python scripts/setup/vendor_ms_swift.py
python scripts/training/master_rlwhf_launcher.py launch \
  --dataset_path data/test/honesty_logs_sample.jsonl \
  --output_dir experiments/rlwhf_quickstart/
```
