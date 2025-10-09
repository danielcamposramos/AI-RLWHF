# Training Scripts

Driver scripts to launch supervised pretraining, RLHF stages, and evaluation sweeps with Multi Vibe collaboration hooks.

## Upcoming Utilities
- `standby_runner.py`: Demonstrates Unsloth Standby initialization (`UNSLOTH_VLLM_STANDBY=1`) and shared weight execution for memory-efficient GRPO.
- `teacher_student_loop.py`: Coordinates Transformer Lab connectors, local Ollama inference, and reward aggregation using the rubric described in `docs/rlwhf-framework.md`.
- `reward_dashboard_export.py`: Streams honesty tuples from `data/processed/honesty_logs` into analytics-friendly CSV/JSON for experimentation dashboards.
