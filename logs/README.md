# Logs

Centralized log output for training, evaluation, and plugin runs. Stream logs to subfolders per phase for debuggability.

## Recommended Channels
- `training/`: RLWHF fine-tune metrics, GRPO reward summaries, GPU telemetry (with Unsloth Standby indicators).
- `plugins/`: Transformer Lab plugin stdout/stderr (teacher evaluators, reward aggregators, synthetic builders).
- `connectors/`: HTTP traces or broker logs for local Ollama and remote API interactions.
