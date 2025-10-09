# Custom Honesty Reward Model

This plugin distills RLWHF tuples into a compact reward artifact that ms-swift and Transformer Lab runners can reuse. It ingests JSONL tuples (prompt, student_answer, teacher_feedback, reward) and derives calibrated token weights that mirror the rubric in `docs/rlwhf-framework.md`.

## Capabilities
- Computes dataset statistics and reward scaling compatible with GRPO/DPO loops.
- Emits portable artifacts (`honesty_reward_model.json`) alongside hardware-aware metadata (`metadata.json`).
- Supports Unsloth Standby and ms-swift launches through `plugins/core/grpo_rlwhf_wrapper.py`.

Use the plugin from the Transformer Lab UI or via CLI:

```bash
python3 -m plugins.core.custom_honesty_rm.main
```
