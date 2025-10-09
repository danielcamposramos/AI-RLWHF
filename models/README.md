# Models

Storage for checkpoints, adapter weights, tokenizer assets, and exported artifacts managed by Transformer Lab or auxiliary scripts.

## Subdirectories
- `checkpoints/`: Full model snapshots persisted from Transformer Lab fine-tunes or Unsloth Standby GRPO runs (git ignored).
- `adapters/`: LoRA/QLoRA weights optimized for the student model within the teacher-student RLWHF loop.
- `exports/`: Distilled or quantized models packaged for deployment; document provenance and honesty benchmarks alongside artifacts.
