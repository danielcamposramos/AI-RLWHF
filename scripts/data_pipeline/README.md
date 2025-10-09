# Data Pipeline Scripts

Batch and streaming ETL helpers for preparing raw corpora, generating synthetic data, and pushing updates downstream.

## Planned Tasks
- Normalize teacher-student dialogue traces into JSONL/Parquet with schema `<prompt_id, model_role, text, reward, timestamp>`.
- Export rubric definitions from `configs/prompts/rubrics.yml` so plugins share identical scoring metadata.
- Stage honesty leaderboard aggregates for visualization notebooks inside `experiments/`.

## New Utilities
- `ms_swift_preprocess.py`: Streams JSONL or Hugging Face datasets into RLWHF honesty tuples while optionally invoking ms-swift's `EncodePreprocessor`. Generated files (default `data/processed/honesty_logs/grpo_ready.jsonl`) feed directly into the custom honesty reward plugin and GRPO wrappers.
- `data_quality_gate.py`: Fast CLI validator that rejects malformed honesty tuples ahead of training.
- `rlwhf_tuple_handler.py`: Consolidates workspace traces into normalized RLWHF JSONL output and applies extended metadata via `extended_metadata_handler.py`.
