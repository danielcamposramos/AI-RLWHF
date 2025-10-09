# Data Pipeline Scripts

Batch and streaming ETL helpers for preparing raw corpora, generating synthetic data, and pushing updates downstream.

## Planned Tasks
- Normalize teacher-student dialogue traces into JSONL/Parquet with schema `<prompt_id, model_role, text, reward, timestamp>`.
- Export rubric definitions from `configs/prompts/rubrics.yml` so plugins share identical scoring metadata.
- Stage honesty leaderboard aggregates for visualization notebooks inside `experiments/`.
