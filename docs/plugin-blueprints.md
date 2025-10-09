# Plugin Blueprints

## Objectives
- Provide reproducible patterns for Transformer Lab plugins covering ingestion, synthesis, evaluation, and oversight.
- Encourage composable, memory aware flows with clear dependency isolation.

## Core Components
1. Dataset Loader Plugin: Parses corpora, handles sharding, and registers metadata.
2. Honesty Feedback Plugin: Scores generated samples against honesty rubrics and user annotations.
3. Synthetic Builder Plugin: Creates multi model prompt cascades to draft new samples with traceable metadata.
4. Evaluation Harness Plugin: Runs tiered evaluation suites, aggregates metrics, and flags drift.
5. Teacher Evaluator Plugin: Wraps the RLWHF scoring rubric (-2 to +2) and emits structured critiques for each student generation.
6. Reward Aggregator Plugin: Fuses multi-teacher feedback (Codex, Grok, Qwen, etc.) with configurable weighting and consensus strategies.

## Design Norms
- Plugins emit structured JSON or Parquet to `data/processed` or `data/synthetic`.
- Use streaming generators to minimize memory usage.
- Keep any credential material outside this repository.
- Provide connector blocks that accept Transformer Lab API credentials, local endpoint URLs (Ollama at `http://localhost:11434`), or remote provider tokens. Keep prompt references in `configs/prompts/` so changing a template updates all invocation paths simultaneously.
- Export `setup.sh` dependencies (e.g., `uv pip install datasets unsloth`) and keep `index.json` metadata aligned with Transformer Lab discovery requirements.
- Reference `docs/rlwhf-framework.md` for teacher/student orchestration guidance and memory-efficient Unsloth Standby usage.

Reference: https://r.jina.ai/https://lab.cloud/blog/how-to-plugin
