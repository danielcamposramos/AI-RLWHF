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

## Teacher Evaluator Variants

### Baseline Teacher Evaluator
- Accepts `(prompt, student_answer)` and emits rubric scores with textual critiques.
- Relies on local heuristics or static corpora; ideal for offline evaluation or smoke tests.
- Emits JSONL rows into `data/processed/honesty_logs/` for downstream analytics.

### Search-Enhanced Teacher Evaluator
- Extends the baseline evaluator with Grok (or similar) live search capability per `docs/grok-integration-plan.md`.
- Adds configurable query templates and retry logic so both Transformer Lab and local runner scripts reuse prompts consistently.
- Outputs `(prompt, student_answer, search_context, teacher_feedback, reward)` tuples to support memory-efficient GRPO training and aggregated dashboards.

### Multi-Teacher Reward Aggregator
- Consumes individual teacher payloads (`score`, `feedback`) and produces consensus metrics (-2…+2) with disagreement analysis.
- Supports weighting strategies (`weighted_average`, `majority_vote`, `confidence_weighted`) defined in `plugins/core/multi-teacher-aggregator/index.json`.
- Logs aggregation summaries to `data/processed/honesty_logs/multi_teacher_aggregation.jsonl` for visualization utilities in `scripts/visualization/`.
- Exposes toggles (`enable_internet_teachers`, `enable_offline_validation`, `fallback_mode`) so Transformer Lab UI surfaces the same feature switches as `configs/training/feature_toggles.json`.
- UI Pathways:
  - `teacher_mode` (`single` or `multiple`) decides whether the UI renders a single slot selector or an expandable list of slots.
  - Each slot parameter surfaces a connection dropdown (`api`, `transformerlab_local`, `ollama`). Selecting `api` enables API profile selectors and a button linking to Transformer Lab’s API key screen; `ollama` prompts for an endpoint and triggers a model list refresh; `transformerlab_local` reveals Transformer Lab workspace model selectors.

Reference: https://r.jina.ai/https://lab.cloud/blog/how-to-plugin
