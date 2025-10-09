# Plugins

Source code for Transformer Lab plugins representing ingestion, generation, evaluation, and oversight layers.

## Plugin Families
- `core/ingestion`: Deterministic corpus loaders, deduplicators, and metadata enrichers.
- `core/teacher`: Evaluator plugins leveraging the RLWHF rubric (-2 to +2) for honesty grading and reward emission.
- `core/reward`: Aggregators that fuse multi-teacher outputs and publish JSONL tuples to `data/processed/honesty_logs`.
- `synthetic-builders`: Multi Vibe Coding In Chain prompt orchestrators for dataset expansion.
- `experimental/`: Sandbox for novel connector experiments (for example `grok_search_evaluator`, internet/offline hybrids, and upcoming teacher variants).

Refer to `docs/plugin-blueprints.md` and the Transformer Lab plugin tutorial for manifest (`index.json`), bootstrap (`setup.sh`), and runtime (`main.py`) expectations.
