# Honesty Logs

Aggregation-friendly outputs produced by teacher evaluators and the multi-teacher aggregator.

## Conventions
- `multi_teacher_aggregation.jsonl`: consensus metrics for each prompt and student answer pair, including slot metadata (`teacher_mode`, connection definitions, fallback selections).
- `grok_search_evaluator.jsonl`: raw outputs from the Grok search evaluator plugin (prompt, student answer, reward, feedback, search metadata).
- Additional per-teacher JSONL streams can be cached alongside aggregation data as other evaluators land.
- Schemas align with `docs/rlwhf-framework.md` and `docs/grok-integration-plan.md` for replay and dashboard scripts.
