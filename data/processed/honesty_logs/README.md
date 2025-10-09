# Honesty Logs

Aggregation-friendly outputs produced by teacher evaluators and the multi-teacher aggregator.

## Conventions
- `multi_teacher_aggregation.jsonl`: consensus metrics for each prompt and student answer pair.
- Per-teacher JSONL streams (for example `grok-search-evaluator.jsonl`) may be cached alongside aggregation data.
- Schemas align with `docs/rlwhf-framework.md` and `docs/grok-integration-plan.md` for replay and dashboard scripts.
