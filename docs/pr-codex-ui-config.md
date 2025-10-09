# PR: Configurable Multi-Teacher Toggles and Offline Fallbacks

## Summary
- Introduced feature toggles plus UI-aligned slot metadata (`configs/training/feature_toggles.json`, `plugins/core/multi-teacher-aggregator/index.json`) so Transformer Lab exposes single vs multi-teacher workflows, API/local/Ollama selections, and fallback modes.
- Expanded the multi-teacher aggregator, runners, and Unsloth Standby helper to honor slot definitions, internet toggles, offline datasets, and fallback preferences without halting evaluations.
- Added the experimental `grok_search_evaluator` plugin with cached Grok search, offline fallbacks, and JSONL logging for dashboard deltas.
- Refreshed visualization utilities (search vs static delta), prompt presets, offline references, and test coverage for the new workflow.

## Testing
- `pytest tests/test_multi_teacher_integration.py -q`
