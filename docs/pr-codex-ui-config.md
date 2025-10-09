# PR: Configurable Multi-Teacher Toggles and Offline Fallbacks

## Summary
- Introduced feature toggles plus UI-aligned slot metadata (`configs/training/feature_toggles.json`, `plugins/core/multi-teacher-aggregator/index.json`) so Transformer Lab exposes single vs multi-teacher workflows, API/local/Ollama selections, and fallback modes.
- Expanded the multi-teacher aggregator, runners, and Unsloth Standby helper to honor slot definitions, internet toggles, offline datasets, and fallback preferences without halting evaluations.
- Added utility modules, offline reference samples, documentation refreshes, and integration tests covering toggle-aware behaviors and logger cleanup.

## Testing
- `pytest tests/test_multi_teacher_integration.py -q`
