# PR: Configurable Multi-Teacher Toggles and Offline Fallbacks

## Summary
- Introduced feature toggles and defaults (`configs/training/feature_toggles.json`) plus UI-exposed plugin parameters so internet-backed evaluators can be switched on/off without code changes.
- Expanded the multi-teacher aggregator, training runners, and Unsloth Standby helper to respect internet/offline toggles, fallback strategies, and offline dataset comparisons.
- Added utility modules, offline reference samples, documentation refreshes, and integration tests covering the new behavior.

## Testing
- `pytest tests/test_multi_teacher_integration.py -q`
