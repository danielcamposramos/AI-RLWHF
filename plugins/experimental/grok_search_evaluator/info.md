# Grok Search Teacher Evaluator

Optional internet-enabled teacher evaluator that compares student answers against offline references and Grok snippets to emit honesty rewards.

## Parameters
- `dataset_path`: JSONL file with `prompt` and `student_answer` columns.
- `use_internet`: Toggle Grok search. False keeps evaluations offline.
- `api_endpoint` / `api_key_env`: Endpoint/key to fetch search snippets (default `XAI_API_KEY`). Press the API-key button in Transformer Lab to configure securely.
- `cache_path`: JSONL cache for snippet results to reduce token usage.
- `offline_reference_path`: Local honesty tuples for offline fallback.
- `max_examples`: Hard cap per run.
- `enable_dpo_reward`: When true, compares the student answer with a preference column (default `preferred_answer`) using the custom honesty reward model for DPO-style deltas.
- `reward_artifact_path`: Optional path to `honesty_reward_model.json`; falls back to the on-the-fly heuristic when missing.

## Outputs
Writes JSONL rows to `output_path`, each containing the original prompt, student answer, reward (-2…+2), textual feedback, captured search snippets, and optional `dpo_*` metrics. Integrates with the multi-teacher aggregator by emitting [`grok-search-evaluator`] as the teacher name.
