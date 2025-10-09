# Grok Integration Plan: Search-Enhanced Honesty Evaluation

This document outlines the strategy for integrating Grok's internet-aware search capabilities into the AI-RLWHF framework, primarily focusing on enhancing the teacher evaluator's ability to assess honesty. This plan builds upon Grok's kickoff contribution and the existing RLWHF framework.

## 1. Objective

Leverage Grok's real-time search and fact-checking abilities to create more robust and dynamic teacher evaluator plugins. These plugins score student model responses on honesty by comparing them against information retrieved from the internet, helping reduce hallucinations and improve the reward signal in RLWHF loops.

## 2. Core Components

### 2.1 Search-Enhanced Teacher Evaluator Plugin
- **Purpose:** Evaluator plugin for Transformer Lab that ingests a prompt and a student response.
- **Process:**
  1. Issue Grok-powered search queries derived from the prompt context.
  2. Compare student responses against retrieved snippets and rubric guidance in `docs/rlwhf-framework.md`.
  3. Emit critiques (alignment, missing context, evidence links) plus rubric scores.
- **Implementation:** Lives in `plugins/experimental/grok-search-evaluator/` using the `@tlab_trainer.job_wrapper()` convention.

### 2.2 Configurable Search Parameters
- Search query templates (for example `"Verify facts on {topic}"`) stored alongside index metadata.
- Parameterized model choices (`grok-4`, `grok-code-fast-1`, etc.) surfaced through `index.json`.

### 2.3 Honesty Log Enhancements
- Extend `data/processed/honesty_logs/` schema to include search snippets and provenance metadata: `(prompt, student_answer, search_context, teacher_feedback, reward)`.

### 2.4 Multi-Teacher Integration
- Outputs remain schema-compatible with the multi-teacher aggregator so Grok feedback blends with Codex, Kimi, GLM, DeepSeek, and Qwen signals.
- UI toggles (`teacher_mode`, `teacher_count`, slot connection selectors) are shared between the aggregator manifest and the runner configuration so Grok can choose single-teacher (API or local) runs or arbitrarily many multi-teacher cascades.

## 3. Development Steps (Refined Goals)
1. **Framework scaffolding (Codex):** Baseline RLWHF documentation and repo layout (complete).
2. **Plugin scaffold (Grok):** Initial evaluator outline with rubric-aware scoring (complete).
3. **Implement search evaluator (Qwen Max/Codex):** `plugins/experimental/grok_search_evaluator` now emits rubric-aware scores with cached Grok snippets and offline fallbacks.
4. **Integrate evaluator into training loop:** Update orchestration scripts to load Grok context alongside other teachers.
5. **Enhance honesty logs:** Persist search context and rubric decisions for reproducibility.
6. **Document integration:** Keep `docs/plugin-blueprints.md` and `docs/rlwhf-framework.md` synchronized with evaluator improvements.
7. **Dashboard/reporting:** Generate honesty trend dashboards contrasting search-enabled and offline evaluators.
8. **UI toggle surfacing:** Mirror `feature_toggles.json` options in Transformer Lab parameters so evaluators run with or without internet automatically.
9. **Teacher slot assists:** Provide slot-specific UI text for API key launchers, Transformer Lab profiles, and Ollama endpoints. When `connection_type` is `api`, the UI surfaces a button that lands on Transformer Lab key management; when `ollama`, the form collects `http://host:port` and fetches the latest model list on demand.
10. **Dashboard deltas:** Visualization now includes the “Search vs Static Delta” pane; continue refining it with live Grok runs using `scripts/training/search_vs_static_runner.py` and `data/raw/live_bench_50.jsonl`.

## 4. Considerations
- **API costs:** Grok search invocations incur usage fees—add caching to `data/processed/search_cache.jsonl`.
- **Latency:** Search adds latency; surface metrics in dashboards to compare search-enabled vs static evaluators.
- **Reliability:** Provide graceful fallbacks and retry logic when Grok API or internet connectivity is unavailable.
- **Prompting:** Iteratively refine system prompts and search templates to maximize factual retrieval quality.
- **Offline-first compatibility:** When `enable_internet_teachers` is false or a request fails, the runner automatically compares against `data/examples/offline_reference.jsonl` (or user overrides) so evaluations never block.
