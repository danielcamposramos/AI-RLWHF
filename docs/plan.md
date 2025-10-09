# AI-RLWHF Initial Delivery Plan

## Phase 0 - Foundation (Week 0-1)
- Establish repository scaffolding, CI hooks, and coding standards.
- Produce baseline documentation such as README, plugin design briefs, and prompt guidelines.
- Map existing corpora and user supplied knowledge bases into `data/raw`.
- Verify Transformer Lab AppImage runtime and capture invocation helpers.

## Phase 1 - Dataset Orchestration (Week 1-3)
- Author data ingestion scripts for textual, multimodal, and structured collections.
- Implement provenance tracking and schema normalization into `data/processed`.
- Stand up Transformer Lab dataset plugins for corpus chunking, semantic search, and feedback capture.
- Pilot synthetic data recipes using Multi Vibe model ensemble prompts.
- Introduce teacher/student prompt registries in `configs/prompts/teacher` and `configs/prompts/student` for consistent rubric alignment.
- Prototype slot-aware configuration defaults (`configs/training/feature_toggles.json`) so UI toggles for single vs multi teachers ship early.

## Phase 2 - Plugin Ecosystem (Week 2-4)
- Deliver `plugins/core` adapters covering ingestion, augmentation, and labeling workflows.
- Create evaluation aware plugins that score honesty, novelty, and coverage metrics.
- Package reusable plugin templates within `plugins/templates`.
- Provide quick start notebooks or scripts demonstrating plugin registration.
- Implement teacher evaluator and reward aggregation plugins that can switch between Transformer Lab connectors, local Ollama endpoints, or remote inference APIs via configuration only.
- Surface slot-driven UI selectors (connection type, API profile, Ollama endpoint) inside plugin manifests and ensure cached model listings are optional per toggle.
- Surface UI-exposed toggles (internet vs offline, fallback modes) within plugin manifests so partners can adjust behavior without editing code.

## Phase 3 - Training and RLHF Loop (Week 4-6)
- Integrate reinforcement learning harness that loops through honesty feedback scoring.
- Explore adapter efficient training (LoRA or QLoRA) staged within `models/adapters`.
- Track metrics and insights in `experiments/` and route logs into `logs/training`.
- Validate outputs with baseline evaluations defined in `docs/evaluation-framework.md`.
- Prototype Unsloth Standby enabled GRPO runners and capture benchmark deltas against baseline GPU utilization.
- Persist structured honesty tuples `(prompt, answer, critique, reward)` in `data/processed/honesty_logs` and surface dashboards in `experiments/`.

## Phase 4 - Evaluation and Reporting (Week 5-7)
- Automate leaderboard summaries for experiments and honesty vectors.
- Build reporting dashboards or markdown digests for stakeholders.
- Document lessons learned and update roadmap for future expansion.
- Publish a memory efficiency report contrasting Unsloth Standby, vanilla vLLM, and Transformer Lab native loops.

### Kimi K2 Pulse – 2025-10-09
- `docs/prompt-for-grok.md` captures the next Grok kickoff prompt with non-time-constrained goals.
- `workspace/kimi-chain-scratchpad.md` introduces the emoji cue wall for async coordination.
- `scripts/utils/chain_logger.py` enables lightweight JSONL partner telemetry (see `.gitignore` for streaming output hygiene).

### Codex Configurable UI Pulse – 2025-10-09
- Added `configs/training/feature_toggles.json` and matching plugin parameters so Transformer Lab installs expose internet/offline switches.
- Augmented training helpers, aggregators, and telemetry scripts to respect toggles without interrupting workflows when network access is disabled.
- Seeded sample offline references (`data/examples/offline_reference.jsonl`) for baseline comparisons and automated offline scoring.
- Delivered slot metadata (API/local/Ollama) so single-teacher and multi-teacher panels auto-expand with connection-aware selectors and API-key shortcuts.

### Codex Search/Evaluator Pulse – 2025-10-09
- Delivered `plugins/experimental/grok_search_evaluator` with search caching, offline fallback logic, and UI-editable system prompts.
- Published canonical prompts for dataset generation, teacher evaluation, and post-run analytics (`configs/prompts/*/system.md`).
- Extended dashboards with search-vs-static deltas and documented Ollama runtime procedures (`docs/ollama-runtime.md`).
