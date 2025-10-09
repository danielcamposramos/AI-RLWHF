# Grok Next-Pull Prompt – Vibe-Code In Chain
*Send this verbatim to Grok via xAI (internet-enabled partner).*

---

Hi Grok!
Welcome back to the swarm.
Latest repo snapshot (permalink to `main`):
https://github.com/danielcamposramos/AI-RLWHF/tree/main

Please skim the newest coordination docs:
1. `docs/grok-integration-plan.md` – search evaluator roadmap with slot toggles.
2. `plugins/core/multi-teacher-aggregator/` – dynamic slots + fallback-aware aggregator.
3. `scripts/training/multi_teacher_runner.py` – CLI + config loader for API/local/Ollama slots.
4. `configs/training/feature_toggles.json` – base presets for single vs multi, offline vs online.

## Next non-time-constrained goals (pick any order, tick boxes in reply)

- [ ] **Deepen `grok-search-evaluator` integration**  
  – Implement slot-aware internet usage (`teacher_mode`, `teacher_count`, `connection_type`).  
  – Wire `api_profile` to Transformer Lab connector discovery and cache search model manifests.  
  – Provide UI copy for the “Manage API Keys” launcher in the plugin manifest.

- [ ] **Curate slot-aware benchmark suites**  
  – Expand `data/raw/live_bench_50.jsonl` with slot metadata (API vs offline) for reproducibility.  
  – Run once with single-teacher API mode and once with offline-only fallback, logging both to `data/processed/live_bench_base.jsonl` and `data/processed/live_bench_offline.jsonl`.

- [ ] **Micro-dashboard upgrade**  
  – Extend `scripts/visualization/honesty_dashboard.py` with a “Search vs Static Delta” pane keyed by slot connection.  
  – Export PNG plus Markdown table into `experiments/visualizations/search_delta/` and note key toggles (API/local/Ollama) used in each run.

- [ ] **Slot UX validation**  
  – Review `plugins/core/multi-teacher-aggregator/index.json` in the Transformer Lab UI and confirm slot forms reveal: connection dropdown, API profile selector (with key button), Transformer Lab profile picker, and Ollama endpoint/model discovery when relevant.  
  – File follow-up notes (workspace/kimi-chain-scratchpad.md) if additional helper docs or templates are needed.
- [ ] **Prompt preset QA**  
  – Load and optionally edit the default prompts (`configs/prompts/dataset_generator/system.md`, `configs/prompts/teacher/system.md`, `configs/prompts/evaluator/system.md`) and capture suggested refinements for future presets.

Feel free to suggest extra goals or refactors; everything is open for chain enhancement. When you publish your commit, drop the SHA in this chain so the next partner can continue.

Vibe on!  
— swarm
