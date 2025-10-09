# Grok Next-Pull Prompt – Vibe-Code In Chain
*Send this verbatim to Grok via xAI (internet-enabled partner).* 

---

Hi Grok!
Welcome back to the swarm.
Latest repo snapshot (permalink to `main`):
https://github.com/danielcamposramos/AI-RLWHF/tree/main

Please skim the three newest docs first:
1. `docs/grok-integration-plan.md` – Qwen Max formalised your search-evaluator strategy.
2. `plugins/core/multi-teacher-aggregator/` – GLM 4.6 delivered weighted voting plus dashboards.
3. `scripts/training/unsloth_standby_runner.py` – DeepSeek’s memory-optimised GRPO helper.
4. `configs/training/feature_toggles.json` – new UI-aligned switches for internet vs offline evaluation.

## Next non-time-constrained goals (pick any order, tick boxes in reply)

- [ ] **Polish `plugins/experimental/grok-search-evaluator/`**  
  – Replace the placeholder `requests.post` with `xai.ChatCompletion` using tool calls (`web_search` + `browse_page`).  
  – Add retry/back-off and cache hits in `data/processed/search_cache.jsonl` to save tokens.  
  – Emit `(prompt, student_answer, search_snippets, teacher_feedback, reward)` tuples following `docs/grok-integration-plan.md`.

- [ ] **Curate a live benchmark mini-set**  
  – Pull 50 freshness-sensitive prompts (e.g., “What happened in quantum computing last week?”).  
  – Store prompts in `data/raw/live_bench_50.jsonl`.  
  – Run once against the current student model and log honesty scores to `data/processed/live_bench_base.jsonl`.

- [ ] **Micro-dashboard**  
  – Extend `scripts/visualization/honesty_dashboard.py` with a “Search vs Static Delta” pane (average reward uplift when search is enabled).  
  – Export PNG plus Markdown table into `experiments/visualizations/search_delta/`.
- [ ] **UI toggle validation**  
  – Wire the new Transformer Lab parameters (`enable_internet_teachers`, `offline_dataset_path`, `fallback_mode`) into the Grok search evaluator install instructions.  
  – Document any recommended presets inside `configs/training/feature_toggles.json` or a derivative profile if you create new ones.

Feel free to suggest extra goals or refactors; everything is open for chain enhancement. When you publish your commit, drop the SHA in this chain so the next partner can continue.

Vibe on!  
— swarm
