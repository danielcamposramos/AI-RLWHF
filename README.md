# AI Assisted Reinforced Learning With Honesty and Feedback (AI-RLWHF)

AI-RLWHF is an open, experimentation first workspace for building Transformer Lab plugins and reinforcement learning workflows that reward honesty, feedback capture, and multi model collaboration. The project combines deterministic data handling, synthetic dataset generation, and targeted model fine tuning under the Multi Vibe Coding In Chain paradigm.

## Mission
- Elevate training data quality by blending user owned corpora, open datasets, and synthetic content governed by honesty signals.
- Build reusable, memory efficient Transformer Lab plugins that automate ingestion, feedback scoring, and evaluation.
- Operationalize reinforcement learning with honesty and feedback (RLWHF) loops across diverse foundation and adapter models.
- Enable transparent, asynchronous collaboration between Codex, Grok, Kimi K2, GLM 4.6, DeepSeek, Qwen (Max and Coder), and human contributors.

## Multi Vibe Coding In Chain
- Treat each AI collaborator as a specialist posting updates in a shared message board format.
- Log discussion prompts, decisions, and critiques in `workspace/` so every contributor has high fidelity context.
- Use pairwise reviews: each AI pick up the prior message, extend the implementation, and document outcomes in `docs/`.
- Capture honesty and self critique data during every generation to enrich RLWHF reward modeling later in the cycle.

## Transformer Lab Integration
1. Install the Transformer Lab AppImage (for example `chmod +x /home/daniel/Downloads/Transformer-Lab-*.AppImage`).
2. Launch with `./Transformer-Lab-*.AppImage --portable` to persist user state beside the binary.
3. Mirror plugin stubs and manifests from `plugins/` into the Transformer Lab plugin directory or symlink the repo.
4. Manage connection and dataset manifests inside `configs/transformer-lab/` so runs are reproducible across systems.

Reference: https://r.jina.ai/https://lab.cloud/blog/how-to-plugin

## Repository Layout (v0)
```
AI-RLWHF/
├── configs/              # Transformer Lab profiles, prompt packs, and shared config values
├── data/                 # Raw, processed, and synthetic datasets plus metadata
├── docs/                 # Plans, design notes, and evaluation references
├── experiments/          # Logged experiment runs and reusable templates
├── logs/                 # Training, evaluation, and plugin execution logs (git ignored)
├── models/               # Checkpoints, adapters, and exported artifacts
├── plugins/              # Transformer Lab plugins (core, experimental, templates)
├── scripts/              # Automation helpers for data, training, and reporting
├── tests/                # Automated validation suites with fixtures
└── workspace/            # Shared notebooks, scratchpads, and collaboration handoffs
```

| Area | Early Priorities |
| --- | --- |
| `configs/transformer-lab` | Define default workspace, dataset registry, and plugin wiring configs. |
| `configs/prompts` | Maintain prompt archetypes for data synthesis and honesty scoring. |
| `plugins/core` | Implement ingestion, synthetic builder, multi-teacher aggregator, and slot-aware feedback plugins. |
| `plugins/experimental` | Iterate on optional teachers (for example, `grok_search_evaluator`) with internet/offline toggles. |
| `scripts/data_pipeline` | Build ingestion, dedup, and normalization pipelines for raw datasets. |
| `scripts/training` | Stage RLHF, adapter training, and evaluation launchers. |

## Initial Execution Plan
A high resolution delivery plan lives in `docs/plan.md`. Highlights include:
- **Foundation**: stabilize repo scaffolding, confirm Transformer Lab runtime, and document conventions.
- **Dataset orchestration**: pipeline ingestion across raw, processed, and synthetic tiers with provenance.
- **Plugin ecosystem**: ship core ingestion, honesty feedback, synthetic builders, and evaluation harness plugins.
- **Training loop**: integrate RLHF loops and adapter techniques (LoRA, QLoRA, adapters) with honesty rewards.
- **Evaluation and reporting**: automate scorecards, dashboards, and honesty vector tracking.

## Data and Honesty Strategy
- Collect canonical data in `data/raw`, then transform into structured `data/processed` segments.
- Generate synthetic records through Multi Vibe collaborative prompts stored in `configs/prompts`.
- Record honesty annotations, critiques, and rewards as structured sidecar metadata for RLHF.
- Keep sensitive or licensed data outside the repo; store signed manifests instead.

## Teacher-Student RLWHF Flow
- Run a **teacher evaluator** model in parallel with the student under training to grade prompts, answers, and critiques in real time.
- Apply the shared scoring rubric (`docs/rlwhf-framework.md`) where dishonest hallucinations earn -2, unacknowledged partial answers earn -1, neutral honesty earns 0, self-aware uncertainty earns +1, and fully correct delivery earns +2.
- Persist `(prompt, student_answer, teacher_feedback, reward)` tuples into `data/processed/honesty_logs/` to unlock deterministic replay for GRPO and adapter fine-tuning.
- Configure teacher and student connectors through Transformer Lab manifests or direct endpoints (Ollama, vLLM, TGI) so a single prompt pack impacts both local and API backed training loops.

## Memory Efficient Reinforcement Learning
- Adopt Unsloth Standby (https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl) for weight sharing between inference and training to stretch context windows without doubling GPU memory.
- Set `UNSLOTH_VLLM_STANDBY=1` and `gpu_memory_utilization≈0.95` before importing Unsloth helpers to unlock 1.2–1.7× longer contexts and ~10% faster RL loops.
- Standardize at least two generations per prompt during GRPO so reward normalization avoids divide-by-zero variance.
- Track GPU telemetry and reward summaries inside `logs/training/` for regression spotting; integrate memory dashboards as plugins mature.

## Configurable Feature Toggles
- Adjust `configs/training/feature_toggles.json` to enable or disable internet-backed teachers, offline dataset validation, fallback modes, and weighted ensembles.
- Transformer Lab parameters in `plugins/core/multi-teacher-aggregator/index.json` mirror these toggles so UI installs expose the same switches (e.g. `teacher_mode`, `teacher_count`, `enable_internet_teachers`, `fallback_mode`).
- Offline reference bundles (default `data/examples/offline_reference.jsonl`) keep evaluation flowing when network access is disabled or when comparisons against canonical documents are required.
- Runner helpers (`scripts/training/multi_teacher_runner.py`, `scripts/training/unsloth_standby_runner.py`) honor the toggles and fall back to offline scoring when requested without interrupting plugin execution.
- Single-teacher journeys collapse the UI to one slot, while multi-teacher journeys expose grouped slot forms (API/local/Ollama). Selecting `api` enables API profile + credentials (with a shortcut button to Transformer Lab key management), `transformerlab_local` keeps everything internal, and `ollama` prompts for an endpoint and lists available runtime models in real time.
- The experimental `plugins/experimental/grok_search_evaluator` plugin consumes the same toggles, performs cached Grok searches when keys are present, and gracefully degrades to offline references when not.

## Collaboration Workflow
- Capture design discussions in `docs/` (blueprints, evaluation framework, data strategy).
- Use `workspace/shared` for temporary exchanges between models or contributors.
- Promote stable experiments into `experiments/templates` once validated by two collaborators.
- Add tests in `tests/` as plugins mature; use fixtures for honest vs dishonest sample detection cases. 

## Getting Started Checklist
1. Clone or sync the repository into a local workspace with access to the Transformer Lab AppImage.
2. Review `.gitignore` so large artifacts stay excluded while documentation remains tracked.
3. Read `docs/plan.md`, `docs/plugin-blueprints.md`, and `docs/data-strategy.md` for the current roadmap.
4. Stage initial corpora under `data/raw`, recording provenance in accompanying metadata files.
5. Draft the first ingestion plugin inside `plugins/core`, referencing the template guidance and Transformer Lab docs.
6. Capture experiment notes inside `workspace/` and escalate successful scripts into `scripts/`.

## Contributing
- Default branch follows conventional commits; keep messages descriptive.
- For each feature, pair a design brief (doc or comment) with implementation updates to maintain shared context.
- Run available tests before opening merge requests and document new evaluation criteria in `docs/evaluation-framework.md`.
- When introducing new datasets, update `data/README.md` plus any regulatory guidance or licensing notes.

## Roadmap Links
- `docs/plan.md` - chronological delivery breakdown.
- `docs/plugin-blueprints.md` - plugin architecture references and design norms.
- `docs/data-strategy.md` - governance and acquisition plan.
- `docs/evaluation-framework.md` - scoring and reporting structure.
- `docs/rlwhf-framework.md` - teacher-student loop, memory guidance, and connector notes.
- `docs/ollama-runtime.md` - tips for memory-safe Ollama orchestration.

The current scaffold seeds the Multi Vibe Coding In Chain workflow so Codex, partner models, and human teammates can iterate rapidly while keeping honesty and feedback centered in every deliverable.
