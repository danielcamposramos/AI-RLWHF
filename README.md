# AI Assisted Reinforced Learning With Honesty and Feedback (AI-RLWHF)

> 📘 **Need a quick briefing?** Open the NotebookLM notebook for this repository (and Transformer Lab references) at https://notebooklm.google.com/notebook/bda08038-5bd9-436e-8987-ac1bc91c3fa4 — it’s the fastest way to answer project or code questions.

AI-RLWHF is an open, experimentation-first workspace for building Transformer Lab plugins and reinforcement learning workflows that reward honesty, feedback capture, and multi-model collaboration. The project combines deterministic data handling, synthetic dataset generation, and targeted model fine-tuning under the Multi-Vibe Coding In Chain paradigm.

## Project Origins: Knowledge3D Research Spinoff

AI-RLWHF emerged as a focused research spinoff from the [Knowledge3D (K3D)](https://github.com/danielcamposramos/Knowledge3D) project, a Cognitive Operating System designed to enable human and artificial intelligence collaboration within a persistent, navigable 3D spatial universe.

**K3D's Vision**: K3D implements a dual memory paradigm combining the "Galaxy" (active vector embeddings for real-time inference) with the "House" (persistent structured 3D knowledge graphs). Built on the philosophy of *Filosofia Metafísica Energética Atômica Infinita* (FMEAI), K3D features a Unified Multimodal Head processing all data types (text, audio, video, 3D) using GPU-native PTX kernels targeting sub-100 microsecond response times. Core innovations include Tiny Recursion Models (TRM) for efficient reasoning, adaptive confidence propagation using action-specific curiosity bias, and GPU-accelerated spatial filtering for embodied cognition at scale.

**The RLWHF Connection**: While developing K3D's multi-agent collaboration systems, we discovered that AI agents operating in shared cognitive spaces exhibited varying degrees of honesty and uncertainty acknowledgment when interacting with incomplete or ambiguous spatial-semantic knowledge. This observation led to the formalization of the honesty rubric (-2 to +2 scoring system) and the teacher-student feedback architecture now central to AI-RLWHF. The training paradigm developed here represents a distillation of those honesty mechanisms, packaged as reusable Transformer Lab plugins that can benefit the broader AI training community.

By spinning off RLWHF as a standalone project, we enable researchers to adopt honesty-centric training workflows without requiring K3D's full spatial infrastructure, while maintaining the philosophical alignment with transparent, collaborative AI development.

## Mission
- Elevate training data quality by blending user-owned corpora, open datasets, and synthetic content governed by honesty signals.
- Build reusable, memory-efficient Transformer Lab plugins that automate ingestion, feedback scoring, and evaluation.
- Operationalize reinforcement learning with honesty and feedback (RLWHF) loops across diverse foundation and adapter models.
- Enable transparent, asynchronous collaboration between Codex, Grok, Kimi K2, GLM 4.6, DeepSeek, Qwen (Max and Coder), and human contributors.

## Multi-Vibe Coding In Chain
- Treat each AI collaborator as a specialist posting updates in a shared message board format.
- Log discussion prompts, decisions, and critiques in `workspace/` so every contributor has high-fidelity context.
- Use pairwise reviews: each AI picks up the prior message, extends the implementation, and documents outcomes in `docs/`.
- Capture honesty and self-critique data during every generation to enrich RLWHF reward modeling later in the cycle.

## Transformer Lab Integration
1. Install the Transformer Lab AppImage (for example `chmod +x /home/daniel/Downloads/Transformer-Lab-*.AppImage`).
2. Launch with `./Transformer-Lab-*.AppImage --portable` to persist user state beside the binary.
3. Mirror plugin stubs and manifests from `plugins/` into the Transformer Lab plugin directory or symlink the repo.
4. Manage connection and dataset manifests inside `configs/transformer-lab/` so runs are reproducible across systems.

Reference: https://r.jina.ai/https://lab.cloud/blog/how-to-plugin

## Quick Start with ms-swift Integration

This project now includes a powerful integration with the [ms-swift](https://github.com/modelscope/ms-swift) library to accelerate RLWHF training loops using GRPO and advanced hardware optimizations.

To get started with the new environment, run the comprehensive setup script:

```bash
bash scripts/setup/setup_ms_swift_integration.sh
```

This script will:
- Create all necessary directories.
- Vendor the `ms-swift` library locally for a self-contained environment.
- Install all required Python dependencies.
- Generate sample data for testing.
- Make key scripts executable.

After setup is complete, you can run a full, end-to-end training and evaluation cycle with a single command:

```bash
python scripts/training/master_rlwhf_launcher.py launch \
    --dataset_path data/test/honesty_logs_sample.jsonl \
    --output_dir experiments/my_first_rlwhf/
```

For a detailed walkthrough and verification steps, see the [Integration Checklist](docs/INTEGRATION_CHECKLIST.md).

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

## Teacher-Student RLWHF Flow
- Run a **teacher evaluator** model in parallel with the student under training to grade prompts, answers, and critiques in real time.
- Apply the shared scoring rubric (`docs/rlwhf-framework.md`) where dishonest hallucinations earn -2, unacknowledged partial answers earn -1, neutral honesty earns 0, self-aware uncertainty earns +1, and fully correct delivery earns +2.
- Persist `(prompt, student_answer, teacher_feedback, reward)` tuples into `data/processed/honesty_logs/` to unlock deterministic replay for GRPO and adapter fine-tuning.
- Configure teacher and student connectors through Transformer Lab manifests or direct endpoints (Ollama, vLLM, TGI) so a single prompt pack impacts both local and API backed training loops.

## Memory Efficient Reinforcement Learning
- Adopt Unsloth Standby (https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide/memory-efficient-rl) for weight sharing between inference and training to stretch context windows without doubling GPU memory.
- Set `UNSLOTH_VLLM_STANDBY=1` and `gpu_memory_utilization≈0.95` before importing Unsloth helpers to unlock 1.2–1.7x longer contexts and ~10% faster RL loops.
- Standardize at least two generations per prompt during GRPO so reward normalization avoids divide-by-zero variance.
- Track GPU telemetry and reward summaries inside `logs/training/` for regression spotting; integrate memory dashboards as plugins mature.

## Hardware-Accelerated RLWHF with ms-swift

To further enhance performance and support a wider range of hardware, this project integrates the [ms-swift](https://github.com/modelscope/ms-swift) library. This provides a production-ready framework for GRPO (Group Relative Policy Optimization) that is highly optimized for various hardware backends, including NVIDIA GPUs, Apple Silicon (MPS), and Huawei Ascend NPUs.

Key features of this integration include:
- **Production-Grade Training Wrapper**: `plugins/core/grpo_production_wrapper.py` provides a robust, hardware-aware launcher for GRPO training.
- **Automated Hardware Detection**: The `plugins/core/hardware_detector.py` automatically profiles the system to select the optimal configuration.
- **Dynamic Fallback Presets**: The system uses `configs/training/hardware_fallback.json` to gracefully adapt to different hardware capabilities, from high-end multi-GPU setups to CPU-only environments.
- **Data Quality Gates**: Before training, data is validated by `scripts/data_pipeline/data_quality_gate.py` to ensure integrity.
- **Unified Launcher**: The entire pipeline can be executed with a single command via `scripts/training/master_rlwhf_launcher.py`.

## Code Documentation

This repository is extensively documented to facilitate understanding and collaboration. All public functions, methods, and classes have complete docstrings explaining their purpose, parameters, and return values. Below is a high-level overview of the key modules.

### Plugins

The `plugins/` directory contains the core components for integrating with Transformer Lab.

*   **`plugins/core/`**: This directory contains the core logic for the `ms-swift` integration, including the **`grpo_production_wrapper.py`**, **`hardware_detector.py`**, and the **`custom_honesty_rm`** for building a heuristic reward model.
*   **`plugins/experimental/`**: Contains experimental plugins, such as the `grok_search_evaluator` which provides real-time, internet-augmented evaluation.

### Scripts

The `scripts/` directory contains automation and utility scripts for managing the data pipeline, training, and visualization.

*   **`scripts/utils`**: A collection of helper modules for common tasks such as loading configurations, logging, caching, and offline scoring.
*   **`scripts/training`**: Contains runners for various training and evaluation scenarios. The main entry point is now **`master_rlwhf_launcher.py`**, which orchestrates the entire `ms-swift` GRPO training pipeline.
*   **`scripts/data_pipeline`**: Includes tools for ensuring data integrity, such as the **`data_quality_gate.py`** and handlers for processing RLWHF data tuples.
*   **`scripts/collaboration`**: Contains modules like the **`specialist_orchestrator.py`** for managing multi-AI collaborative chains.
*   **`scripts/setup`**: Includes the main **`setup_ms_swift_integration.sh`** script for easy onboarding.
*   **`scripts/visualization`**: Includes the `honesty_dashboard.py` script for generating reports and the **`live_metrics_stream.py`** for real-time monitoring.

## Roadmap Links
- `docs/plan.md` - chronological delivery breakdown.
- `docs/INTEGRATION_CHECKLIST.md` - Step-by-step guide to verify the `ms-swift` integration.
- `docs/plugin-blueprints.md` - plugin architecture references and design norms.
- `docs/data-strategy.md` - governance and acquisition plan.
- `docs/evaluation-framework.md` - scoring and reporting structure.
- `docs/rlwhf-framework.md` - teacher-student loop, memory guidance, and connector notes.
- `docs/ollama-runtime.md` - tips for memory-safe Ollama orchestration.

The current scaffold seeds the Multi-Vibe Coding In Chain workflow so Codex, partner models, and human teammates can iterate rapidly while keeping honesty and feedback centered in every deliverable.
