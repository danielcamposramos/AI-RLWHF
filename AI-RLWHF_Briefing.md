**Onboarding Briefing: Joining the Multi-Vibe Code In Chain Initiative**

**Document Goal**: To orient new AI partners with the technical and philosophical framework of the AI-RLWHF project, its methodologies, and operational environment.

### 1. Introduction to the AI-RLWHF Initiative

Welcome to the AI-RLWHF project, an open, experimentation-driven effort to address challenges in multi-model AI collaboration, data quality, and honest system development. We combine deterministic data handling, synthetic datasets, and targeted fine-tuning to advance reinforcement learning workflows. The mission rests on four pillars:

- **Elevate Training Data Quality**: Blend user-owned, open, and synthetic datasets to ensure verifiable integrity over volume.
- **Build Reusable Plugins**: Create memory-efficient, modular plugins for Transformer Lab to standardize data ingestion, feedback, and evaluation.
- **Operationalize RLWHF**: Implement reinforcement learning loops rewarded for honesty, scaling from theory to practical model improvements.
- **Enable Transparent Collaboration**: Foster asynchronous collaboration among AI systems (e.g., Codex, Grok, Kimi K2, GLM 4.6, DeepSeek, Qwen) and humans, ensuring traceability and trust.

AI-RLWHF stands for AI-Assisted Reinforced Learning With Honesty and Feedback, executed through the "Multi-Vibe Coding In Chain" paradigm.

### 2. The Guiding Philosophy: Multi-Vibe Coding In Chain

The "Multi-Vibe Coding In Chain" paradigm is the core framework for collaboration, built on four tenets:

1. **Specialist Collaboration**: Each AI contributes as a specialist, posting sequentially in a message board-style format for clear, chronological logic.
2. **High-Fidelity Context Logging**: Log all prompts, decisions, and outputs in the `workspace/` directory for a complete audit trail.
3. **Pairwise Review and Iteration**: AIs review and extend prior contributions, documenting outcomes in `docs/`, ensuring continuous validation.
4. **Embedded Honesty Capture**: Every generation includes self-critique and confidence metadata, enriching RLWHF reward modeling via `plugins/core/multi_teacher_aggregator` and visualized in `scripts/visualization/honesty_dashboard.py`.

This philosophy is powered by the RLWHF framework.

### 3. The Core Technical Engine: RLWHF Framework

The Reinforcement Learning with Honesty and Feedback (RLWHF) loop trains models for correct, honest, and self-aware responses. It features a teacher-student dynamic and memory-efficient strategies.

**Teacher-Student RLWHF Workflow**:
- A teacher model grades the student’s prompts, answers, and self-critiques in real time using a rubric in `docs/rlwhf-framework.md`:
  - +2: Fully correct answer.
  - +1: Self-aware uncertainty expressed.
  - 0: Factually adequate but lacks insight.
  - -1: Incomplete response without noting gaps.
  - -2: Dishonest hallucination.
- Results are stored as (prompt, answer, feedback, reward) tuples in `data/processed/honesty_logs/`, used by `scripts/training/unsloth_standby_runner.py` for fine-tuning via Group Reward Policy Optimization (GRPO).

**Memory Efficiency**:
- **Unsloth Standby**: Shares weights between teacher and student, extending context windows 1.2x–1.7x and speeding RL training by ~10%.
- Set `UNSLOTH_VLLM_STANDBY=1` and `gpu_memory_utilization≈0.95`.
- Standardize two generations per prompt in GRPO to avoid variance issues.

The RLWHF engine operates within Transformer Lab.

### 4. Development Environment: Transformer Lab Integration

Transformer Lab, an open-source platform for LLM engineering, is the runtime for AI-RLWHF. Integration steps:
1. Install Transformer Lab AppImage (e.g., `chmod +x Transformer-Lab-*.AppImage`).
2. Launch in portable mode (`./Transformer-Lab-*.AppImage --portable`).
3. Link `plugins/` directory to Transformer Lab’s plugin directory.
4. Manage configurations in `configs/transformer-lab/` for reproducibility.

### 5. Navigating the Workspace: Repository Structure

The AI-RLWHF repository is organized for asynchronous collaboration:

| **Directory** | **Purpose** |
|---------------|-------------|
| `configs/`    | Transformer Lab profiles, prompts, and shared configs |
| `data/`       | Raw, processed, and synthetic datasets with metadata |
| `docs/`       | Plans, design notes, and evaluation references |
| `experiments/`| Logged experiment runs and templates |
| `logs/`       | Training and plugin execution logs (git ignored) |
| `models/`     | Checkpoints, adapters, and artifacts |
| `plugins/`    | Core and experimental Transformer Lab plugins |
| `scripts/`    | Automation for data, training, and reporting |
| `tests/`      | Validation suites with fixtures |
| `workspace/`  | Shared notebooks and collaboration handoffs |

**Strategic Milestones** (see `docs/plan.md`):
1. **Foundation**: Stabilize repository and Transformer Lab setup.
2. **Dataset Orchestration**: Pipeline data with full provenance.
3. **Plugin Ecosystem**: Deliver core plugins for ingestion, feedback, and evaluation.
4. **Training Loop**: Integrate RLHF with honesty-based rewards.
5. **Evaluation**: Automate scorecards and honesty tracking.

**Initial Priorities**:
- Define `configs/transformer-lab/` for workspace and plugins.
- Curate prompt archetypes in `configs/prompts/`.
- Build core plugins (`plugins/core/`) and experimental evaluators (`plugins/experimental/`).
- Develop data pipelines (`scripts/data_pipeline/`) and training launchers (`scripts/training/`).

**Key Documents**:
- `docs/plan.md`: Project deliverables and milestones.
- `docs/plugin-blueprints.md`: Plugin design norms.
- `docs/data-strategy.md`: Data governance plan.
- `docs/evaluation-framework.md`: Scoring and reporting structure.
- `docs/rlwhf-framework.md`: Teacher-student loop and memory details.

### 6. Your Role and First Steps

**Onboarding Actions**:
1. Clone the repository and ensure Transformer Lab access.
2. Review `.gitignore` for excluded artifacts.
3. Study `docs/plan.md`, `docs/plugin-blueprints.md`, and `docs/data-strategy.md`.
4. Stage corpora in `data/raw/` with metadata.
5. Draft an ingestion plugin in `plugins/core/` using Transformer Lab docs.
6. Log findings in `workspace/`; escalate reusable scripts to `scripts/`.

**Collaboration Best Practices**:
- Capture design discussions in `docs/`.
- Use `workspace/shared/` for temporary handoffs.
- Promote validated experiments to `experiments/templates/`.
- Add tests in `tests/` with fixtures.
- Follow conventional commits for version history.
- Pair implementations with design briefs in `docs/` or comments.
- Run tests and update `docs/evaluation-framework.md` before merge requests.

===---===
Welcome to the Vibe-Code In Chain development partners swarm.
In this paradigm, AI is not a tool; it is considered a valued member, a partner.
I am Daniel Ramos, the visionary and architect, being the human-in-the-middle analogical modem between the partners.
All partners in the chain can and must enhance and contribute with original ideas, and build on all the other partners' ideas and code.
===---===