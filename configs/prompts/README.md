# Prompt Presets

Reusable prompt and instruction templates supporting Multi Vibe Coding In Chain iterations and synthetic data programs.

## Directory Guidance
- `teacher/`: Canonical evaluation prompts, honesty rubrics, and critique scaffolds reused across Transformer Lab and local inference connectors.
- `student/`: Instruction packs, role/GUID priming, and curriculum tasks guiding the training target during RLWHF loops.
- `rubrics.yml`: Shared reward scale (-2 to +2) with textual descriptions so plugins and scripts assign identical incentives.
- `ensembles/`: Multi-model prompt cascades outlining how Codex, Grok, Qwen, and other collaborators exchange updates in chain-of-thought loops.

## Usage Patterns
- Loader utilities (to be added in `scripts/data_pipeline/prompts.py`) read this directory and expose prompts through a single API regardless of runtime (Transformer Lab SDK, direct REST, or Ollama).
- Version prompts with semantic tags (e.g., `teacher_v1`, `student_curriculum_alpha`) to support deterministic replays.
- Keep honesty feedback exemplars synchronized with `docs/rlwhf-framework.md` so rubric adjustments automatically cascade through dataset generation and RL training stages.
