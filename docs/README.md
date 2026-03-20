# Documentation

Centralized technical references, design logs, and collaboration briefs for AI-RLWHF contributors.

## Architecture

- `rlwhf-framework.md`: Teacher-student reinforcement loop, scoring rubric, connectors, and memory-efficient RL guidance.
- `CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md`: Three-axis contrastive learning with honesty as principal goal. Defines the decomposed teacher schema, InfoNCE loss formulation, triplet mining, embedding topology, and teacher-student cascade architecture.
- `CONTRASTIVE_HONESTY_IMPLEMENTATION_REPORT.md`: Professional implementation report. Component map, file inventory, design decisions, validation results, and next steps.
- `evaluation-framework.md`: Metrics, honesty scoring, and reporting definitions.

## Planning & Integration

- `plan.md`: Phase roadmap and milestone timing.
- `plugin-blueprints.md`: Transformer Lab plugin guidance and component catalogue.
- `data-strategy.md`: Acquisition and governance strategy.
- `ms-swift-integration.md`: Bridging plan for importing ms-swift GRPO/DPO tooling into AI-RLWHF pipelines with hardware-aware presets.
- `INTEGRATION_CHECKLIST.md`: Step-by-step guide for validating the ms-swift bridge locally and in CI.

## Connectors & Providers

- `grok-integration-plan.md`: Search-enhanced evaluator roadmap and Grok orchestration notes.
- `ollama-runtime.md`: Guidance for loading/unloading local Ollama models with context buffers.
- `prompt-for-grok.md`: Ready-to-send kickoff brief for Grok’s next contribution.

## Operations

- `pr-codex-ui-config.md`: Commit-ready summary for slot-aware toggles, Grok evaluator, and visualization upgrades.
- `live-bench.md`: Live benchmark workflow and search delta instructions.
