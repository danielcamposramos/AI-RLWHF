# Scripts

Automation utilities for data processing, evaluation, and orchestration. Favor idempotent CLI interfaces.

## Planned Modules
- `data_pipeline/`: Prompt loaders, dataset provenance stampers, and rubric exporters shared across plugins.
- `training/`: GRPO launchers, Unsloth Standby bootstrap helpers, and teacher-student supervision loops.
- `connectors/`: Lightweight clients for Transformer Lab SDK, Ollama REST, vLLM, and other inference backends to keep plugin logic declarative.
