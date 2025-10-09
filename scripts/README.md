# Scripts

Automation utilities for data processing, evaluation, and orchestration. Favor idempotent CLI interfaces.

## Planned Modules
- `data_pipeline/`: Prompt loaders, dataset provenance stampers, and rubric exporters shared across plugins.
- `training/`: GRPO launchers, Unsloth Standby bootstrap helpers, and teacher-student supervision loops.
- `connectors/`: Lightweight clients for Transformer Lab SDK, Ollama REST, vLLM, and other inference backends to keep plugin logic declarative.
- `utils/`: Shared helpers (`chain_logger`, `config_loader`, `offline_scoring`, `search_cache`) consumed by evaluators and runners.
- `training/search_vs_static_runner.py`: Convenience CLI to compare Grok-enabled vs offline teacher runs and populate search delta dashboards.
