# Transformer Lab Profiles

Connection profiles, dataset manifests, and plugin wiring for the local Transformer Lab AppImage deployment.

## Recommended Files
- `profiles/*.yaml`: Define teacher/student connectors, dataset mounts, and plugin install lists.
- `prompts.yml`: Lightweight registry pointing prompts in `configs/prompts/` to Transformer Lab job identifiers.
- `env.sample`: References environment flags such as `UNSLOTH_VLLM_STANDBY=1` for memory efficient RL runners.
