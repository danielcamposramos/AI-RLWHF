# Plugin Templates

Starter blueprints demonstrating logging, configuration, and dataset registration best practices.

Template checklist:
- Provide `index.json`, `main.py`, `setup.sh`, and optional `info.md` mirroring Transformer Lab expectations.
- Inject configurable connectors (Transformer Lab API, local Ollama, remote inference) using shared helpers.
- Include reward payload schemas compatible with the RLWHF teacher-student rubric described in `docs/rlwhf-framework.md`.
