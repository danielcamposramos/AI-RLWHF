# Core Plugins

Battle tested plugins maintained by the team for mission critical data and training flows.

## Expected Packages
- `ingestion/`: Deterministic chunkers and metadata enrichers aligning to Transformer Lab dataset manifests.
- `teacher/`: Evaluators implementing the -2…+2 honesty rubric and exporting structured critiques.
- `reward/`: Aggregators harmonizing feedback from multiple teachers or critique engines.
- `synthetic/`: Builders that spin Multi Vibe Coding In Chain prompts into synthetic datasets with provenance tags.
- `multi-teacher-aggregator/`: Slot-aware aggregator plugin powering UI toggles for single/multiple teachers across API, Transformer Lab local, and Ollama connections.
