# Experiments

Contains tracked experiment runs, hyperparameter sweeps, and reinforcement learning studies. Use structured naming `YYYYMMDD_model_objective`.

## Tracking Guidance
- Record GPU telemetry (peak memory, standby status, context length) for each RLWHF training sweep.
- Store reward distribution summaries (counts per -2…+2) alongside metrics for easy honesty trend analysis.
- Reference `docs/rlwhf-framework.md` for required columns when logging teacher-student tuples.
- Mirror search-enabled vs static runs by exporting the visualization bundle in `experiments/visualizations/` (see `search_delta.png` + `search_delta.md`).
