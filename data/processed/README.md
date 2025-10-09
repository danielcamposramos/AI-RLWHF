# Processed Data

Normalized, tokenized, and feature enriched datasets derived from raw assets. Scripts in `scripts/data_pipeline` materialize this layer.

### Subdirectories
- `honesty_logs/`: Teacher-student tuples with prompt ids, responses, critiques, and rubric-aligned reward signals.
- `curricula/`: Prepared lesson plans and evaluation packs for batching during RLWHF loops.
