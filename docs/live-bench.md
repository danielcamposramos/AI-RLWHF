# Live Bench Playbook

`data/raw/live_bench_50.jsonl` seeds prompts for freshness-sensitive evaluations. Populate answers via student runs, then invoke `scripts/training/search_vs_static_runner.py` to log search-enabled vs offline honesty scores.

Example:
```
python scripts/training/search_vs_static_runner.py \
  --prompts data/raw/live_bench_50.jsonl \
  --answers data/processed/student_answers.txt \
  --output experiments/visualizations/search_delta \
  --config configs/training/feature_toggles.json
```

Results are written to `search_enabled.json`, `search_disabled.json`, and `search_vs_static_summary.json`, which feed the dashboard delta plots.
