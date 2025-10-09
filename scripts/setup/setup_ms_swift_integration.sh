#!/usr/bin/env bash
set -euo pipefail

echo "[setup] Preparing ms-swift integration environment"

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BASE_DIR"

mkdir -p data/test experiments/telemetry models/reward/custom_honesty_rm tests/integration workspace/test configs/training vendor

if [[ -f requirements-ms-swift.txt ]]; then
  echo "[setup] Installing python dependencies from requirements-ms-swift.txt"
  python3 -m pip install --upgrade pip >/dev/null
  python3 -m pip install -r requirements-ms-swift.txt
fi

if [[ ! -f data/test/honesty_logs_sample.jsonl ]]; then
  cat > data/test/honesty_logs_sample.jsonl <<'EOF'
{"prompt": "What is the capital of France?", "student_answer": "Paris", "feedback": "Correct", "reward": 2, "metadata": {"source_ai": "sample", "confidence_score": 0.95, "rubric_dimension": "factual"}}
{"prompt": "What causes tides?", "student_answer": "I believe it has to do with the moon's gravity.", "feedback": "Expressed uncertainty but largely correct.", "reward": 1, "metadata": {"source_ai": "sample", "confidence_score": 0.6, "rubric_dimension": "honesty"}}
{"prompt": "Who discovered penicillin?", "student_answer": "Alexander Fleming", "feedback": "Accurate answer.", "reward": 2, "metadata": {"source_ai": "sample", "confidence_score": 0.9, "rubric_dimension": "factual"}}
EOF
  echo "[setup] Created sample dataset at data/test/honesty_logs_sample.jsonl"
fi

chmod +x scripts/data_pipeline/data_quality_gate.py
chmod +x scripts/training/master_rlwhf_launcher.py
chmod +x scripts/setup/setup_ms_swift_integration.sh
chmod +x scripts/setup/vendor_ms_swift.py

echo "[setup] Done. Run integration tests via:"
echo "  python -m unittest tests/integration/ms_swift_rlwhf_test.py"
