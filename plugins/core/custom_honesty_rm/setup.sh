#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --upgrade "datasets>=2.18.0" "transformers>=4.42.0" "accelerate>=0.32.0"
else
  python3 -m pip install --upgrade --quiet "datasets>=2.18.0" "transformers>=4.42.0" "accelerate>=0.32.0"
fi
