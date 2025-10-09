#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet pandas matplotlib seaborn
else
  python3 -m pip install --quiet pandas matplotlib seaborn
fi
