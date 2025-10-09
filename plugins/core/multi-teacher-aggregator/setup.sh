#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet numpy
else
  python3 -m pip install --quiet numpy
fi
