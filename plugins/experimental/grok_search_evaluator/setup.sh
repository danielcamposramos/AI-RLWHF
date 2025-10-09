#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet requests
else
  python3 -m pip install --quiet requests
fi
