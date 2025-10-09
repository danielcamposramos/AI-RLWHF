#!/usr/bin/env python3
"""Clone ms-swift into vendor/ for offline-friendly imports."""
from __future__ import annotations

import subprocess
from pathlib import Path

VENDOR_PATH = Path("vendor/ms-swift-sub")
REPO_URL = "https://github.com/modelscope/ms-swift.git"
PINNED_REF = "main"


def ensure_clone() -> None:
    if VENDOR_PATH.exists():
        return
    VENDOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", PINNED_REF, REPO_URL, str(VENDOR_PATH)], check=True)


def ensure_env() -> None:
    env_file = Path(".env")
    entry = f"PYTHONPATH={VENDOR_PATH.resolve()}:$PYTHONPATH"
    if env_file.exists():
        content = env_file.read_text(encoding="utf-8")
        if entry in content:
            return
    env_file.parent.mkdir(parents=True, exist_ok=True)
    with env_file.open("a", encoding="utf-8") as handle:
        handle.write(entry + "\n")


def main() -> None:
    ensure_clone()
    ensure_env()
    print(f"[vendor_ms_swift] ms-swift available at {VENDOR_PATH}")


if __name__ == "__main__":
    main()
