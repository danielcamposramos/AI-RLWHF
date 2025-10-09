"""Utilities for loading JSON/YAML configuration bundles with defaults."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

try:  # YAML optional
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml optional
    yaml = None  # type: ignore


def load_config(path: Path | str, default: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Loads a JSON or YAML config file, applying defaults for missing values.

    This function can handle both JSON and YAML files. If the file does not exist
    or is empty, it returns the default dictionary.

    Args:
        path: The path to the configuration file.
        default: An optional dictionary of default values.

    Returns:
        A dictionary containing the loaded configuration.
    """
    path_obj = Path(path)
    data: MutableMapping[str, Any] = dict(default or {})
    if not path_obj.exists():
        return dict(data)
    text = path_obj.read_text(encoding="utf-8").strip()
    if not text:
        return dict(data)
    if path_obj.suffix.lower() in {".yaml", ".yml"} and yaml is not None:
        loaded = yaml.safe_load(text)
    else:
        loaded = json.loads(text)
    if isinstance(loaded, Mapping):
        data.update(loaded)
    return dict(data)


__all__ = ["load_config"]
