"""Simple JSONL-based cache for search responses."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


class SearchCache:
    """A simple, file-based cache for search results using a JSONL file.

    This class provides a key-value store that persists to a file,
    allowing search results to be cached and reused across runs.

    Attributes:
        path: The path to the cache file.
    """
    def __init__(self, path: Path | str) -> None:
        """Initializes the SearchCache.

        Args:
            path: The path to the cache file.
        """
        self.path = Path(path)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Loads the cache from the file if it hasn't been loaded yet."""
        if self._loaded:
            return
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    payload = json.loads(line)
                    key = payload.get("cache_key")
                    if key:
                        self._cache[str(key)] = payload
        self._loaded = True

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieves an item from the cache.

        Args:
            key: The key of the item to retrieve.

        Returns:
            The cached data as a dictionary, or None if the key is not found.
        """
        self._ensure_loaded()
        return self._cache.get(key)

    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Adds or updates an item in the cache and appends it to the file.

        Args:
            key: The key of the item to set.
            data: The data to store, which must be JSON-serializable.
        """
        self._ensure_loaded()
        payload = {"cache_key": key, **data}
        self._cache[key] = payload
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["SearchCache"]
