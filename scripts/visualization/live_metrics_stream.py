"""Async streaming helpers for the honesty dashboard."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterable, Dict


class LiveMetricsStream:
    """Stream metrics from training loops to dashboard-friendly sinks."""

    def __init__(self, dashboard_endpoint: str | None = None, buffer_path: str | Path | None = None) -> None:
        self.dashboard_endpoint = dashboard_endpoint
        self.buffer_path = Path(buffer_path) if buffer_path else None
        if self.buffer_path:
            self.buffer_path.parent.mkdir(parents=True, exist_ok=True)

    async def stream_metrics(self, metrics_stream: AsyncIterable[Dict[str, Any]]) -> None:
        async for batch_metrics in metrics_stream:
            formatted = self._format_metrics(batch_metrics)
            await self._dispatch(formatted)

    async def _dispatch(self, payload: Dict[str, Any]) -> None:
        if self.buffer_path:
            serialized = json.dumps(payload, ensure_ascii=False)
            self.buffer_path.write_text(serialized + "\n", encoding="utf-8")
        if self.dashboard_endpoint:
            await self._simulate_http_post(payload)

    async def _simulate_http_post(self, payload: Dict[str, Any]) -> None:
        await asyncio.sleep(0)

    def _format_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        formatted = dict(metrics)
        formatted.setdefault("series", "honesty_training")
        return formatted


__all__ = ["LiveMetricsStream"]
