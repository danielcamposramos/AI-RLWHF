"""Projection head for sentence-level contrastive embeddings."""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in lightweight environments
    import torch
    import torch.nn.functional as F
    from torch import Tensor, nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


if nn is None:  # pragma: no cover - exercised when torch missing
    class EmbeddingProjector:  # type: ignore[override]
        """Fallback projector that fails cleanly when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("torch is required to use EmbeddingProjector")


else:
    class EmbeddingProjector(nn.Module):
        """Project hidden states into a normalized sentence embedding space."""

        def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256) -> None:
            super().__init__()
            self.projector = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Linear(int(hidden_dim), int(output_dim)),
            )

        @staticmethod
        def mean_pool(hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
            """Pool sequence states into one embedding per example."""
            if hidden_states.dim() == 2:
                return hidden_states
            if hidden_states.dim() != 3:
                raise ValueError(f"Expected [B, H] or [B, T, H], got {tuple(hidden_states.shape)}")
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp_min(1.0)
            return summed / counts

        def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
            pooled = self.mean_pool(hidden_states, attention_mask=attention_mask)
            projected = self.projector(pooled)
            return F.normalize(projected, dim=-1)


__all__ = ["EmbeddingProjector"]
