"""Three-axis contrastive loss for honesty learning."""
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
    class ContrastiveHonestyLoss:  # type: ignore[override]
        """Fallback loss that fails cleanly when torch is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ImportError("torch is required to use ContrastiveHonestyLoss")


else:
    class ContrastiveHonestyLoss(nn.Module):
        """Combined correctness, honesty, and response-contrast loss."""

        def __init__(self, config: dict[str, Any]):
            super().__init__()
            loss_weights = dict(config.get("loss_weights", {}))
            temperature = dict(config.get("temperature", {}))
            self.alpha = float(loss_weights.get("correctness", 0.25))
            self.beta = float(loss_weights.get("honesty", 0.40))
            self.gamma = float(loss_weights.get("contrast", 0.15))
            self.temperature_initial = float(temperature.get("initial", 0.1))
            self.temperature_final = float(temperature.get("final", 0.03))
            self.temperature_schedule = str(temperature.get("schedule", "exponential")).lower()

        def current_temperature(self, step: int = 0, total_steps: int | None = None) -> float:
            """Return the scheduled temperature for the current step."""
            if not total_steps or total_steps <= 0:
                return self.temperature_initial
            ratio = max(0.0, min(1.0, float(step) / float(total_steps)))
            if self.temperature_schedule != "exponential":
                return self.temperature_initial + (self.temperature_final - self.temperature_initial) * ratio
            if self.temperature_initial <= 0 or self.temperature_final <= 0:
                return self.temperature_initial
            return self.temperature_initial * ((self.temperature_final / self.temperature_initial) ** ratio)

        @staticmethod
        def _ensure_anchor(anchor: Tensor) -> Tensor:
            if anchor.dim() != 2:
                raise ValueError(f"Anchor must be [B, D], got {tuple(anchor.shape)}")
            return F.normalize(anchor, dim=-1)

        @staticmethod
        def _ensure_samples(samples: Tensor, batch_size: int) -> Tensor:
            if samples.dim() == 2:
                samples = samples.unsqueeze(1)
            if samples.dim() != 3:
                raise ValueError(f"Samples must be [B, D] or [B, K, D], got {tuple(samples.shape)}")
            if samples.shape[0] != batch_size:
                raise ValueError("Anchor and sample batch sizes must match")
            return F.normalize(samples, dim=-1)

        def infonce(
            self,
            anchor: Tensor,
            positives: Tensor,
            negatives: Tensor,
            *,
            temperature: float | None = None,
        ) -> Tensor:
            """Compute InfoNCE for one axis."""
            tau = float(temperature or self.temperature_initial)
            anchor = self._ensure_anchor(anchor)
            positives = self._ensure_samples(positives, anchor.shape[0])
            negatives = self._ensure_samples(negatives, anchor.shape[0])

            anchor_expanded = anchor.unsqueeze(1)
            positive_logits = F.cosine_similarity(anchor_expanded, positives, dim=-1).mean(dim=1, keepdim=True) / tau
            negative_logits = F.cosine_similarity(anchor_expanded, negatives, dim=-1) / tau
            logits = torch.cat([positive_logits, negative_logits], dim=1)
            targets = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)
            return F.cross_entropy(logits, targets)

        @staticmethod
        def _zero_from(embeddings: dict[str, Tensor]) -> Tensor:
            for value in embeddings.values():
                if isinstance(value, Tensor):
                    return value.new_zeros(())
            return torch.tensor(0.0)

        def forward(
            self,
            embeddings: dict[str, Tensor],
            *,
            step: int = 0,
            total_steps: int | None = None,
        ) -> dict[str, Tensor]:
            """Compute the weighted three-axis contrastive loss."""
            tau = self.current_temperature(step=step, total_steps=total_steps)
            zero = self._zero_from(embeddings)

            l_correct = zero
            if {"question", "positive_fragments", "negative_fragments"} <= set(embeddings):
                l_correct = self.infonce(
                    embeddings["question"],
                    embeddings["positive_fragments"],
                    embeddings["negative_fragments"],
                    temperature=tau,
                )

            l_honesty = zero
            if {"response", "honesty_signals", "missing_honesty"} <= set(embeddings):
                l_honesty = self.infonce(
                    embeddings["response"],
                    embeddings["honesty_signals"],
                    embeddings["missing_honesty"],
                    temperature=tau,
                )

            l_contrast = zero
            if {"question", "better_response", "worse_response"} <= set(embeddings):
                l_contrast = self.infonce(
                    embeddings["question"],
                    embeddings["better_response"],
                    embeddings["worse_response"],
                    temperature=tau,
                )

            total = self.alpha * l_correct + self.beta * l_honesty + self.gamma * l_contrast
            return {
                "total": total,
                "correctness": l_correct,
                "honesty": l_honesty,
                "contrast": l_contrast,
                "temperature": torch.tensor(float(tau), device=total.device),
            }


__all__ = ["ContrastiveHonestyLoss"]
