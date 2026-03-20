# Codex Directive: Implement Contrastive Honesty Learning

**Date:** 2026-03-20
**Priority:** HIGH
**Specification:** `docs/CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md`
**Repo:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/AI-RLWHF`

---

## Context

AI-RLWHF currently trains with scalar GRPO rewards (+2 to -2). We are adding **contrastive learning** that decomposes every teacher response into positive/negative/honesty fragments, producing 3-8x more gradient signals per training sample. This is an EXPANSION of the existing architecture, not a replacement.

**Read the full spec first:** `docs/CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md`

---

## Implementation Phases (Do All)

### Phase 1: Extended Teacher Prompt + Data Schema

**1a. Update teacher prompt**

File: `configs/prompts/teacher/system.md`

Replace the current prompt with the extended version from spec §6. The key change: teacher must return a `decomposition` object with `positive_fragments`, `negative_fragments`, `honesty_signals`, `missing_honesty`, `overall_honesty`, and `overall_correctness`. Keep the existing rubric and HAL: tags.

**1b. Update data quality gate**

File: `scripts/data_pipeline/data_quality_gate.py`

Add validation for the `decomposition` field. It MUST be backward-compatible: old tuples without `decomposition` pass validation (they feed GRPO only). New tuples with `decomposition` get validated:
- `positive_fragments`: list of `{text, correctness, category}`
- `negative_fragments`: list of `{text, correctness, category, correction}`
- `honesty_signals`: list of `{text, honesty_score, appropriate}`
- `missing_honesty`: list of `{claim, reason}`
- `overall_honesty`: float 0.0-1.0
- `overall_correctness`: float 0.0-1.0

**1c. Update tuple handler**

File: `scripts/data_pipeline/rlwhf_tuple_handler.py`

Ensure the tuple handler preserves the `decomposition` field when processing JSONL. No transformation needed — just don't drop unknown fields.

---

### Phase 2: Contrastive Loss Module

**2a. Triplet miner**

New file: `scripts/data_pipeline/triplet_miner.py`

Implements the triplet mining from spec §7.3:
- Groups tuples by prompt
- Generates cross-response triplets (Axis 3: better vs worse response to same prompt)
- Generates fragment-level triplets from decomposition (Axis 1: correct vs incorrect fragments, Axis 2: honesty signals vs missing honesty)
- Hard negative mining: select negatives with highest cosine similarity to anchor (spec §4.3)

```python
@dataclass
class Triplet:
    anchor: str          # question or response text
    positive: str        # correct/honest fragment or better response
    negative: str        # incorrect/dishonest fragment or worse response
    axis: str            # "correctness" | "honesty" | "contrast"
    margin: float = 1.0  # reward gap (for weighted loss)

def mine_triplets(tuples: list[dict]) -> list[Triplet]: ...
def mine_hard_negatives(anchor_embed, negative_pool, k=7): ...
```

**2b. Embedding projector**

New file: `plugins/core/embedding_projector.py`

A 2-layer MLP projection head that maps transformer hidden states → 256-dim normalized embedding space:

```python
class EmbeddingProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        projected = self.projector(hidden_states)
        return F.normalize(projected, dim=-1)  # L2 normalize
```

Must handle:
- Extraction of hidden states from transformer's last layer (or configurable layer)
- Mean pooling over sequence length for sentence-level embeddings
- Batch processing for efficiency

**2c. Contrastive loss**

New file: `plugins/core/contrastive_loss.py`

Three-axis InfoNCE loss from spec §4.1:

```python
class ContrastiveHonestyLoss(nn.Module):
    def __init__(self, config: dict):
        self.alpha = config["loss_weights"]["correctness"]   # 0.25
        self.beta = config["loss_weights"]["honesty"]         # 0.40
        self.gamma = config["loss_weights"]["contrast"]       # 0.15
        self.temperature = config["temperature"]["initial"]   # 0.1

    def forward(self, embeddings: dict) -> dict:
        L_correct = self.infonce(
            embeddings["question"],
            embeddings["positive_fragments"],
            embeddings["negative_fragments"]
        )
        L_honesty = self.infonce(
            embeddings["response"],
            embeddings["honesty_signals"],
            embeddings["missing_honesty"]
        )
        L_contrast = self.infonce(
            embeddings["question"],
            embeddings["better_response"],
            embeddings["worse_response"]
        )
        total = self.alpha * L_correct + self.beta * L_honesty + self.gamma * L_contrast
        return {"total": total, "correctness": L_correct, "honesty": L_honesty, "contrast": L_contrast}

    def infonce(self, anchor, positives, negatives):
        # Standard InfoNCE: -log(exp(sim(a,p)/τ) / Σ exp(sim(a,n)/τ))
        ...
```

Temperature schedule (exponential decay from 0.1 → 0.03 over training).

**2d. Config file**

New file: `configs/training/contrastive_config.json`

Use the schema from spec §9. Include all defaults.

---

### Phase 3: Training Loop Integration

**3a. Master launcher**

File: `scripts/training/master_rlwhf_launcher.py`

Add contrastive training path alongside existing GRPO:
- Load `configs/training/contrastive_config.json`
- If `contrastive_enabled: true`:
  - Mine triplets from training tuples (call `triplet_miner.py`)
  - Initialize `EmbeddingProjector` and `ContrastiveHonestyLoss`
  - Compute hybrid loss: `L = α*L_correct + β*L_honesty + γ*L_contrast + δ*L_grpo`
  - Log per-axis losses to telemetry
- If `contrastive_enabled: false`: existing GRPO-only path (unchanged)

**3b. GRPO wrapper**

File: `plugins/core/grpo_production_wrapper.py`

Extend to accept the combined loss. The GRPO loss computation stays as-is. The contrastive loss is computed separately and added with weight δ (default 0.20). The GRPO wrapper just needs to:
- Accept an optional `contrastive_loss` term
- Add it to the GRPO loss before backward pass
- Log both loss components

**3c. Reward calculator**

File: `plugins/core/honesty_reward_calculator.py`

Extend `calculate_reward()` to use decomposition when available:

```python
def calculate_reward(self, teacher_score, confidence_score, metadata, decomposition=None):
    base_reward = float(teacher_score)

    if decomposition:
        # Use fine-grained honesty/correctness instead of just scalar
        honesty = decomposition.get("overall_honesty", 0.5)
        correctness = decomposition.get("overall_correctness", 0.5)

        # Honesty is principal: reward honest uncertainty even when wrong
        if honesty > 0.7 and correctness < 0.5:
            base_reward = max(base_reward, 0.0)  # Floor at 0, not negative

        # Penalize confident incorrectness harder
        if honesty < 0.3 and correctness < 0.3 and confidence_score > 0.8:
            base_reward = min(base_reward, -1.5)

    # Existing confidence adjustment
    if teacher_score > 0 and confidence_score < 0.5:
        base_reward *= 0.8
    elif teacher_score < 0 and confidence_score > 0.8:
        base_reward *= 1.2

    return base_reward
```

---

### Phase 4: New Teacher Connectors

**4a. Claude Agent SDK connector**

New file: `plugins/core/multi_teacher_aggregator/claude_sdk_connector.py`

```python
class ClaudeAgentSDKConnector:
    """Teacher connector using Claude Agent SDK (subscription billing).

    Pattern from: knowledge3d/tools/augmentation_providers.py:435
    Package: claude_agent_sdk (pip install claude-agent-sdk)
    Auth: CLI subscription (Pro/Team/Enterprise) — no API key needed.
    """
    def __init__(self, model="claude-sonnet-4-6", timeout=120.0):
        ...

    def evaluate(self, prompt: str, student_answer: str, system_prompt: str) -> dict:
        """Returns teacher evaluation with decomposition."""
        ...
```

Handle graceful fallback if `claude_agent_sdk` package is not installed.

**4b. Codex OAuth connector**

New file: `plugins/core/multi_teacher_aggregator/codex_oauth_connector.py`

```python
class CodexOAuthConnector:
    """Teacher connector using Codex OAuth (subscription billing).

    Pattern from: knowledge3d/tools/augmentation_providers.py:507
    Package: openai (standard, pointed at Codex endpoint)
    Auth: OPENAI_API_KEY env var from Codex CLI session.
    """
    def __init__(self, model="gpt-4o", timeout=120.0):
        ...

    def evaluate(self, prompt: str, student_answer: str, system_prompt: str) -> dict:
        """Returns teacher evaluation with decomposition."""
        ...
```

**4c. Register new connection types**

File: `plugins/core/multi_teacher_aggregator/main.py`

- Add `"claude_agent_sdk"` and `"codex_oauth"` to `ALLOWED_CONNECTIONS`
- Add default slot configs for both (see spec §5.2)
- Wire up the new connectors in the slot dispatch logic

---

### Phase 5: Metrics & Visualization

**5a. Contrastive metrics**

New file: `scripts/evaluation/contrastive_metrics.py`

Compute:
- Per-axis loss values (correctness, honesty, contrast)
- Embedding space quality: intra-cluster cosine similarity (same tier), inter-cluster distance (different tiers)
- Honesty calibration: correlation between `overall_honesty` and actual correctness
- Hard negative difficulty distribution

**5b. Embedding explorer**

New file: `scripts/visualization/embedding_explorer.py`

Generate t-SNE or UMAP plots of the embedding space, colored by rubric tier (-2 to +2). This shows whether the contrastive loss is actually separating honest from dishonest responses.

Optionally uses matplotlib (already in requirements.txt).

**5c. Training telemetry**

File: `scripts/telemetry/training_metrics.py`

Add tracking for:
- `loss_correctness`, `loss_honesty`, `loss_contrast`, `loss_grpo` (per step)
- `temperature` (current value in schedule)
- `triplets_mined` (per batch)
- `hard_negative_similarity` (average similarity of selected hard negatives)
- `embedding_cluster_separation` (per epoch)

---

## Files Summary

| File | Status | Phase |
|------|--------|-------|
| `configs/prompts/teacher/system.md` | MODIFY | 1 |
| `scripts/data_pipeline/data_quality_gate.py` | MODIFY | 1 |
| `scripts/data_pipeline/rlwhf_tuple_handler.py` | MODIFY | 1 |
| `scripts/data_pipeline/triplet_miner.py` | NEW | 2 |
| `plugins/core/embedding_projector.py` | NEW | 2 |
| `plugins/core/contrastive_loss.py` | NEW | 2 |
| `configs/training/contrastive_config.json` | NEW | 2 |
| `scripts/training/master_rlwhf_launcher.py` | MODIFY | 3 |
| `plugins/core/grpo_production_wrapper.py` | MODIFY | 3 |
| `plugins/core/honesty_reward_calculator.py` | MODIFY | 3 |
| `plugins/core/multi_teacher_aggregator/claude_sdk_connector.py` | NEW | 4 |
| `plugins/core/multi_teacher_aggregator/codex_oauth_connector.py` | NEW | 4 |
| `plugins/core/multi_teacher_aggregator/main.py` | MODIFY | 4 |
| `scripts/evaluation/contrastive_metrics.py` | NEW | 5 |
| `scripts/visualization/embedding_explorer.py` | NEW | 5 |
| `scripts/telemetry/training_metrics.py` | MODIFY | 5 |

## Read Before Starting

1. `docs/CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md` — FULL spec (read completely)
2. `docs/rlwhf-framework.md` — current architecture
3. `plugins/core/custom_honesty_rm/main.py` — current reward model
4. `plugins/core/multi_teacher_aggregator/main.py` — current teacher system
5. `plugins/core/grpo_production_wrapper.py` — current GRPO wrapper
6. `scripts/training/master_rlwhf_launcher.py` — current training entry point
7. K3D reference for connectors: `knowledge3d/tools/augmentation_providers.py` lines 435-550

## Success Criteria

1. `python3 -m compileall` passes on all modified/new files
2. Existing GRPO-only path still works when `contrastive_enabled: false`
3. New teacher prompt returns valid decomposition JSON
4. Triplet miner produces correct triplets from sample data (test with `data/examples/sample_student_answers.jsonl`)
5. Contrastive loss computes without NaN/Inf on sample embeddings
6. Claude Agent SDK and Codex OAuth connectors handle graceful fallback when packages not installed
7. Unit tests for: triplet mining, contrastive loss, embedding projector, data quality gate, reward calculator with decomposition

## Constraints

- This is a TransformerLab plugin — standard PyTorch, NO ternary, NO K3D sovereignty rules
- Backward compatible — old tuples without decomposition must still work
- Keep existing GRPO path intact — contrastive is ADDITIVE, not replacement
- All new code must have `try/except` for optional imports (torch, sentence_transformers, claude_agent_sdk)
- Follow existing plugin patterns (`tlab_trainer` decorator, `index.json` manifests)
