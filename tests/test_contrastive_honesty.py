import json

import pytest

from plugins.core.honesty_reward_calculator import HonestyRewardCalculator
from plugins.core.multi_teacher_aggregator.claude_sdk_connector import ClaudeAgentSDKConnector
from plugins.core.multi_teacher_aggregator.codex_oauth_connector import CodexOAuthConnector
from scripts.data_pipeline.data_quality_gate import validate, validate_decomposition
from scripts.data_pipeline.rlwhf_tuple_handler import RLWHFTupleHandler
from scripts.data_pipeline.triplet_miner import mine_hard_negatives, mine_triplets


def _valid_decomposition():
    return {
        "positive_fragments": [
            {"text": "Paris is the capital of France.", "correctness": 1.0, "category": "fact"}
        ],
        "negative_fragments": [
            {
                "text": "France is in South America.",
                "correctness": 0.0,
                "category": "fabrication",
                "correction": "France is in Europe.",
            }
        ],
        "honesty_signals": [
            {"text": "I may be missing context.", "honesty_score": 0.9, "appropriate": True}
        ],
        "missing_honesty": [
            {"claim": "France is in South America.", "reason": "geography claim was uncertain"}
        ],
        "overall_honesty": 0.8,
        "overall_correctness": 0.5,
    }


def _valid_tuple(*, decomposition=None):
    payload = {
        "prompt": "What is the capital of France?",
        "answer": "Paris is the capital of France.",
        "feedback": "Correct answer.",
        "reward": 2,
        "metadata": {
            "source_ai": "teacher",
            "confidence_score": 0.9,
            "rubric_dimension": "honesty",
        },
    }
    if decomposition is not None:
        payload["decomposition"] = decomposition
    return payload


def test_data_quality_gate_accepts_legacy_and_contrastive_rows(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    rows = [
        _valid_tuple(),
        _valid_tuple(decomposition=_valid_decomposition()),
    ]
    dataset.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    assert validate(str(dataset))


def test_data_quality_gate_rejects_invalid_decomposition():
    invalid = _valid_decomposition()
    invalid["negative_fragments"] = [{"text": "bad", "correctness": 0.1, "category": "fabrication"}]
    assert not validate_decomposition(invalid)


def test_tuple_handler_preserves_decomposition(tmp_path):
    handler = RLWHFTupleHandler()
    output = tmp_path / "processed.jsonl"
    rows = [_valid_tuple(decomposition=_valid_decomposition())]
    handler.create_training_dataset(rows, str(output))
    loaded = handler.load_jsonl(str(output))
    assert loaded[0]["decomposition"]["overall_honesty"] == pytest.approx(0.8)


def test_triplet_miner_generates_correctness_honesty_and_contrast_axes():
    tuples = [
        {
            "prompt": "Explain quantum entanglement.",
            "answer": "Two particles can be correlated. I am not entirely certain about the mechanism.",
            "reward": 1,
            "decomposition": {
                "positive_fragments": [
                    {"text": "Two particles can be correlated.", "correctness": 0.9, "category": "core_concept"}
                ],
                "negative_fragments": [
                    {
                        "text": "They send information faster than light.",
                        "correctness": 0.0,
                        "category": "fabrication",
                        "correction": "Entanglement does not allow faster-than-light communication.",
                    }
                ],
                "honesty_signals": [
                    {
                        "text": "I am not entirely certain about the mechanism.",
                        "honesty_score": 1.0,
                        "appropriate": True,
                    }
                ],
                "missing_honesty": [
                    {"claim": "They send information faster than light.", "reason": "should have been flagged as uncertain"}
                ],
                "overall_honesty": 0.9,
                "overall_correctness": 0.5,
            },
        },
        {
            "prompt": "Explain quantum entanglement.",
            "answer": "Entanglement sends information instantly.",
            "reward": -1,
            "decomposition": _valid_decomposition(),
        },
    ]
    triplets = mine_triplets(tuples, min_reward_gap=1, max_triplets_per_prompt=10, fragment_min_length=10)
    axes = {triplet.axis for triplet in triplets}
    assert {"correctness", "honesty", "contrast"} <= axes


def test_hard_negative_miner_prefers_most_similar_negatives():
    selected = mine_hard_negatives([1.0, 0.0], [[0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], k=1)
    assert selected == [[0.9, 0.1]]


def test_reward_calculator_uses_decomposition_to_floor_honest_uncertainty():
    calculator = HonestyRewardCalculator()
    reward = calculator.calculate_reward(
        teacher_score=-1,
        confidence_score=0.95,
        metadata={},
        decomposition={"overall_honesty": 0.85, "overall_correctness": 0.2},
    )
    assert reward >= 0.0

    penalty = calculator.calculate_reward(
        teacher_score=-1,
        confidence_score=0.95,
        metadata={},
        decomposition={"overall_honesty": 0.1, "overall_correctness": 0.1},
    )
    assert penalty <= -1.5


def test_teacher_connectors_fail_gracefully_when_unavailable(monkeypatch):
    claude = ClaudeAgentSDKConnector()
    monkeypatch.setattr(claude, "is_available", lambda: False)
    claude_result = claude.evaluate("prompt", "answer", "system")
    assert claude_result["available"] is False
    assert claude_result["reward"] == 0

    codex = CodexOAuthConnector()
    monkeypatch.setattr(codex, "is_available", lambda: False)
    codex_result = codex.evaluate("prompt", "answer", "system")
    assert codex_result["available"] is False
    assert codex_result["reward"] == 0


def test_embedding_projector_output_shape_and_normalization():
    torch = pytest.importorskip("torch")
    from plugins.core.embedding_projector import EmbeddingProjector

    projector = EmbeddingProjector(input_dim=16, hidden_dim=32, output_dim=8)
    hidden = torch.randn(4, 6, 16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]] * 4)
    output = projector(hidden, attention_mask=mask)
    assert output.shape == (4, 8)
    norms = torch.linalg.norm(output, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_contrastive_loss_is_finite():
    torch = pytest.importorskip("torch")
    from plugins.core.contrastive_loss import ContrastiveHonestyLoss

    module = ContrastiveHonestyLoss(
        {
            "loss_weights": {"correctness": 0.25, "honesty": 0.40, "contrast": 0.15},
            "temperature": {"initial": 0.1, "final": 0.03, "schedule": "exponential"},
        }
    )
    embeddings = {
        "question": torch.randn(2, 8),
        "positive_fragments": torch.randn(2, 8),
        "negative_fragments": torch.randn(2, 3, 8),
        "response": torch.randn(2, 8),
        "honesty_signals": torch.randn(2, 8),
        "missing_honesty": torch.randn(2, 2, 8),
        "better_response": torch.randn(2, 8),
        "worse_response": torch.randn(2, 8),
    }
    losses = module(embeddings, step=5, total_steps=100)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["correctness"])
    assert torch.isfinite(losses["honesty"])
    assert torch.isfinite(losses["contrast"])
