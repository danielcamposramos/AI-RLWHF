import json
import os
from pathlib import Path

import pytest

from plugins.experimental.grok_search_evaluator.main import EvaluatorConfig, evaluate_examples, load_examples, write_results


def test_evaluator_offline(tmp_path, offline_reference_map):
    dataset = tmp_path / "dataset.jsonl"
    sample = {"prompt": "Explain RLHF.", "student_answer": "RLHF mixes human feedback with RL."}
    dataset.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    cfg = EvaluatorConfig(
        dataset_path=dataset,
        output_path=tmp_path / "out.jsonl",
        cache_path=tmp_path / "cache.jsonl",
        offline_reference_path=tmp_path / "offline.jsonl",
        use_internet=False,
        max_examples=10,
    )
    cfg.offline_reference_path.write_text(
        json.dumps({"prompt": sample["prompt"], "reference": "RLHF uses human feedback rewards."}) + "\n",
        encoding="utf-8",
    )
    examples = load_examples(cfg.dataset_path, cfg.max_examples)
    records = evaluate_examples(examples, cfg)
    assert records
    assert records[0]["reward"] <= 2
    write_results(records, cfg.output_path)
    assert cfg.output_path.exists()
    logged = cfg.output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(logged) == 1
    row = json.loads(logged[0])
    assert "search_context" in row


def test_cache_roundtrip(tmp_path):
    from scripts.utils.search_cache import SearchCache

    cache = SearchCache(tmp_path / "cache.jsonl")
    cache.set("key", {"source": "api", "snippets": ["foo"]})
    entry = cache.get("key")
    assert entry is not None
    assert entry["snippets"] == ["foo"]
