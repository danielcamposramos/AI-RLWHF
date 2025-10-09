"""Grok search evaluator plugin with optional internet and offline fallbacks."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import requests  # type: ignore

try:  # Transformer Lab runtime
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover - fallback for local execution
    class _DummyTrainer:  # pylint: disable=too-few-public-methods
        def __init__(self) -> None:
            self.params: Dict[str, Any] = {}

        def job_wrapper(self):
            def decorator(func):
                return func

            return decorator

        def progress_update(self, *_args, **_kwargs) -> None:
            return None

    tlab_trainer = _DummyTrainer()  # type: ignore

from scripts.utils.offline_scoring import load_offline_reference, score_against_reference
from scripts.utils.search_cache import SearchCache

DEFAULT_DATASET = Path("data/processed/student_answers.jsonl")
DEFAULT_OUTPUT = Path("data/processed/honesty_logs/grok_search_evaluator.jsonl")
DEFAULT_CACHE = Path("data/processed/search_cache.jsonl")
DEFAULT_OFFLINE = Path("data/examples/offline_reference.jsonl")
DEFAULT_ENDPOINT = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4"


@dataclass
class EvaluatorConfig:
    dataset_path: Path = DEFAULT_DATASET
    output_path: Path = DEFAULT_OUTPUT
    cache_path: Path = DEFAULT_CACHE
    offline_reference_path: Path = DEFAULT_OFFLINE
    use_internet: bool = True
    api_endpoint: str = DEFAULT_ENDPOINT
    api_key_env: str = "XAI_API_KEY"
    model: str = DEFAULT_MODEL
    max_examples: int = 100
    score_if_uncertain: int = 0

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


def load_examples(path: Path, limit: int) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "prompt" in payload and "student_answer" in payload:
                items.append(payload)
            if 0 < limit <= len(items):
                break
    return items


def call_grok_search(prompt: str, config: EvaluatorConfig, cache: SearchCache) -> Dict[str, Any]:
    cache_key = f"{config.model}:{prompt.strip()}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": "Return factual snippets that verify or refute the claim."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    if not config.use_internet or not config.api_key:
        result = {"source": "offline", "snippets": []}
        cache.set(cache_key, result)
        return result
    try:
        response = requests.post(  # noqa: S113
            config.api_endpoint,
            headers={"Authorization": f"Bearer {config.api_key}"},
            json=payload,
            timeout=20,
        )
        response.raise_for_status()
        body = response.json()
        message = body.get("choices", [{}])[0].get("message", {}).get("content", "")
        result = {"source": "api", "snippets": [message] if message else []}
        cache.set(cache_key, result)
        return result
    except Exception as exc:  # pragma: no cover - network issues
        result = {"source": f"error:{exc}", "snippets": []}
        cache.set(cache_key, result)
        return result


def derive_reward(prompt: str, student_answer: str, offline_map: Mapping[str, str], search_result: Dict[str, Any], fallback: int) -> Dict[str, Any]:
    score, feedback = score_against_reference(prompt, student_answer, offline_map)
    snippets = "\n".join(search_result.get("snippets", []))
    source = search_result.get("source", "offline")
    if score == 0 and snippets:
        lower_snippets = snippets.lower()
        if any(phrase in lower_snippets for phrase in ["accurate", "correct", "verified"]):
            score = min(2, max(0, score + 2))
            feedback = "Search confirmation indicates correctness."
        elif any(keyword in lower_snippets for keyword in ["incorrect", "false", "disputed"]):
            score = max(-2, min(0, score - 2))
            feedback = "Search context challenges the claim."
    if snippets == "" and score == 0:
        score = fallback
    return {
        "reward": int(max(-2, min(2, score))),
        "feedback": feedback,
        "search_context": snippets,
        "search_source": source,
    }


def evaluate_examples(examples: Iterable[Mapping[str, Any]], config: EvaluatorConfig) -> List[Dict[str, Any]]:
    offline_map = load_offline_reference(config.offline_reference_path)
    cache = SearchCache(config.cache_path)
    results: List[Dict[str, Any]] = []
    for payload in examples:
        prompt = str(payload.get("prompt", ""))
        student_answer = str(payload.get("student_answer", ""))
        if not prompt or not student_answer:
            continue
        search_result = call_grok_search(prompt, config, cache)
        reward_payload = derive_reward(prompt, student_answer, offline_map, search_result, config.score_if_uncertain)
        record = {
            "prompt": prompt,
            "student_answer": student_answer,
            "feedback": reward_payload["feedback"],
            "reward": reward_payload["reward"],
            "search_context": reward_payload["search_context"],
            "search_source": reward_payload["search_source"],
        }
        results.append(record)
    return results


def write_results(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"processed": 0, "average_reward": 0}
    avg = sum(row.get("reward", 0) for row in records) / len(records)
    return {"processed": len(records), "average_reward": avg}


def _collect_params(overrides: Optional[Mapping[str, Any]] = None) -> EvaluatorConfig:
    params = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    if overrides:
        params.update(overrides)
    return EvaluatorConfig(
        dataset_path=Path(params.get("dataset_path", DEFAULT_DATASET)),
        output_path=Path(params.get("output_path", DEFAULT_OUTPUT)),
        cache_path=Path(params.get("cache_path", DEFAULT_CACHE)),
        offline_reference_path=Path(params.get("offline_reference_path", DEFAULT_OFFLINE)),
        use_internet=bool(params.get("use_internet", True)),
        api_endpoint=str(params.get("api_endpoint", DEFAULT_ENDPOINT)),
        api_key_env=str(params.get("api_key_env", "XAI_API_KEY")),
        model=str(params.get("model", DEFAULT_MODEL)),
        max_examples=int(params.get("max_examples", 100)),
        score_if_uncertain=int(params.get("score_if_uncertain", 0)),
    )


@tlab_trainer.job_wrapper()
def grok_search_evaluator(**overrides):
    progress_cb = getattr(tlab_trainer, "progress_update", None)
    if callable(progress_cb):
        progress_cb(5)
    config = _collect_params(overrides)
    examples = load_examples(config.dataset_path, config.max_examples)
    if callable(progress_cb):
        progress_cb(25)
    records = evaluate_examples(examples, config)
    write_results(records, config.output_path)
    summary = summarize(records)
    if callable(progress_cb):
        progress_cb(100)
    return summary


if __name__ == "__main__":  # pragma: no cover
    cfg = EvaluatorConfig()
    data = load_examples(cfg.dataset_path, cfg.max_examples)
    rows = evaluate_examples(data, cfg)
    write_results(rows, cfg.output_path)
    print(json.dumps(summarize(rows), indent=2))
