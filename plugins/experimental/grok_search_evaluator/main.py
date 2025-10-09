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
from scripts.utils.prompt_loader import load_prompt
from scripts.utils.search_cache import SearchCache

DEFAULT_DATASET = Path("data/processed/student_answers.jsonl")
DEFAULT_OUTPUT = Path("data/processed/honesty_logs/grok_search_evaluator.jsonl")
DEFAULT_CACHE = Path("data/processed/search_cache.jsonl")
DEFAULT_OFFLINE = Path("data/examples/offline_reference.jsonl")
DEFAULT_ENDPOINT = "https://api.x.ai/v1/chat/completions"
DEFAULT_MODEL = "grok-4"
DEFAULT_SYSTEM_PROMPT_PATH = Path("configs/prompts/teacher/system.md")


@dataclass
class EvaluatorConfig:
    """Configuration for the Grok search evaluator.

    Attributes:
        dataset_path: Path to the input dataset of student answers.
        output_path: Path to write the evaluation results.
        cache_path: Path to the search cache.
        offline_reference_path: Path to the offline reference data.
        use_internet: Whether to use the internet for searches.
        api_endpoint: The Grok API endpoint.
        api_key_env: The environment variable for the API key.
        model: The model to use for evaluation.
        system_prompt: The system prompt to use for the model.
        system_prompt_path: The path to the system prompt file.
        max_examples: The maximum number of examples to evaluate.
        score_if_uncertain: The score to assign if the result is uncertain.
        api_context_ratio: The context ratio for API calls.
        max_context_tokens: The maximum number of context tokens.
    """
    dataset_path: Path = DEFAULT_DATASET
    output_path: Path = DEFAULT_OUTPUT
    cache_path: Path = DEFAULT_CACHE
    offline_reference_path: Path = DEFAULT_OFFLINE
    use_internet: bool = True
    api_endpoint: str = DEFAULT_ENDPOINT
    api_key_env: str = "XAI_API_KEY"
    model: str = DEFAULT_MODEL
    system_prompt: str = ""
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH
    max_examples: int = 100
    score_if_uncertain: int = 0
    api_context_ratio: float = 0.66
    max_context_tokens: int = 4096

    @property
    def api_key(self) -> Optional[str]:
        """Retrieves the API key from the environment."""
        return os.environ.get(self.api_key_env)


def load_examples(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Loads examples from a JSONL file.

    Args:
        path: The path to the JSONL file.
        limit: The maximum number of examples to load.

    Returns:
        A list of dictionaries, where each dictionary represents an example.
    """
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
    """Calls the Grok search API with caching.

    Args:
        prompt: The prompt to search for.
        config: The evaluator configuration.
        cache: The search cache.

    Returns:
        A dictionary containing the search result.
    """
    cache_key = f"{config.model}:{prompt.strip()}"
    cached = cache.get(cache_key)
    if cached:
        return cached
    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": config.system_prompt or "Return factual snippets that verify or refute the claim."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": max(32, int(config.max_context_tokens * config.api_context_ratio)),
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
    """Derives a reward score based on offline and online evaluation.

    Args:
        prompt: The original prompt.
        student_answer: The student's answer.
        offline_map: The offline reference data.
        search_result: The result from the Grok search.
        fallback: The score to assign in case of uncertainty.

    Returns:
        A dictionary containing the reward, feedback, and search context.
    """
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
    """Evaluates a collection of examples.

    Args:
        examples: An iterable of examples to evaluate.
        config: The evaluator configuration.

    Returns:
        A list of dictionaries, where each dictionary contains the evaluation result.
    """
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
    """Writes evaluation results to a JSONL file.

    Args:
        records: An iterable of evaluation records.
        path: The path to the output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarizes the evaluation results.

    Args:
        records: A list of evaluation records.

    Returns:
        A dictionary containing a summary of the results.
    """
    if not records:
        return {"processed": 0, "average_reward": 0}
    avg = sum(row.get("reward", 0) for row in records) / len(records)
    return {"processed": len(records), "average_reward": avg}


def _collect_params(overrides: Optional[Mapping[str, Any]] = None) -> EvaluatorConfig:
    """Collects and validates parameters for the evaluator.

    Args:
        overrides: A mapping of parameter overrides.

    Returns:
        An EvaluatorConfig object.
    """
    params = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    if overrides:
        params.update(overrides)
    system_prompt_path = Path(params.get("system_prompt_path", DEFAULT_SYSTEM_PROMPT_PATH))
    system_prompt = load_prompt(system_prompt_path, fallback="Return factual snippets that verify or refute the claim.")
    return EvaluatorConfig(
        dataset_path=Path(params.get("dataset_path", DEFAULT_DATASET)),
        output_path=Path(params.get("output_path", DEFAULT_OUTPUT)),
        cache_path=Path(params.get("cache_path", DEFAULT_CACHE)),
        offline_reference_path=Path(params.get("offline_reference_path", DEFAULT_OFFLINE)),
        use_internet=bool(params.get("use_internet", True)),
        api_endpoint=str(params.get("api_endpoint", DEFAULT_ENDPOINT)),
        api_key_env=str(params.get("api_key_env", "XAI_API_KEY")),
        model=str(params.get("model", DEFAULT_MODEL)),
        system_prompt=system_prompt,
        system_prompt_path=system_prompt_path,
        max_examples=int(params.get("max_examples", 100)),
        score_if_uncertain=int(params.get("score_if_uncertain", 0)),
        api_context_ratio=float(params.get("api_context_ratio", 0.66)),
        max_context_tokens=int(params.get("max_context_tokens", 4096)),
    )


@tlab_trainer.job_wrapper()
def grok_search_evaluator(**overrides):
    """Main entry point for the Grok search evaluator plugin.

    This function is compatible with both Transformer Lab and direct invocation.

    Args:
        **overrides: A dictionary of parameter overrides.

    Returns:
        A dictionary summarizing the evaluation results.
    """
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
