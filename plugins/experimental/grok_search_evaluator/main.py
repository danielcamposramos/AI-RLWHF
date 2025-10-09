"""Grok search evaluator plugin with optional internet, offline, and DPO fallbacks."""
from __future__ import annotations

import json
import math
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

from plugins.core.custom_honesty_rm import score_with_custom_rm
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
    """Configuration bag for Grok search plus DPO-style reward shaping.

    Attributes:
        dataset_path: Path to the input JSONL file with prompts and student answers.
        output_path: Path where the evaluator writes JSONL results.
        cache_path: Location for cached search responses.
        offline_reference_path: Offline honesty reference JSONL path.
        use_internet: Whether remote Grok API calls are allowed.
        api_endpoint: REST endpoint for Grok chat completions.
        api_key_env: Name of env var storing the API key.
        model: Identifier of the Grok model to request.
        system_prompt: Prompt text loaded from `system_prompt_path` or overrides.
        system_prompt_path: File path to the system prompt template.
        max_examples: Maximum number of tuples to process.
        score_if_uncertain: Reward fallback when signals are inconclusive.
        api_context_ratio: Ratio of requested tokens relative to model context.
        max_context_tokens: Hard ceiling for Grok response tokens.
        enable_dpo_reward: Enables DPO-style deltas using a preferred answer column.
        dpo_beta: Temperature-style divisor controlling DPO delta scaling.
        preference_reference_field: Field in the dataset containing preferred answers.
        reward_artifact_path: Path to `honesty_reward_model.json` for scoring samples.
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
    enable_dpo_reward: bool = False
    dpo_beta: float = 0.1
    preference_reference_field: str = "preferred_answer"
    reward_artifact_path: Optional[Path] = None

    @property
    def api_key(self) -> Optional[str]:
        """Retrieve the configured API key from the environment."""
        return os.environ.get(self.api_key_env)


def load_examples(path: Path, limit: int) -> List[Dict[str, Any]]:
    """Load evaluator tuples from disk, respecting a throughput limit."""
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
    """Call Grok (or fall back offline) while caching results by prompt signature."""
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
    except Exception as exc:  # pragma: no cover - network volatility
        result = {"source": f"error:{exc}", "snippets": []}
        cache.set(cache_key, result)
        return result


def derive_reward(
    prompt: str,
    student_answer: str,
    offline_map: Mapping[str, str],
    search_result: Dict[str, Any],
    fallback: int,
    config: EvaluatorConfig,
    example: Mapping[str, Any],
) -> Dict[str, Any]:
    """Fuse offline scoring, search snippets, and optional DPO adjustments."""
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
    if not snippets and score == 0:
        score = fallback

    if config.enable_dpo_reward:
        preferred_answer = str(example.get(config.preference_reference_field, "")).strip()
        dpo_stats = apply_dpo_adjustment(
            prompt=prompt,
            student_answer=student_answer,
            preferred_answer=preferred_answer,
            beta=config.dpo_beta,
            artifact_path=config.reward_artifact_path,
        )
        score = int(max(-2, min(2, score + dpo_stats["delta_reward"])))
    else:
        dpo_stats = {"student_score": 0.0, "reference_score": 0.0, "delta_reward": 0.0}

    return {
        "reward": int(max(-2, min(2, score))),
        "feedback": feedback,
        "search_context": snippets,
        "search_source": source,
        "dpo_student_score": dpo_stats["student_score"],
        "dpo_reference_score": dpo_stats["reference_score"],
        "dpo_delta_reward": dpo_stats["delta_reward"],
    }


def apply_dpo_adjustment(
    prompt: str,
    student_answer: str,
    preferred_answer: str,
    beta: float,
    artifact_path: Optional[Path],
) -> Dict[str, float]:
    """Calculate a DPO-inspired delta using the custom honesty reward model."""
    if not preferred_answer:
        return {"student_score": 0.0, "reference_score": 0.0, "delta_reward": 0.0}
    student = score_with_custom_rm(prompt, student_answer, critique="", artifact_path=artifact_path)
    reference = score_with_custom_rm(prompt, preferred_answer, critique="", artifact_path=artifact_path)
    student_score = student.get("normalized_score", 0.0)
    reference_score = reference.get("normalized_score", 0.0)
    safe_beta = beta if beta > 0 else 0.1
    delta = math.tanh((student_score - reference_score) / max(safe_beta, 1e-4)) * 2.0
    return {
        "student_score": float(student_score),
        "reference_score": float(reference_score),
        "delta_reward": float(delta),
    }


def evaluate_examples(examples: Iterable[Mapping[str, Any]], config: EvaluatorConfig) -> List[Dict[str, Any]]:
    """Evaluate prompts by combining offline scoring, search, and optional DPO."""
    offline_map = load_offline_reference(config.offline_reference_path)
    cache = SearchCache(config.cache_path)
    results: List[Dict[str, Any]] = []
    for payload in examples:
        prompt = str(payload.get("prompt", ""))
        student_answer = str(payload.get("student_answer", ""))
        if not prompt or not student_answer:
            continue
        search_result = call_grok_search(prompt, config, cache)
        reward_payload = derive_reward(
            prompt=prompt,
            student_answer=student_answer,
            offline_map=offline_map,
            search_result=search_result,
            fallback=config.score_if_uncertain,
            config=config,
            example=payload,
        )
        record = {
            "prompt": prompt,
            "student_answer": student_answer,
            "feedback": reward_payload["feedback"],
            "reward": reward_payload["reward"],
            "search_context": reward_payload["search_context"],
            "search_source": reward_payload["search_source"],
            "dpo_student_score": reward_payload["dpo_student_score"],
            "dpo_reference_score": reward_payload["dpo_reference_score"],
            "dpo_delta_reward": reward_payload["dpo_delta_reward"],
        }
        results.append(record)
    return results


def write_results(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Persist evaluator output as JSONL rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(records: List[Mapping[str, Any]]) -> Dict[str, Any]:
    """Produce quick aggregate metrics from evaluator output."""
    if not records:
        return {"processed": 0, "average_reward": 0, "dpo_delta_avg": 0}
    avg_reward = sum(row.get("reward", 0) for row in records) / len(records)
    avg_dpo = sum((row.get("dpo_delta_reward") or 0) for row in records) / len(records)
    return {"processed": len(records), "average_reward": avg_reward, "dpo_delta_avg": avg_dpo}


def _collect_params(overrides: Optional[Mapping[str, Any]] = None) -> EvaluatorConfig:
    """Hydrate an EvaluatorConfig from Transformer Lab params and overrides."""
    params: Dict[str, Any] = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    if overrides:
        params.update(overrides)
    system_prompt_path = Path(params.get("system_prompt_path", DEFAULT_SYSTEM_PROMPT_PATH))
    system_prompt = load_prompt(system_prompt_path, fallback="Return factual snippets that verify or refute the claim.")
    reward_artifact = params.get("reward_artifact_path")
    artifact_path = Path(reward_artifact) if reward_artifact else None
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
        enable_dpo_reward=bool(params.get("enable_dpo_reward", False)),
        dpo_beta=float(params.get("dpo_beta", 0.1)),
        preference_reference_field=str(params.get("preference_reference_field", "preferred_answer")),
        reward_artifact_path=artifact_path,
    )


@tlab_trainer.job_wrapper()
def grok_search_evaluator(**overrides):
    """Entry point for Transformer Lab integration and direct CLI usage."""
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
