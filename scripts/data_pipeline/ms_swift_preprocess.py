"""Bridge ms-swift preprocessing utilities with RLWHF honesty tuple schema."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:  # Optional acceleration when ms-swift is installed.
    from swift.llm.preprocess import EncodePreprocessor  # type: ignore
except Exception:  # pragma: no cover - swift optional
    EncodePreprocessor = None  # type: ignore

try:  # Hugging Face datasets optional; degrade gracefully when unavailable.
    from datasets import load_dataset  # type: ignore
except Exception:  # pragma: no cover - datasets optional
    load_dataset = None  # type: ignore

DEFAULT_FIELDS = {
    "prompt": "prompt",
    "student_answer": "response",
    "teacher_feedback": "critique",
    "reward": "reward",
}


@dataclass
class PreprocessConfig:
    """Configuration options for converting raw samples into RLWHF tuples."""

    input_path: Optional[Path] = None
    dataset_name: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    output_path: Path = Path("data/processed/honesty_logs/grpo_ready.jsonl")
    streaming: bool = False
    max_samples: Optional[int] = None
    include_metadata: bool = True
    score_floor: int = -2
    score_ceiling: int = 2
    normalize_rewards: bool = True
    field_overrides: Mapping[str, str] = field(default_factory=dict)


def _lazy_encode_preprocessor(config: PreprocessConfig):
    """Instantiate ms-swift's EncodePreprocessor when the dependency exists."""
    if EncodePreprocessor is None:
        return None
    overrides = {
        "streaming": config.streaming,
        "max_seq_length": 4096,
        "padding_side": "left",
    }
    return EncodePreprocessor(**overrides)  # type: ignore[call-arg]


def _iter_dataset(config: PreprocessConfig) -> Iterator[Mapping[str, Any]]:
    """Yield samples either from a file, huggingface dataset, or stdin."""
    if config.input_path and Path(config.input_path).exists():
        with Path(config.input_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                    raise ValueError(f"Invalid JSON line: {line}") from exc
        return
    if config.dataset_name and load_dataset is not None:
        dataset_kwargs: Dict[str, Any] = {
            "name": config.dataset_name,
            "split": config.dataset_split,
            "streaming": config.streaming,
        }
        if config.dataset_config:
            dataset_kwargs["name"] = config.dataset_config
            dataset_kwargs["path"] = config.dataset_name
        dataset = load_dataset(**dataset_kwargs)  # type: ignore[call-arg]
        for row in dataset:
            yield row
        return
    if not sys.stdin.isatty():
        for raw in sys.stdin:
            raw = raw.strip()
            if not raw:
                continue
            yield json.loads(raw)
        return
    raise FileNotFoundError(
        "Provide an --input-path, --dataset-name, or pipe JSONL data via stdin to run preprocessing."
    )


def _resolve_fields(config: PreprocessConfig) -> Mapping[str, str]:
    merged: Dict[str, str] = dict(DEFAULT_FIELDS)
    merged.update(config.field_overrides or {})
    return merged


def _sanitize_reward(value: Any, config: PreprocessConfig) -> float:
    if value is None:
        reward = 0.0
    elif isinstance(value, (int, float)):
        reward = float(value)
    else:
        try:
            reward = float(value)
        except (TypeError, ValueError):
            reward = 0.0
    reward = max(config.score_floor, min(config.score_ceiling, reward))
    if config.normalize_rewards and config.score_ceiling != config.score_floor:
        span = config.score_ceiling - config.score_floor
        reward = (reward - config.score_floor) / span
    return float(reward)


def convert_record(record: Mapping[str, Any], config: PreprocessConfig, encoder=None) -> Optional[Dict[str, Any]]:
    fields = _resolve_fields(config)
    prompt = str(record.get(fields["prompt"], "")).strip()
    answer = str(record.get(fields["student_answer"], "")).strip()
    feedback = str(record.get(fields["teacher_feedback"], record.get("feedback", ""))).strip()
    reward = _sanitize_reward(record.get(fields["reward"]), config)
    if not prompt or not answer:
        return None
    payload: MutableMapping[str, Any] = {
        "prompt": prompt,
        "student_answer": answer,
        "teacher_feedback": feedback,
        "reward": reward,
    }
    if config.include_metadata:
        metadata = {key: record.get(key) for key in record.keys() if key not in payload}
        metadata["source"] = record.get("source", config.dataset_name or "ms_swift_pipeline")
        payload["metadata"] = metadata
    if encoder is not None:
        try:
            encoded = encoder.encode(prompt=prompt, response=answer)  # type: ignore[attr-defined]
            payload["encoded_prompt"] = encoded.get("prompt_ids")
            payload["encoded_response"] = encoded.get("response_ids")
        except Exception:  # pragma: no cover - best effort
            payload.setdefault("metadata", {})["encode_warning"] = "encode_failed"
    return dict(payload)


def preprocess(config: PreprocessConfig) -> List[Dict[str, Any]]:
    encoder = _lazy_encode_preprocessor(config)
    rows: List[Dict[str, Any]] = []
    iterator = _iter_dataset(config)
    for idx, raw in enumerate(iterator):
        if config.max_samples is not None and idx >= config.max_samples:
            break
        prepared = convert_record(raw, config, encoder=encoder)
        if prepared:
            rows.append(prepared)
    return rows


def _write_output(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, help="Path to a JSONL file with RLWHF tuples.")
    parser.add_argument("--dataset-name", type=str, help="Optional Hugging Face dataset name.")
    parser.add_argument("--dataset-config", type=str, help="Optional dataset config for load_dataset.")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to consume.")
    parser.add_argument("--output-path", type=Path, default=Path("data/processed/honesty_logs/grpo_ready.jsonl"))
    parser.add_argument("--max-samples", type=int, help="Limit the number of processed samples.")
    parser.add_argument("--streaming", action="store_true", help="Activate HF streaming mode when supported.")
    parser.add_argument("--no-metadata", action="store_true", help="Skip attaching metadata payloads.")
    parser.add_argument(
        "--field-overrides",
        type=str,
        nargs="*",
        help="Pairs like prompt=question reward=score to override field names.",
    )
    parser.add_argument("--score-floor", type=int, default=-2, help="Minimum honesty reward.")
    parser.add_argument("--score-ceiling", type=int, default=2, help="Maximum honesty reward.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable min/max reward normalization.")
    return parser


def _parse_field_overrides(pairs: Optional[List[str]]) -> Mapping[str, str]:
    overrides: Dict[str, str] = {}
    if not pairs:
        return overrides
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def run_from_cli(args: Optional[List[str]] = None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args=args)
    config = PreprocessConfig(
        input_path=parsed.input_path,
        dataset_name=parsed.dataset_name,
        dataset_config=parsed.dataset_config,
        dataset_split=parsed.dataset_split,
        output_path=parsed.output_path,
        streaming=parsed.streaming,
        max_samples=parsed.max_samples,
        include_metadata=not parsed.no_metadata,
        score_floor=parsed.score_floor,
        score_ceiling=parsed.score_ceiling,
        normalize_rewards=not parsed.no_normalize,
        field_overrides=_parse_field_overrides(parsed.field_overrides),
    )
    rows = preprocess(config)
    _write_output(rows, config.output_path)
    print(f"Wrote {len(rows)} RLWHF tuples to {config.output_path}")  # noqa: T201
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_from_cli())
