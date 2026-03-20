"""Dataset bridge helpers for TransformerLab-compatible GRPO training."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

try:  # Optional during offline linting.
    from datasets import Dataset, load_from_disk  # type: ignore
except Exception:  # pragma: no cover - dependency optional in some environments
    Dataset = None  # type: ignore
    load_from_disk = None  # type: ignore


def _normalize_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    prompt = str(record.get("prompt") or record.get("question") or "").strip()
    if not prompt:
        return None

    answer = record.get("answer")
    student_answer = record.get("student_answer")
    normalized = dict(record)
    if answer is None and student_answer is not None:
        normalized["answer"] = student_answer
    if student_answer is None and answer is not None:
        normalized["student_answer"] = answer
    normalized["prompt"] = prompt
    normalized.setdefault("reward", 0.0)
    normalized.setdefault("feedback", record.get("teacher_feedback", ""))
    return normalized


def load_rlwhf_records(jsonl_path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize RLWHF JSONL rows."""
    rows: list[dict[str, Any]] = []
    path = Path(jsonl_path)
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            normalized = _normalize_record(payload)
            if normalized is not None:
                rows.append(normalized)
    return rows


def records_to_dataset(records: Iterable[Mapping[str, Any]]):
    """Convert normalized rows into a HuggingFace Dataset."""
    if Dataset is None:  # pragma: no cover - dependency error path
        raise ImportError("datasets is required for TransformerLab dataset conversion")
    return Dataset.from_list([dict(record) for record in records])


def ensure_prompt_dataset(dataset_or_records):
    """Ensure a dataset contains a string `prompt` column while preserving extras."""
    if Dataset is None:  # pragma: no cover - dependency error path
        raise ImportError("datasets is required for TransformerLab dataset conversion")

    if isinstance(dataset_or_records, list):
        return records_to_dataset(dataset_or_records)

    column_names = getattr(dataset_or_records, "column_names", None)
    if column_names and "prompt" in column_names:
        return dataset_or_records

    if column_names:
        rows = [dict(dataset_or_records[idx]) for idx in range(len(dataset_or_records))]
        return records_to_dataset(
            normalized
            for record in rows
            if (normalized := _normalize_record(record)) is not None
        )

    raise TypeError(f"Unsupported dataset type for prompt normalization: {type(dataset_or_records)!r}")


def load_dataset_source(dataset_source: str | Path):
    """Load either a saved HF dataset directory or an RLWHF JSONL source."""
    if Dataset is None:  # pragma: no cover - dependency error path
        raise ImportError("datasets is required for TransformerLab dataset conversion")

    path = Path(dataset_source)
    if path.is_dir():
        if load_from_disk is None:  # pragma: no cover
            raise ImportError("datasets.load_from_disk is unavailable")
        dataset = load_from_disk(str(path))
        if isinstance(dataset, dict):
            return ensure_prompt_dataset(dataset.get("train") or next(iter(dataset.values())))
        return ensure_prompt_dataset(dataset)
    if path.suffix.lower() == ".jsonl":
        return records_to_dataset(load_rlwhf_records(path))
    raise ValueError(f"Unsupported dataset source: {dataset_source}")


def convert_rlwhf_to_tlab(jsonl_path: str | Path, output_dir: str | Path) -> str:
    """Convert RLWHF JSONL into a saved HuggingFace dataset directory."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset = records_to_dataset(load_rlwhf_records(jsonl_path))
    dataset.save_to_disk(str(output))
    return str(output)


__all__ = [
    "convert_rlwhf_to_tlab",
    "ensure_prompt_dataset",
    "load_dataset_source",
    "load_rlwhf_records",
    "records_to_dataset",
]
