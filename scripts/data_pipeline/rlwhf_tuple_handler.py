"""Utilities for constructing RLWHF tuples from workspace artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from scripts.data_pipeline.extended_metadata_handler import ExtendedMetadataHandler

TupleDict = Dict[str, object]


class RLWHFTupleHandler:
    """Parse workspace traces and generate RLWHF JSONL tuples."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    def __init__(self, metadata_handler: Optional[ExtendedMetadataHandler] = None) -> None:
        self.metadata_handler = metadata_handler or ExtendedMetadataHandler()

    def process_workspace_logs(self, workspace_path: str | Path = "workspace") -> List[TupleDict]:
        base_path = Path(workspace_path)
        tuples: List[TupleDict] = []
        if not base_path.exists():
            return tuples
        for file_path in base_path.rglob("*"):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            tuples.extend(self._read_json_records(file_path))
        return tuples

    def create_training_dataset(
        self,
        tuple_list: Iterable[Mapping[str, object]],
        output_path: str | Path,
        ensure_metadata: bool = True,
    ) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for entry in tuple_list:
                record = dict(entry)
                if ensure_metadata:
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, Mapping):
                        metadata = {}
                    record["metadata"] = self.metadata_handler.extend_metadata(dict(metadata))
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return output

    def _read_json_records(self, path: Path) -> List[TupleDict]:
        records: List[TupleDict] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                record = self._normalize_record(payload)
                if record:
                    records.append(record)
        return records

    def _normalize_record(self, payload: Mapping[str, object]) -> Optional[TupleDict]:
        prompt = payload.get("prompt") or payload.get("instruction")
        answer = payload.get("student_answer") or payload.get("answer")
        feedback = payload.get("teacher_feedback") or payload.get("feedback", "")
        reward = payload.get("reward", 0)
        if not prompt or not answer:
            return None
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        normalized: MutableMapping[str, object] = {
            "prompt": prompt,
            "student_answer": answer,
            "feedback": feedback,
            "reward": reward,
            "metadata": dict(metadata),
        }
        normalized["metadata"] = self.metadata_handler.extend_metadata(normalized["metadata"])
        return dict(normalized)


__all__ = ["RLWHFTupleHandler"]
