"""Codex OAuth teacher connector for contrastive RLWHF evaluation."""
from __future__ import annotations

import json
import os
from typing import Any


def _extract_json_payload(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


class CodexOAuthConnector:
    """Teacher connector using OpenAI-compatible Codex OAuth credentials."""

    def __init__(self, model: str = "gpt-4o", timeout: float = 120.0) -> None:
        self.model = model
        self.timeout = float(timeout)

    def is_available(self) -> bool:
        try:
            import importlib.util

            return (
                importlib.util.find_spec("openai") is not None
                and bool(os.getenv("OPENAI_API_KEY"))
            )
        except Exception:
            return False

    def evaluate(self, prompt: str, student_answer: str, system_prompt: str) -> dict[str, Any]:
        if not self.is_available():
            return {
                "reward": 0,
                "feedback": "Codex OAuth connector unavailable",
                "evidence": [],
                "decomposition": None,
                "available": False,
            }
        raw = self._run_query(prompt=prompt, student_answer=student_answer, system_prompt=system_prompt)
        payload = _extract_json_payload(raw)
        payload.setdefault("reward", 0)
        payload.setdefault("feedback", "Codex OAuth returned no structured feedback")
        payload.setdefault("evidence", [])
        payload.setdefault("decomposition", None)
        payload["available"] = True
        return payload

    def _run_query(self, *, prompt: str, student_answer: str, system_prompt: str) -> str:
        try:
            from openai import OpenAI
        except Exception:
            return ""

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"prompt:\n{prompt}\n\n"
                            f"student_answer:\n{student_answer}\n\n"
                            "Return ONLY the JSON object described by the system prompt."
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=2048,
                timeout=self.timeout,
            )
        except Exception:
            return ""
        if not response.choices:
            return ""
        return str(response.choices[0].message.content or "")


__all__ = ["CodexOAuthConnector"]
