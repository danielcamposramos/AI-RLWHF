"""Claude Agent SDK teacher connector for contrastive RLWHF evaluation."""
from __future__ import annotations

import json
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
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}
    return {}


class ClaudeAgentSDKConnector:
    """Teacher connector using Claude Agent SDK subscription auth."""

    def __init__(self, model: str = "claude-sonnet-4-6", timeout: float = 120.0) -> None:
        self.model = model
        self.timeout = float(timeout)

    def is_available(self) -> bool:
        try:
            import importlib.util

            return importlib.util.find_spec("claude_agent_sdk") is not None
        except Exception:
            return False

    def evaluate(self, prompt: str, student_answer: str, system_prompt: str) -> dict[str, Any]:
        """Return teacher evaluation payload with decomposition when available."""
        if not self.is_available():
            return {
                "reward": 0,
                "feedback": "Claude Agent SDK connector unavailable",
                "evidence": [],
                "decomposition": None,
                "available": False,
            }
        raw = self._run_query(prompt=prompt, student_answer=student_answer, system_prompt=system_prompt)
        payload = _extract_json_payload(raw)
        payload.setdefault("reward", 0)
        payload.setdefault("feedback", "Claude Agent SDK returned no structured feedback")
        payload.setdefault("evidence", [])
        payload.setdefault("decomposition", None)
        payload["available"] = True
        return payload

    def _run_query(self, *, prompt: str, student_answer: str, system_prompt: str) -> str:
        try:
            import anyio
            from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query as agent_query
        except Exception:
            return ""

        user_message = (
            f"prompt:\n{prompt}\n\n"
            f"student_answer:\n{student_answer}\n\n"
            "Return ONLY the JSON object described by the system prompt."
        )

        async def _do_query() -> str:
            result_text = ""
            options = ClaudeAgentOptions(
                system_prompt=system_prompt,
                allowed_tools=[],
                max_turns=1,
                model=self.model,
            )
            async for message in agent_query(prompt=user_message, options=options):
                if isinstance(message, ResultMessage):
                    result_text = str(message.result or "")
            return result_text

        try:
            return anyio.run(_do_query)
        except Exception:
            return ""


__all__ = ["ClaudeAgentSDKConnector"]
