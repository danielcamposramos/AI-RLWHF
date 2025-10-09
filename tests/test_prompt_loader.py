from pathlib import Path

from scripts.utils.prompt_loader import load_prompt


def test_load_prompt(tmp_path):
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Test prompt", encoding="utf-8")
    assert load_prompt(prompt_file) == "Test prompt"
    assert load_prompt(tmp_path / "missing.md", fallback="fallback") == "fallback"
