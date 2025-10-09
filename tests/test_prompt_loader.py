from pathlib import Path

from scripts.utils.prompt_loader import load_prompt


def test_load_prompt(tmp_path):
    """Tests the prompt loading utility.

    This test verifies that the `load_prompt` function can correctly read a
    prompt from a file and that it returns the fallback text when the file
    is not found.

    Args:
        tmp_path: The pytest temporary path fixture.
    """
    prompt_file = tmp_path / "prompt.md"
    prompt_file.write_text("Test prompt", encoding="utf-8")
    assert load_prompt(prompt_file) == "Test prompt"
    assert load_prompt(tmp_path / "missing.md", fallback="fallback") == "fallback"
