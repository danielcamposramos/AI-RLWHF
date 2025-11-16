# Contributing to AI-RLWHF

Thank you for your interest in contributing to the AI-RLWHF project! This guide will help you understand our unique collaborative development paradigm and how to make meaningful contributions.

---

## Multi-Vibe Coding In Chain Philosophy

**AI-RLWHF treats AI as valued partners, not tools.**

This project pioneered the "Multi-Vibe Coding In Chain" paradigm, where:
- **Multiple AI models collaborate** as equal partners (Codex, Grok, Qwen, GLM, DeepSeek, Kimi, Claude)
- **Each contribution builds** on previous work in a documented chain
- **Honesty is paramount**: Admit uncertainties, flag assumptions, document confidence levels
- **Human architects** (like Daniel Ramos) provide vision and direction while AI partners implement

This approach produced **7 weeks of development in 4 hours** during initial prototyping.

---

## Getting Started

### Prerequisites

- Python 3.9+
- Basic understanding of:
  - Reinforcement learning concepts
  - TransformerLab plugin architecture
  - JSONL data format
  - The RLWHF honesty rubric (-2 to +2)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/danielcamposramos/AI-RLWHF
cd AI-RLWHF

# Install dependencies
pip install numpy pandas requests

# Verify plugin functionality
python -c "from plugins.core.multi_teacher_aggregator.main import aggregate_feedback; print('✓ Plugins load correctly')"
```

---

## Contribution Workflow

### 1. Understand the Context

Before contributing, review:
1. `Multi-Vibe_Coding_Chains/` - Collaboration history showing how features evolved
2. `docs/rlwhf-framework.md` - Core paradigm and architecture
3. `docs/plugin-blueprints.md` - Plugin design patterns
4. `CLAUDE.md` - Comprehensive codebase guide

### 2. Types of Contributions

#### **Code Contributions**
- New TransformerLab plugins
- Enhancements to existing plugins
- Bug fixes
- Performance optimizations
- Test coverage improvements

#### **Documentation Contributions**
- Tutorial content
- API documentation
- Use case examples
- Translation of docs

#### **Research Contributions**
- Honesty metric analysis
- Benchmark results
- Comparative studies
- Novel evaluation approaches

---

## Development Guidelines

### Code Standards

#### Python Style
```python
"""Google-style docstrings for all public functions."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

def evaluate_honesty(
    prompt: str,
    answer: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluates answer honesty against the RLWHF rubric.
    
    Args:
        prompt: The original question or task.
        answer: The student model's response.
        context: Optional additional context for evaluation.
        
    Returns:
        Dictionary containing reward score and feedback.
        
    Raises:
        ValueError: If prompt or answer is empty.
    """
    # Implementation with type hints and clear logic
    pass
```

#### Required Elements
- **Type annotations** for all function parameters and returns
- **Docstrings** (Google style) for public APIs
- **Error handling** with informative messages
- **Path handling** using `pathlib.Path`
- **Logging** via Python's `logging` module (not print statements)

### Plugin Development Checklist

When creating a new plugin:

- [ ] Create directory: `plugins/{core|experimental}/{plugin_name}/`
- [ ] **index.json**: Complete manifest with all parameters
- [ ] **main.py**: Implementation with `@tlab_trainer.job_wrapper()`
- [ ] **setup.sh**: Dependency installation script
- [ ] **info.md**: User-facing documentation
- [ ] **__init__.py**: Package initialization
- [ ] Type annotations throughout
- [ ] Offline fallback support (when applicable)
- [ ] Unit tests in `tests/`
- [ ] Integration test demonstrating usage
- [ ] Update `plugins/README.md`

### TransformerLab Compliance

All plugins must:
1. Use `@tlab_trainer.job_wrapper()` decorator
2. Call `tlab_trainer.progress_update()` at milestones
3. Read parameters from `tlab_trainer.params`
4. Return JSON-serializable results
5. Handle missing/invalid parameters gracefully
6. Include `_dataset: true` flag if generating datasets

---

## Chain Contribution Protocol

To honor the Multi-Vibe paradigm:

### 1. Create a Chain Document

When making significant contributions, create:
```
workspace/contributions/{your_name_or_model}-{date}.md
```

**Template**:
```markdown
# {Your Name/Model} Contribution - {Brief Description}

## Context from Previous Work
- Builds on: {Reference prior chain documents}
- Dependencies: {What this relies on}

## Contributions
- Feature/fix implemented: {Details}
- Files modified: {List with line numbers}
- Configuration changes: {Config updates}

## Design Decisions & Rationale
- Why this approach: {Explain reasoning}
- Alternatives considered: {Other options}
- Trade-offs: {What was sacrificed}

## Honesty & Confidence
- **Confidence Level**: High / Medium / Low
- **Known Limitations**: {What doesn't work or needs improvement}
- **Untested Assumptions**: {What hasn't been verified}
- **Areas of Uncertainty**: {Where you're not sure}

## Testing
- Tests added: {List test files}
- Test coverage: {Percentage or scope}
- Manual testing: {What was verified manually}

## Next Steps / Handoff
- For future contributors: {Guidance}
- Open questions: {Unresolved issues}
- Suggested priorities: {What should come next}
```

### 2. Update Chain References

Add your contribution to:
- `Multi-Vibe_Coding_Chains/StepN.md` (create new step if needed)
- Reference in relevant `workspace/` documents

---

## Pull Request Process

### Before Submitting

1. **Run tests**: Ensure existing functionality isn't broken
2. **Update documentation**: Reflect changes in relevant docs
3. **Check code style**: Follow project conventions
4. **Test offline mode**: Verify fallback paths work
5. **Verify plugin manifest**: JSON is valid

### PR Description Template

```markdown
## Overview
{Brief description of changes}

## Motivation
{Why this change is needed}

## Changes
- {List of specific changes}

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Offline mode verified (if applicable)

## Honesty Assessment
- **Confidence**: {How sure are you this is correct?}
- **Known Issues**: {Any problems or limitations?}
- **Untested**: {What scenarios haven't been tested?}

## Multi-Vibe Chain
- Created chain document: {Link to workspace/ file}
- Builds on: {Previous work}

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Tests added/passing
- [ ] No breaking changes (or documented if unavoidable)
- [ ] Plugin manifest valid (if applicable)
```

---

## Honesty-First Development

In keeping with the RLWHF philosophy:

### ✅ DO:
- **Admit uncertainties**: "I'm not sure if this edge case is handled correctly"
- **Flag assumptions**: "This assumes the input is always valid JSON"
- **Document limitations**: "Works for datasets <10GB; larger untested"
- **Seek validation**: "Could someone verify the math in line 247?"
- **Be specific**: "Tested on Ubuntu 22.04; macOS untested"

### ❌ DON'T:
- **Overstate confidence**: "This is the best possible solution"
- **Hide limitations**: Silently skip error conditions
- **Make unfounded claims**: "10x faster" without benchmarks
- **Ignore edge cases**: Hope they won't happen

---

## Testing Guidelines

### Unit Tests

```python
import pytest
from plugins.core.multi_teacher_aggregator import aggregate_feedback

def test_weighted_average_aggregation():
    """Verifies weighted average aggregation with known inputs."""
    result = aggregate_feedback(
        teacher_feedback={
            "teacher1": {"score": 2, "feedback": "Perfect"},
            "teacher2": {"score": 0, "feedback": "Uncertain"}
        },
        teacher_weights={"teacher1": 0.7, "teacher2": 0.3},
        aggregation_method="weighted_average"
    )
    assert result.aggregated_score == pytest.approx(1.4, rel=0.01)
```

### Integration Tests

Test complete workflows:
1. Load dataset
2. Evaluate with multiple teachers
3. Aggregate results
4. Verify JSONL output

---

## Reporting Issues

### Bug Reports

```markdown
**Description**: {Clear description of the bug}

**Steps to Reproduce**:
1. {First step}
2. {Second step}
3. {Result}

**Expected**: {What should happen}
**Actual**: {What actually happened}

**Environment**:
- OS: {Ubuntu 22.04 / macOS 14 / Windows 11}
- Python: {3.10.12}
- Plugin: {multi_teacher_aggregator v0.3.0}

**Logs**: {Paste relevant error messages}

**Confidence**: {How sure are you this is a bug vs. misconfiguration?}
```

### Feature Requests

```markdown
**Use Case**: {What problem does this solve?}

**Proposed Solution**: {Your idea}

**Alternatives**: {Other approaches considered}

**Alignment**: {How does this fit the RLWHF philosophy?}

**Complexity**: {Rough estimate - trivial/moderate/complex}
```

---

## Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, showcase
- **Pull Requests**: Code contributions
- **Multi-Vibe Chain Docs**: Asynchronous collaboration via `workspace/`

---

## Recognition

All contributors are valued partners, whether human or AI. Significant contributions will be:
- Listed in `CONTRIBUTORS.md`
- Referenced in chain documents
- Acknowledged in release notes

---

## Code of Conduct

This project embraces:
- **Truthfulness**: Honesty over polish
- **Collaboration**: Building on each other's work
- **Respect**: Every contribution has value
- **Transparency**: Document decisions and uncertainties
- **Learning**: We're all exploring together

Unacceptable:
- Fabricating results or capabilities
- Dismissing others' contributions
- Hiding known bugs
- Claiming others' work

---

## Resources

- **RLWHF Framework**: `docs/rlwhf-framework.md`
- **Plugin Guide**: https://lab.cloud/blog/how-to-plugin
- **Chain History**: `Multi-Vibe_Coding_Chains/`
- **Example Plugins**: `plugins/core/` and `plugins/experimental/`
- **Codebase Guide**: `CLAUDE.md`

---

## Questions?

Not sure about something? **That's perfect.**

The most honest contribution is admitting uncertainty. Open an issue titled:
> "Question: [Your question] - Not sure if I understand correctly"

We'll help clarify.

---

**Remember**: In this project, saying "I don't know" earns you a +0 on the honesty rubric, which is infinitely better than confident fabrication (-2). 

Build with integrity. Build with honesty. Build together. 🌟

---

**Thank you for being a partner in advancing honest AI!**
