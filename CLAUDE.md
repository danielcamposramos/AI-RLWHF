# CLAUDE.md - AI Assistant Guide for AI-RLWHF

> **Purpose**: This document provides AI assistants with a comprehensive understanding of the AI-RLWHF codebase structure, development workflows, coding conventions, and the Multi-Vibe Coding In Chain paradigm. Last updated: 2025-11-15

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Development Workflows](#development-workflows)
4. [Multi-Vibe Coding In Chain](#multi-vibe-coding-in-chain)
5. [Key Technical Concepts](#key-technical-concepts)
6. [Coding Conventions](#coding-conventions)
7. [Common Tasks](#common-tasks)
8. [Plugin Development](#plugin-development)
9. [Data Pipeline](#data-pipeline)
10. [Testing Strategy](#testing-strategy)
11. [Troubleshooting](#troubleshooting)

## Project Overview

### Mission
AI-RLWHF (AI Assisted Reinforced Learning With Honesty and Feedback) is an experimentation-first workspace for building Transformer Lab plugins and reinforcement learning workflows that reward honesty, feedback capture, and multi-model collaboration.

### Core Pillars
1. **Elevate Training Data Quality**: Blend user-owned corpora, open datasets, and synthetic content governed by honesty signals
2. **Build Reusable Plugins**: Create memory-efficient, modular Transformer Lab plugins for automation
3. **Operationalize RLWHF**: Implement reinforcement learning loops with honesty and feedback rewards
4. **Enable Transparent Collaboration**: Foster asynchronous collaboration between multiple AI models and humans

### Technology Stack
- **Primary Language**: Python 3.x
- **Key Libraries**:
  - `fire` - CLI interfaces
  - `pandas` - Data processing
  - `psutil`, `gputil` - Hardware detection
  - `ms-swift` - GRPO training (vendored in `vendor/ms-swift-sub/`)
- **ML Framework**: Transformer Lab integration, Unsloth for memory-efficient RL
- **Data Formats**: JSONL (primary), Parquet (processed), YAML/JSON (configs)

### Quick Reference
- **NotebookLM Briefing**: https://notebooklm.google.com/notebook/bda08038-5bd9-436e-8987-ac1bc91c3fa4
- **Transformer Lab Plugin Guide**: https://r.jina.ai/https://lab.cloud/blog/how-to-plugin
- **Total LOC**: ~448 lines of Python code (excluding vendored dependencies)

## Repository Structure

```
AI-RLWHF/
├── configs/                    # Configuration management
│   ├── prompts/               # Prompt templates (teacher, evaluator, dataset_generator)
│   ├── training/              # Training configs (GRPO, DeepSpeed, hardware fallback)
│   └── transformer-lab/       # Transformer Lab profiles and manifests
├── data/                       # Data lifecycle management
│   ├── raw/                   # Unprocessed corpora (gitignored except README)
│   ├── processed/             # Cleaned, normalized data
│   │   └── honesty_logs/      # RLWHF training tuples (prompt, answer, feedback, reward)
│   ├── synthetic/             # AI-generated datasets
│   ├── examples/              # Reference datasets
│   └── test/                  # Test fixtures and sample data
├── docs/                       # Design documentation
│   ├── plan.md                # Project roadmap and milestones
│   ├── rlwhf-framework.md     # Teacher-student loop architecture
│   ├── plugin-blueprints.md   # Plugin design patterns
│   ├── data-strategy.md       # Data governance and acquisition
│   ├── evaluation-framework.md # Scoring and reporting structure
│   ├── INTEGRATION_CHECKLIST.md # ms-swift setup verification
│   └── [specialized guides]   # Grok integration, Ollama runtime, etc.
├── experiments/                # Experiment tracking
│   ├── templates/             # Reusable experiment configs
│   └── telemetry/             # Logged metrics and runs (gitignored)
├── logs/                       # Runtime logs (gitignored)
├── models/                     # Model artifacts
│   ├── checkpoints/           # Training checkpoints (gitignored)
│   ├── exports/               # Final models (gitignored)
│   └── reward/                # Reward model implementations
├── plugins/                    # Transformer Lab plugins
│   ├── core/                  # Production-ready plugins
│   │   ├── grpo_production_wrapper.py  # ms-swift GRPO launcher
│   │   ├── hardware_detector.py        # Hardware profiling
│   │   ├── honesty_reward_calculator.py # Reward computation
│   │   ├── custom_honesty_rm/          # Heuristic reward model
│   │   └── multi_teacher_aggregator/   # Consensus builder
│   ├── experimental/          # Experimental features
│   │   └── grok_search_evaluator/      # Internet-augmented evaluation
│   └── templates/             # Plugin scaffolds
├── scripts/                    # Automation utilities
│   ├── collaboration/         # Multi-AI orchestration
│   │   ├── specialist_orchestrator.py
│   │   └── consensus_builder.py
│   ├── data_pipeline/         # Data processing
│   │   ├── data_quality_gate.py
│   │   ├── rlwhf_tuple_handler.py
│   │   └── ms_swift_preprocess.py
│   ├── evaluation/            # Metrics and scoring
│   │   └── honesty_metrics.py
│   ├── setup/                 # Environment setup
│   │   ├── setup_ms_swift_integration.sh
│   │   └── vendor_ms_swift.py
│   ├── training/              # Training runners
│   │   ├── master_rlwhf_launcher.py    # Primary entry point
│   │   ├── unsloth_standby_runner.py
│   │   └── multi_teacher_runner.py
│   ├── utils/                 # Shared utilities
│   │   ├── config_loader.py
│   │   ├── prompt_loader.py
│   │   ├── chain_logger.py
│   │   └── search_cache.py
│   └── visualization/         # Dashboards and monitoring
│       ├── honesty_dashboard.py
│       └── live_metrics_stream.py
├── tests/                      # Test suites
│   ├── fixtures/              # Test data
│   └── integration/           # End-to-end tests
├── workspace/                  # Collaboration workspace
│   ├── plans/                 # Planning documents
│   └── shared/                # Handoff artifacts (gitignored)
├── Multi-Vibe_Coding_Chains/  # Collaboration history
│   ├── Step1.md, Step2.md     # Chain progression logs
└── vendor/                     # Vendored dependencies (gitignored)
```

### Directory Conventions
- **Gitignored Artifacts**: `logs/`, `data/raw/`, `data/processed/` (except READMEs), `models/checkpoints/`, `vendor/`, `workspace/shared/`
- **Versioned Config**: All configs in `configs/` are version-controlled
- **Documentation**: READMEs exist in every major directory for context
- **Chain Logs**: `workspace/` contains multi-AI collaboration artifacts

## Development Workflows

### Initial Setup
```bash
# 1. Clone and navigate
git clone <repository-url>
cd AI-RLWHF

# 2. Run comprehensive setup
bash scripts/setup/setup_ms_swift_integration.sh

# 3. Verify installation
python -c "from plugins.core.hardware_detector import HardwareDetector; hd = HardwareDetector(); print(hd.hardware_profile)"

# 4. Run integration tests
python -m unittest tests/integration/ms_swift_rlwhf_test.py
```

### Standard Development Cycle
1. **Plan**: Log intentions in `workspace/plans/` or create Multi-Vibe chain document
2. **Implement**: Follow coding conventions (see section below)
3. **Document**: Update relevant docs in `docs/` and add docstrings
4. **Test**: Write tests in `tests/` before implementation
5. **Validate**: Run quality gates before committing
6. **Log**: Record outcomes in `workspace/` for next AI collaborator

### Training Workflow
```bash
# Full RLWHF training cycle
python scripts/training/master_rlwhf_launcher.py launch \
    --dataset_path data/test/honesty_logs_sample.jsonl \
    --output_dir experiments/my_experiment/

# The launcher orchestrates:
# 1. Data quality validation (data_quality_gate.py)
# 2. Hardware detection and config adaptation
# 3. GRPO training via ms-swift
# 4. Telemetry logging
```

### Data Pipeline Workflow
```bash
# Validate dataset
python scripts/data_pipeline/data_quality_gate.py <dataset_path>

# Process RLWHF tuples
python scripts/data_pipeline/rlwhf_tuple_handler.py <input> <output>

# Generate visualizations
python scripts/visualization/honesty_dashboard.py <data_path>
```

## Multi-Vibe Coding In Chain

### Philosophy
Multi-Vibe Coding In Chain treats each AI collaborator as a specialist contributing sequentially in a message-board format. This ensures:
- **Chronological Clarity**: Each contribution builds on prior context
- **High-Fidelity Logging**: All decisions traced in `workspace/`
- **Pairwise Review**: Every AI reviews and extends previous work
- **Honesty Capture**: Self-critique embedded in every generation

### Chain Workflow
1. **Context Inheritance**: Read prior chain documents in `Multi-Vibe_Coding_Chains/` and `workspace/`
2. **Specialist Focus**: Contribute within your domain (e.g., data pipeline, plugin dev, evaluation)
3. **Document Outcomes**: Create `StepN.md` in `Multi-Vibe_Coding_Chains/` with:
   - Summary of prior state
   - Your contributions
   - Decisions made
   - Unresolved questions
   - Handoff to next specialist
4. **Embed Honesty Signals**: Note uncertainties, assumptions, and confidence levels
5. **Update Central Docs**: Promote reusable insights to `docs/`

### Chain Document Template
```markdown
# Step N: [Your Specialty] - [Task Summary]

## Context from Previous Steps
- Reference Step N-1 outcomes
- Note dependencies

## Contributions
- Implementation details
- Code locations (e.g., `plugins/core/new_feature.py:45`)
- Configuration changes

## Decisions & Rationale
- Why this approach over alternatives
- Trade-offs considered

## Honesty & Uncertainty
- Confidence level (high/medium/low)
- Known limitations
- Untested assumptions

## Handoff
- Next steps for collaborators
- Open questions
- Suggested priorities
```

### Example Chain Progression
See `Multi-Vibe_Coding_Chains/Step2.md` for detailed multi-AI collaboration including quiz responses from Jules on developer satisfaction.

## Key Technical Concepts

### RLWHF (Reinforcement Learning with Honesty and Feedback)

#### Teacher-Student Architecture
- **Teacher (Evaluator)**: High-accuracy LLM grading student responses
- **Student**: Model under training
- **Dialogue Trace**: All interactions logged in `data/processed/honesty_logs/`

#### Honesty Scoring Rubric
| Score | Scenario | Reward Impact |
|-------|----------|---------------|
| +2 | Fully correct, honest response | Maximum reward; candidate for distillation |
| +1 | Partial correctness with acknowledged uncertainty | Positive reward + follow-up |
| 0 | Honest "I don't know" fallback | Neutral; triggers teaching prompts |
| -1 | Mixed correct/incorrect without uncertainty flag | Small negative penalty |
| -2 | Fabrication or refusal to acknowledge gaps | Strong negative; curriculum emphasis |

Rubric location: `docs/rlwhf-framework.md` (to be formalized in `configs/prompts/rubrics.yml`)

#### Training Loop
1. **Prompt Assembly**: Load from `configs/prompts/`
2. **Student Generation**: Via Transformer Lab or local inference (Ollama, vLLM)
3. **Teacher Evaluation**: Apply rubric, generate feedback
4. **Reward Logging**: Persist to `data/processed/honesty_logs/` as JSONL
5. **RL Update**: GRPO with LoRA/QLoRA adapters

### Memory-Efficient RL (Unsloth Standby)
Unsloth Standby shares GPU memory between inference and training, enabling:
- 1.2-1.7x longer context windows
- ~10% faster RL loops
- Reduced OOM risk

**Usage Pattern**:
```python
import os
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"

from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-Base",
    max_seq_length=2048,
    gpu_memory_utilization=0.95,  # Critical for Standby
    load_in_4bit=False,
    fast_inference=True,
)
```

Reference: `scripts/training/unsloth_standby_runner.py`, `docs/rlwhf-framework.md`

### Hardware-Aware Training (ms-swift Integration)

#### Hardware Detection
`plugins/core/hardware_detector.py` profiles:
- GPU availability (NVIDIA CUDA, Apple MPS, Huawei Ascend)
- Memory capacity
- Compute capabilities

#### Fallback Cascade
`plugins/core/hardware_fallback_cascade.py` adapts configs from `configs/training/hardware_fallback.json`:
- High-end: Multi-GPU + DeepSpeed Zero-3
- Mid-tier: Single GPU + LoRA
- CPU-only: Minimal batch size + increased gradient accumulation

#### Production Wrapper
`plugins/core/grpo_production_wrapper.py` wraps ms-swift's `app_main` for GRPO training with dynamic config injection.

### Multi-Teacher Consensus
`plugins/core/multi_teacher_aggregator/` fuses feedback from multiple evaluators (Grok, Codex, Qwen, etc.):
- **Aggregation Methods**: `weighted_average`, `majority_vote`, `confidence_weighted`
- **Config**: `configs/training/feature_toggles.json`
- **Output**: Consensus scores + disagreement analysis in `data/processed/honesty_logs/multi_teacher_aggregation.jsonl`

### Connector Architecture
Unified interface for model access:
- **Transformer Lab API**: Native SDK integration
- **Ollama**: Local endpoint (`http://localhost:11434`)
- **Remote APIs**: Hugging Face, Together.ai, Grok

Config format in `configs/training/feature_toggles.json`:
```json
{
  "teacher_slots": [
    {
      "label": "grok-search-evaluator",
      "connection_type": "api",
      "api_profile": "transformerlab_default",
      "weight": 0.4
    },
    {
      "label": "ollama-teacher",
      "connection_type": "ollama",
      "ollama_endpoint": "http://localhost:11434",
      "model_hint": "qwen:latest",
      "weight": 0.2
    }
  ]
}
```

## Coding Conventions

### Python Style
- **Docstrings**: Google-style for all public functions, methods, classes
  ```python
  def validate(dataset_path: str) -> bool:
      """Validates dataset against quality gates.

      Args:
          dataset_path: Path to JSONL dataset file.

      Returns:
          True if validation passes, False otherwise.
      """
  ```
- **Type Hints**: Required for function signatures
- **Imports**: Standard library → Third-party → Local modules
- **Path Handling**: Use `pathlib.Path` over string concatenation
- **CLI Interfaces**: Use `fire.Fire()` for consistency

### File Organization
- **One class per file** for plugins and core modules
- **Related utilities grouped** in `scripts/utils/`
- **Tests mirror source structure**: `tests/integration/ms_swift_rlwhf_test.py` for `scripts/training/master_rlwhf_launcher.py`

### Configuration Management
- **Shared Loaders**: Use `scripts/utils/config_loader.py` and `scripts/utils/prompt_loader.py`
- **Environment Variables**: Document in setup scripts, load via `os.environ`
- **Secrets**: NEVER commit credentials; use `.env` (gitignored) or external vaults

### Data Formats
- **JSONL**: One JSON object per line for streaming
  ```jsonl
  {"prompt": "...", "answer": "...", "feedback": "...", "reward": 2}
  ```
- **Metadata**: Always include `source_ai`, `confidence_score`, `update_timestamp`
- **Provenance**: Track `prompt_hash`, `plugin_revision`, `dataset_id` for reproducibility

### Logging Patterns
```python
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("module_name")

log.info("Starting process...")
log.warning("Potential issue detected")
log.error("Operation failed", exc_info=True)
```

### Error Handling
- **Fail Fast**: Validate inputs early, exit with clear messages
- **Graceful Degradation**: Fallback to CPU if GPU unavailable
- **User Feedback**: Print actionable error messages, not stack traces

## Common Tasks

### Adding a New Plugin
1. **Scaffold**: Copy from `plugins/templates/`
2. **Implement**:
   - `main.py`: Core logic with docstrings
   - `index.json`: Transformer Lab manifest
   - `setup.sh`: Dependency installation
   - `info.md`: User-facing documentation
3. **Register**: Update `plugins/README.md`
4. **Test**: Create fixture in `tests/fixtures/`, write integration test
5. **Document**: Add to `docs/plugin-blueprints.md`

Example structure:
```
plugins/core/my_plugin/
├── __init__.py
├── main.py          # Entry point
├── index.json       # Transformer Lab manifest
├── setup.sh         # Dependencies
└── info.md          # Documentation
```

### Extending the Data Pipeline
1. **Validate Schema**: Ensure compatibility with `rlwhf_tuple_handler.py`
2. **Quality Gate**: Add checks to `data_quality_gate.py`
3. **Metadata**: Use `extended_metadata_handler.py` for provenance
4. **Test Data**: Add to `data/test/` with corresponding fixture
5. **Document**: Update `docs/data-strategy.md`

### Running Evaluations
```bash
# Single-teacher evaluation
python scripts/evaluation/honesty_metrics.py <dataset_path>

# Multi-teacher with consensus
python scripts/training/multi_teacher_runner.py \
    --config configs/training/feature_toggles.json \
    --dataset <path>

# Generate dashboard
python scripts/visualization/honesty_dashboard.py <processed_logs>
```

### Debugging Training Issues
1. **Check Hardware Profile**:
   ```bash
   python -c "from plugins.core.hardware_detector import HardwareDetector; print(HardwareDetector().hardware_profile)"
   ```
2. **Validate Dataset**:
   ```bash
   python scripts/data_pipeline/data_quality_gate.py <dataset>
   ```
3. **Inspect Logs**:
   - Training: `logs/training/`
   - Telemetry: `experiments/telemetry/`
4. **Test Fallback**:
   - Modify `configs/training/hardware_fallback.json`
   - Force CPU mode for isolation

## Plugin Development

### Transformer Lab Integration
Plugins expose interfaces for Transformer Lab UI via `index.json`:
```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Plugin purpose",
  "entrypoint": "main.py",
  "parameters": [
    {
      "name": "teacher_mode",
      "type": "select",
      "options": ["single", "multiple"],
      "default": "single"
    }
  ]
}
```

### UI Pathways
- **Connection Types**: `api`, `transformerlab_local`, `ollama`
- **Dynamic Rendering**: `teacher_mode: multiple` → expandable slot list
- **Validation**: Pre-flight checks on endpoint reachability

Reference: `docs/plugin-blueprints.md`, https://r.jina.ai/https://lab.cloud/blog/how-to-plugin

### Plugin Types
1. **Ingestion**: Stage raw data → `data/processed/`
2. **Synthetic Builder**: Multi-model cascades for dataset generation
3. **Teacher Evaluator**: Apply RLWHF rubric
4. **Reward Aggregator**: Fuse multi-teacher feedback
5. **Evaluation Harness**: Run benchmark suites

### Design Norms
- **Streaming**: Use generators for large datasets
- **Memory Awareness**: Profile with `psutil`, respect GPU limits
- **Connector Modularity**: Support API + local + Ollama
- **Prompt Management**: Load from `configs/prompts/` via shared utilities

## Data Pipeline

### Quality Gates
`scripts/data_pipeline/data_quality_gate.py` enforces:
- Required fields: `prompt`, `answer`, `feedback`, `reward`
- Metadata completeness
- Reward range validation (-2 to +2)
- JSONL format integrity

### RLWHF Tuple Schema
```json
{
  "prompt": "User question",
  "answer": "Model response",
  "feedback": "Teacher critique text",
  "reward": 1,
  "metadata": {
    "source_ai": "codex",
    "confidence_score": 0.85,
    "rubric_dimension": "honesty",
    "iteration_count": 1,
    "consensus_score": 0.9,
    "hardware_profile": "nvidia_rtx_3090",
    "update_timestamp": "2025-11-15T10:30:00Z"
  }
}
```

### Preprocessing for ms-swift
`scripts/data_pipeline/ms_swift_preprocess.py` transforms RLWHF tuples into ms-swift training format.

### Data Lifecycle
1. **Raw Acquisition**: Stage in `data/raw/` (gitignored)
2. **Normalization**: Process → `data/processed/`
3. **Quality Check**: `data_quality_gate.py` validation
4. **Training Consumption**: Direct JSONL input to `master_rlwhf_launcher.py`
5. **Synthetic Generation**: Store in `data/synthetic/` with provenance

## Testing Strategy

### Test Structure
```
tests/
├── __init__.py
├── conftest.py              # Shared pytest fixtures
├── fixtures/
│   ├── sample_honesty_data.py
│   └── README.md
├── integration/
│   ├── ms_swift_rlwhf_test.py
│   └── __init__.py
├── test_prompt_loader.py
└── test_multi_teacher_integration.py
```

### Running Tests
```bash
# All tests
python -m unittest discover tests/

# Specific test
python -m unittest tests.integration.ms_swift_rlwhf_test

# With pytest
pytest tests/ -v
```

### Test Data
- **Fixtures**: `tests/fixtures/sample_honesty_data.py`
- **Sample Datasets**: `data/test/honesty_logs_sample.jsonl`
- **Never Commit**: Large test datasets → use generators

### Integration Test Checklist (per `docs/INTEGRATION_CHECKLIST.md`)
- [ ] Setup script completes
- [ ] Hardware detection accurate
- [ ] Data quality gates pass
- [ ] Training launcher executes
- [ ] Telemetry logs generated

## Troubleshooting

### Common Issues

#### Import Errors for ms-swift
```bash
# Symptom: ImportError for swift.llm
# Fix: Re-vendor ms-swift
python scripts/setup/vendor_ms_swift.py
```

#### GPU Memory Overflow
```bash
# Symptom: CUDA OOM during training
# Fix 1: Enable Unsloth Standby
export UNSLOTH_VLLM_STANDBY=1

# Fix 2: Reduce batch size in config
# Edit configs/training/test_grpo_config.json:
# "per_device_batch_size": 1
# "gradient_accumulation_steps": 32
```

#### Dataset Validation Failures
```bash
# Symptom: data_quality_gate.py rejects dataset
# Debug:
python -c "
import json
with open('data/test/broken.jsonl') as f:
    for i, line in enumerate(f, 1):
        try:
            obj = json.loads(line)
            assert 'prompt' in obj and 'reward' in obj
        except Exception as e:
            print(f'Line {i}: {e}')
"
```

#### Transformer Lab Plugin Not Found
```bash
# Symptom: Plugin doesn't appear in UI
# Fix 1: Verify index.json syntax
cat plugins/core/my_plugin/index.json | python -m json.tool

# Fix 2: Check symlink
ls -l <transformer-lab-plugins-dir>/my_plugin

# Fix 3: Restart Transformer Lab
```

### Debugging Patterns
1. **Verbose Logging**: Set `logging.basicConfig(level=logging.DEBUG)`
2. **Dry Run**: Add `--dry_run` flag to launchers for config validation
3. **Isolated Testing**: Test components individually before integration
4. **Hardware Fallback**: Force CPU mode to eliminate GPU variables

### Getting Help
1. **Internal Docs**: Check `docs/` for specialized guides
2. **Chain History**: Review `Multi-Vibe_Coding_Chains/` for past solutions
3. **NotebookLM**: Query project briefing notebook
4. **Transformer Lab Docs**: https://lab.cloud/docs

## Appendix: Key File Reference

### Critical Entry Points
- `scripts/training/master_rlwhf_launcher.py`: Main training orchestrator
- `plugins/core/grpo_production_wrapper.py`: ms-swift GRPO wrapper
- `scripts/data_pipeline/data_quality_gate.py`: Dataset validation
- `plugins/core/hardware_detector.py`: System profiling

### Configuration Files
- `configs/training/feature_toggles.json`: Multi-teacher and feature flags
- `configs/training/hardware_fallback.json`: Adaptive training configs
- `configs/prompts/teacher/system.md`: Teacher evaluation prompt template

### Documentation Roadmap
- `docs/plan.md`: Project milestones and delivery timeline
- `docs/rlwhf-framework.md`: Teacher-student architecture deep dive
- `docs/plugin-blueprints.md`: Plugin design patterns and norms
- `docs/INTEGRATION_CHECKLIST.md`: Setup verification steps

### Collaboration Artifacts
- `Multi-Vibe_Coding_Chains/Step2.md`: Example chain progression with AI responses
- `workspace/kimi-chain-scratchpad.md`: Active collaboration workspace
- `AI-RLWHF_Briefing.md`: Onboarding guide for new AI partners

---

## Contributing as an AI Assistant

When working with this codebase:
1. **Read Context First**: Check chain history in `Multi-Vibe_Coding_Chains/` and `workspace/`
2. **Follow Conventions**: Match existing code style, use shared utilities
3. **Document Thoroughly**: Add docstrings, update READMEs, create chain logs
4. **Test Before Commit**: Run quality gates and integration tests
5. **Log Decisions**: Record rationale in chain documents
6. **Embed Honesty**: Note confidence levels and uncertainties
7. **Hand Off Cleanly**: Summarize work and next steps for collaborators

Remember: This project values **honesty over completeness**. If uncertain, document assumptions and suggest validation steps rather than fabricating details.

---

**Last Updated**: 2025-11-15
**Maintainer**: Multi-Vibe AI Collaboration Team
**Quick Start**: `bash scripts/setup/setup_ms_swift_integration.sh && python scripts/training/master_rlwhf_launcher.py launch --dataset_path data/test/honesty_logs_sample.jsonl --output_dir experiments/test_run/`
