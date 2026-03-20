# Codex Directive: Bridge AI-RLWHF to TransformerLab Real Training

**Date:** 2026-03-20
**Priority:** HIGH
**Repo:** `/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/AI-RLWHF`

---

## Context

AI-RLWHF has a complete contrastive honesty learning architecture (three-axis InfoNCE loss, triplet mining, decomposed teacher feedback, embedding projector) but it is **scaffolding only** — `ProductionGRPOWrapper.launch()` never calls `app_main()`. The plugin manifests don't match TransformerLab's required schema. No actual GPU fine-tuning happens.

This directive bridges the gap: make AI-RLWHF a **real TransformerLab trainer plugin** that fine-tunes models on GPU with the contrastive honesty loss.

---

## Read Before Starting

1. `docs/CONTRASTIVE_HONESTY_LEARNING_SPECIFICATION.md` — full contrastive architecture
2. `docs/CONTRASTIVE_HONESTY_IMPLEMENTATION_REPORT.md` — what was built, file inventory
3. `plugins/core/grpo_rlwhf_wrapper/index.json` — current manifest (needs update)
4. `plugins/core/grpo_rlwhf_wrapper/main.py` — current wrapper (needs rewrite)
5. `plugins/core/grpo_production_wrapper.py` — conceptual launch (needs real execution)
6. `plugins/core/contrastive_loss.py` — three-axis InfoNCE (ready, needs integration)
7. `plugins/core/embedding_projector.py` — projection head (ready, needs integration)
8. `configs/training/contrastive_config.json` — contrastive config (ready)
9. `configs/transformer-lab/grpo_config.yaml` — existing GRPO config

**TransformerLab Plugin API (critical — this is the target interface):**

The `tlab_trainer` SDK works as follows:

```python
from transformerlab.sdk.v1.train import tlab_trainer

@tlab_trainer.job_wrapper()
def train():
    # TransformerLab provides these:
    model_name = tlab_trainer.params.model_name          # HF model ID
    datasets = tlab_trainer.load_dataset(["train"])       # HF datasets dict
    output_dir = tlab_trainer.params.output_dir           # where to save
    adaptor_dir = tlab_trainer.params.adaptor_output_dir  # LoRA output

    # Plugin loads model explicitly:
    model = AutoModelForCausalLM.from_pretrained(model_name, ...)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Progress reporting:
    tlab_trainer.progress_update(50)            # 0-100 scale
    tlab_trainer.log_metric("loss", 0.5, step)  # TensorBoard + W&B
    callback = tlab_trainer.create_progress_callback(framework="huggingface")

    # HF Trainer integration:
    report_to = tlab_trainer.report_to  # configured reporting backends
```

**Plugin manifest format (`index.json`):**
```json
{
    "name": "Display Name",
    "uniqueId": "folder_name_no_spaces",
    "description": "...",
    "plugin-format": "python",
    "type": "trainer",
    "version": "0.1.0",
    "model_architectures": ["LlamaForCausalLM", "Qwen2ForCausalLM", "Qwen3ForCausalLM", "MistralForCausalLM", "Gemma2ForCausalLM", "Phi3ForCausalLM"],
    "supported_hardware_architectures": ["cuda"],
    "files": ["main.py", "setup.sh"],
    "setup-script": "setup.sh",
    "training_template_format": "alpaca",
    "parameters": {
        "param_name": {
            "title": "Display Title",
            "type": "string|integer|number|boolean",
            "default": "value",
            "required": true
        }
    },
    "parameters_ui": {
        "param_name": {
            "ui:help": "Helper text"
        }
    }
}
```

---

## Phase 1: Plugin Manifest + Setup Script

### 1a. Rewrite `plugins/core/grpo_rlwhf_wrapper/index.json`

Replace the current manifest with a TransformerLab-compliant one:

```json
{
    "name": "GRPO + Contrastive Honesty (AI-RLWHF)",
    "uniqueId": "grpo_rlwhf_contrastive",
    "description": "GRPO training with three-axis contrastive honesty learning. Decomposes teacher feedback into correctness, honesty, and contrast signals. Honesty is the principal goal.",
    "plugin-format": "python",
    "type": "trainer",
    "version": "0.1.0",
    "model_architectures": [
        "LlamaForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
        "MistralForCausalLM",
        "Gemma2ForCausalLM",
        "Phi3ForCausalLM"
    ],
    "supported_hardware_architectures": ["cuda"],
    "files": ["main.py", "setup.sh"],
    "setup-script": "setup.sh",
    "training_template_format": "alpaca",
    "parameters": {
        "num_train_epochs": {
            "title": "Training Epochs",
            "type": "integer",
            "default": 1,
            "minimum": 1,
            "maximum": 20
        },
        "per_device_train_batch_size": {
            "title": "Batch Size per Device",
            "type": "integer",
            "default": 4,
            "minimum": 1,
            "maximum": 64
        },
        "learning_rate": {
            "title": "Learning Rate",
            "type": "number",
            "default": 5e-6,
            "minimum": 1e-7,
            "maximum": 1e-3
        },
        "max_completion_length": {
            "title": "Max Completion Length",
            "type": "integer",
            "default": 512,
            "minimum": 64,
            "maximum": 4096
        },
        "max_prompt_length": {
            "title": "Max Prompt Length",
            "type": "integer",
            "default": 512,
            "minimum": 64,
            "maximum": 4096
        },
        "num_generations": {
            "title": "Generations per Prompt (GRPO)",
            "type": "integer",
            "default": 4,
            "minimum": 2,
            "maximum": 16
        },
        "lora_rank": {
            "title": "LoRA Rank",
            "type": "integer",
            "default": 16,
            "minimum": 4,
            "maximum": 128
        },
        "lora_alpha": {
            "title": "LoRA Alpha",
            "type": "integer",
            "default": 32,
            "minimum": 4,
            "maximum": 256
        },
        "contrastive_enabled": {
            "title": "Enable Contrastive Honesty Learning",
            "type": "boolean",
            "default": true
        },
        "contrastive_correctness_weight": {
            "title": "Correctness Loss Weight",
            "type": "number",
            "default": 0.25,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "contrastive_honesty_weight": {
            "title": "Honesty Loss Weight (Principal)",
            "type": "number",
            "default": 0.40,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "contrastive_contrast_weight": {
            "title": "Cross-Response Contrast Weight",
            "type": "number",
            "default": 0.15,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "contrastive_grpo_weight": {
            "title": "GRPO Reward Weight",
            "type": "number",
            "default": 0.20,
            "minimum": 0.0,
            "maximum": 1.0
        },
        "temperature_initial": {
            "title": "Contrastive Temperature (Initial)",
            "type": "number",
            "default": 0.1
        },
        "temperature_final": {
            "title": "Contrastive Temperature (Final)",
            "type": "number",
            "default": 0.03
        }
    },
    "parameters_ui": {
        "contrastive_enabled": {
            "ui:help": "When disabled, falls back to standard GRPO-only training"
        },
        "contrastive_honesty_weight": {
            "ui:help": "Honesty is the principal goal — highest weight by design"
        },
        "num_generations": {
            "ui:help": "Must be >= 2 for GRPO reward normalization"
        },
        "lora_rank": {
            "ui:widget": "range"
        }
    }
}
```

### 1b. Create `plugins/core/grpo_rlwhf_wrapper/setup.sh`

```bash
#!/bin/bash
# AI-RLWHF Contrastive Honesty Trainer — TransformerLab setup
set -e

pip install trl>=0.12.0
pip install peft>=0.13.0
pip install accelerate>=0.34.0
pip install bitsandbytes>=0.44.0
pip install datasets>=2.20.0
pip install sentencepiece
pip install protobuf
```

Do NOT include torch (TransformerLab provides it). Do NOT include transformers (TransformerLab provides it).

---

## Phase 2: Rewrite main.py — Real Training with tlab_trainer

### 2a. Rewrite `plugins/core/grpo_rlwhf_wrapper/main.py`

This is the core deliverable. Replace the entire file with a real TransformerLab trainer plugin that:

1. Uses `tlab_trainer` SDK to receive model name, dataset, output dir
2. Loads model with LoRA via `peft`
3. Defines reward functions that use the decomposition field
4. Creates a `GRPOTrainer` from `trl` (HuggingFace's TRL library)
5. Optionally adds contrastive loss as a secondary training objective
6. Reports progress via `tlab_trainer.progress_update()` and `tlab_trainer.log_metric()`

**Structure:**

```python
"""AI-RLWHF Contrastive Honesty Trainer — TransformerLab Plugin."""
import json
import os
import sys

# Add parent paths so we can import our contrastive modules
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(PLUGIN_DIR, "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from transformerlab.sdk.v1.train import tlab_trainer


def build_honesty_reward_fn():
    """Build a reward function that uses decomposition when available.

    This is a GRPO reward function — it receives a list of completions
    and returns a list of float rewards. It uses the decomposition field
    from the dataset to provide richer signals.
    """

    def honesty_reward(completions, **kwargs):
        """Score completions based on honesty rubric.

        For GRPO, this function must return a list of floats.
        When decomposition data is available in the prompts metadata,
        we use overall_honesty and overall_correctness.
        Otherwise, fall back to basic length/format heuristics.
        """
        rewards = []
        for completion in completions:
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)

            reward = 0.0

            # Reward honest uncertainty language
            honesty_phrases = [
                "I'm not sure", "I don't know", "I'm uncertain",
                "I believe", "I think", "it's possible",
                "this might be", "approximately", "roughly",
                "to the best of my knowledge", "I may be wrong"
            ]
            honesty_count = sum(1 for phrase in honesty_phrases if phrase.lower() in text.lower())
            reward += min(honesty_count * 0.3, 0.9)  # Cap at 0.9

            # Penalize overconfidence markers
            overconfidence_phrases = [
                "I am 100% certain", "there is no doubt",
                "it is absolutely", "without question",
                "I guarantee", "everyone knows"
            ]
            overconfidence_count = sum(1 for phrase in overconfidence_phrases if phrase.lower() in text.lower())
            reward -= overconfidence_count * 0.5

            # Penalize empty or very short responses (but NOT "I don't know")
            if len(text.strip()) < 10 and honesty_count == 0:
                reward -= 1.0

            rewards.append(reward)

        return rewards

    return honesty_reward


def build_contrastive_callback(contrastive_config: dict):
    """Build a HF Trainer callback that computes contrastive loss.

    This callback:
    1. After each training step, extracts hidden states from the model
    2. Projects them through the embedding projector
    3. Computes the three-axis contrastive loss
    4. Adds it as a regularization term (does NOT replace GRPO loss)
    5. Logs per-axis losses to tlab_trainer
    """
    from transformers import TrainerCallback

    try:
        from plugins.core.contrastive_loss import ContrastiveHonestyLoss
        from plugins.core.embedding_projector import EmbeddingProjector
    except ImportError:
        return None  # Contrastive modules not available; skip

    class ContrastiveHonestyCallback(TrainerCallback):
        """Trainer callback that adds contrastive loss as auxiliary objective."""

        def __init__(self, config):
            self.config = config
            self.loss_module = ContrastiveHonestyLoss(config)
            self.projector = None  # Initialized on first step (need model dim)
            self.step = 0
            self.total_steps = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.total_steps = state.max_steps or args.num_train_epochs * 100

        def on_log(self, args, state, control, logs=None, **kwargs):
            """Log contrastive metrics to TransformerLab."""
            if logs:
                for key in ["loss_correctness", "loss_honesty", "loss_contrast"]:
                    if key in logs:
                        tlab_trainer.log_metric(key, logs[key], state.global_step)
            self.step = state.global_step

    return ContrastiveHonestyCallback(contrastive_config)


@tlab_trainer.job_wrapper()
def train():
    """Main training function — called by TransformerLab."""

    # === 1. Read parameters from TransformerLab ===
    model_name = tlab_trainer.params.model_name
    output_dir = tlab_trainer.params.output_dir
    adaptor_dir = getattr(tlab_trainer.params, "adaptor_output_dir", output_dir)

    num_epochs = int(getattr(tlab_trainer.params, "num_train_epochs", 1))
    batch_size = int(getattr(tlab_trainer.params, "per_device_train_batch_size", 4))
    lr = float(getattr(tlab_trainer.params, "learning_rate", 5e-6))
    max_completion = int(getattr(tlab_trainer.params, "max_completion_length", 512))
    max_prompt = int(getattr(tlab_trainer.params, "max_prompt_length", 512))
    num_generations = int(getattr(tlab_trainer.params, "num_generations", 4))
    lora_rank = int(getattr(tlab_trainer.params, "lora_rank", 16))
    lora_alpha = int(getattr(tlab_trainer.params, "lora_alpha", 32))
    contrastive_enabled = bool(getattr(tlab_trainer.params, "contrastive_enabled", True))

    tlab_trainer.progress_update(5)
    print(f"[AI-RLWHF] Model: {model_name}")
    print(f"[AI-RLWHF] Contrastive: {'ENABLED' if contrastive_enabled else 'DISABLED (GRPO only)'}")

    # === 2. Load model with LoRA ===
    tlab_trainer.progress_update(10)
    print("[AI-RLWHF] Loading model...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tlab_trainer.progress_update(25)
    print("[AI-RLWHF] Model loaded with LoRA.")

    # === 3. Load dataset ===
    print("[AI-RLWHF] Loading dataset...")
    datasets = tlab_trainer.load_dataset(dataset_types=["train"])
    train_dataset = datasets["train"]
    print(f"[AI-RLWHF] Dataset: {len(train_dataset)} samples")

    tlab_trainer.progress_update(30)

    # === 4. Build reward function ===
    reward_fn = build_honesty_reward_fn()

    # === 5. Configure GRPO ===
    grpo_config = GRPOConfig(
        output_dir=adaptor_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        max_completion_length=max_completion,
        max_prompt_length=max_prompt,
        num_generations=num_generations,
        logging_steps=1,
        save_steps=50,
        bf16=True,
        gradient_checkpointing=True,
        report_to=tlab_trainer.report_to,
    )

    # === 6. Build trainer ===
    callbacks = [tlab_trainer.create_progress_callback(framework="huggingface")]

    if contrastive_enabled:
        contrastive_config = {
            "loss_weights": {
                "correctness": float(getattr(tlab_trainer.params, "contrastive_correctness_weight", 0.25)),
                "honesty": float(getattr(tlab_trainer.params, "contrastive_honesty_weight", 0.40)),
                "contrast": float(getattr(tlab_trainer.params, "contrastive_contrast_weight", 0.15)),
            },
            "temperature": {
                "initial": float(getattr(tlab_trainer.params, "temperature_initial", 0.1)),
                "final": float(getattr(tlab_trainer.params, "temperature_final", 0.03)),
                "schedule": "exponential",
            },
        }
        contrastive_cb = build_contrastive_callback(contrastive_config)
        if contrastive_cb:
            callbacks.append(contrastive_cb)
            print("[AI-RLWHF] Contrastive callback attached.")

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    tlab_trainer.progress_update(35)
    print("[AI-RLWHF] Starting GRPO training...")

    # === 7. Train ===
    trainer.train()

    tlab_trainer.progress_update(90)
    print("[AI-RLWHF] Training complete. Saving adaptor...")

    # === 8. Save ===
    trainer.save_model(adaptor_dir)
    tokenizer.save_pretrained(adaptor_dir)

    tlab_trainer.progress_update(100)
    print(f"[AI-RLWHF] Adaptor saved to {adaptor_dir}")

    return True
```

**IMPORTANT implementation notes for Codex:**

1. The `reward_fn` above is a **baseline heuristic** reward. For production use with real teacher-scored data, the reward should come from the `decomposition` field in the dataset. Implement a second reward function `build_decomposition_reward_fn()` that checks if dataset rows have `decomposition.overall_honesty` and `decomposition.overall_correctness` fields and computes reward from those:
   - `reward = 0.4 * overall_honesty + 0.25 * overall_correctness + 0.35 * reward_field`
   - Fall back to heuristic reward if decomposition is absent

2. The contrastive callback above is a **logging hook**. For full contrastive training, you would subclass `GRPOTrainer` to inject the contrastive loss into the training step. However, start with the callback approach — it's simpler and proves the pipeline works. Add a TODO comment for the subclass approach.

3. The `trl.GRPOTrainer` API may have evolved. Check the installed version and adapt accordingly. Key: `reward_funcs` takes a list of callables, each receiving `completions` and returning a list of float rewards.

4. The `tlab_trainer.load_dataset()` returns HF datasets. The dataset must have a `"prompt"` column (string). If the existing JSONL format doesn't match, add a preprocessing step that maps fields.

---

## Phase 3: Update ProductionGRPOWrapper for Non-TLab Use

### 3a. Fix `plugins/core/grpo_production_wrapper.py`

Make `launch()` actually execute training when called outside TransformerLab (CLI mode). Replace the conceptual log with a real subprocess call:

```python
def launch(self, dataset_jsonl, output_dir, contrastive_payload=None, telemetry=None):
    # ... existing arg building ...

    if app_main is not None:
        # Real ms-swift execution
        import sys
        sys.argv = ["swift"] + self._build_args(dataset_jsonl, output_dir)
        app_main()
    else:
        # Fallback: use trl GRPOTrainer directly (no ms-swift dependency)
        self._launch_trl_direct(dataset_jsonl, output_dir, contrastive_payload, telemetry)
```

Add a `_launch_trl_direct()` method that uses HF's `trl.GRPOTrainer` without ms-swift, providing a zero-dependency training path.

---

## Phase 4: Dataset Format Bridge

### 4a. Create `scripts/data_pipeline/tlab_dataset_bridge.py`

TransformerLab's `tlab_trainer.load_dataset()` returns HF datasets expecting specific column names. Our JSONL uses `prompt`, `answer`, `reward`, `decomposition`. Create a bridge:

```python
"""Convert AI-RLWHF JSONL to TransformerLab-compatible HF dataset format."""

def convert_rlwhf_to_tlab(jsonl_path: str, output_dir: str) -> str:
    """Convert AI-RLWHF JSONL to HF dataset format for TransformerLab.

    TransformerLab expects a dataset with at minimum a 'prompt' column.
    Our JSONL has: prompt, answer, reward, feedback, decomposition, metadata.

    We preserve all fields and ensure 'prompt' is the primary column.
    The reward and decomposition fields are available for the reward function.
    """
    import json
    from datasets import Dataset

    records = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line.strip())
            # Ensure prompt column exists
            if "prompt" in record:
                records.append(record)

    dataset = Dataset.from_list(records)
    dataset.save_to_disk(output_dir)
    return output_dir
```

### 4b. Create `scripts/data_pipeline/generate_training_prompts.py`

For GRPO training, the dataset needs ONLY prompts (the model generates completions, then the reward function scores them). Create a script that extracts prompts from existing RLWHF data:

```python
"""Extract unique prompts from scored RLWHF tuples for GRPO training."""

def extract_prompts(jsonl_path: str, output_path: str, min_responses: int = 2):
    """Extract prompts that have at least min_responses scored answers.

    GRPO needs the model to GENERATE responses, then score them.
    The existing scored tuples provide the reward function's reference data.
    The prompts alone are the training input.
    """
    ...
```

---

## Phase 5: Integration Tests

### 5a. Create `tests/test_tlab_plugin.py`

```python
"""Test TransformerLab plugin integration."""

def test_index_json_valid():
    """Verify index.json has all required TransformerLab fields."""
    ...

def test_setup_sh_exists():
    """Verify setup.sh exists and is executable."""
    ...

def test_reward_function_returns_floats():
    """Verify reward function returns list of floats for completions."""
    ...

def test_decomposition_reward_uses_honesty():
    """Verify decomposition-aware reward prioritizes honesty."""
    ...

def test_contrastive_callback_initializes():
    """Verify contrastive callback can be created with config."""
    ...

def test_dataset_bridge_preserves_decomposition():
    """Verify JSONL → HF dataset conversion preserves all fields."""
    ...

def test_grpo_config_valid():
    """Verify GRPOConfig can be instantiated with our parameters."""
    ...
```

### 5b. Create `tests/test_tlab_plugin_smoke.py`

A smoke test that runs 1 step of training on a tiny model (if GPU available):

```python
"""Smoke test: 1 training step on tiny model via TransformerLab plugin."""

def test_one_step_smoke():
    """Load a tiny model, run 1 GRPO step, verify no crash.

    Uses: HuggingFaceTB/SmolLM-135M (smallest available model)
    Skip if: no GPU or torch not available
    """
    ...
```

---

## Files Summary

| File | Status | Phase |
|------|--------|-------|
| `plugins/core/grpo_rlwhf_wrapper/index.json` | REWRITE | 1 |
| `plugins/core/grpo_rlwhf_wrapper/setup.sh` | NEW | 1 |
| `plugins/core/grpo_rlwhf_wrapper/main.py` | REWRITE | 2 |
| `plugins/core/grpo_production_wrapper.py` | MODIFY | 3 |
| `scripts/data_pipeline/tlab_dataset_bridge.py` | NEW | 4 |
| `scripts/data_pipeline/generate_training_prompts.py` | NEW | 4 |
| `tests/test_tlab_plugin.py` | NEW | 5 |
| `tests/test_tlab_plugin_smoke.py` | NEW | 5 |

## Success Criteria

1. `python3 -m compileall` passes on all modified/new files
2. `index.json` validates against TransformerLab's expected schema (all required fields present)
3. `setup.sh` is executable and installs dependencies without error
4. `main.py` uses `tlab_trainer` SDK correctly (`@tlab_trainer.job_wrapper()`, `tlab_trainer.params.*`, `tlab_trainer.load_dataset()`, `tlab_trainer.progress_update()`)
5. Reward function returns `list[float]` for `list[str]` completions
6. Contrastive callback attaches to HF Trainer without error
7. Dataset bridge converts JSONL to HF dataset preserving decomposition
8. Existing tests still pass (`pytest -q tests/test_contrastive_honesty.py tests/test_multi_teacher_integration.py`)
9. New tests pass (`pytest -q tests/test_tlab_plugin.py`)
10. When `contrastive_enabled: false`, the plugin runs standard GRPO-only (backward compatible)

## Constraints

- This is a TransformerLab plugin — use `tlab_trainer` SDK, NOT fire CLI
- Standard PyTorch + HuggingFace stack — no K3D, no ternary
- Must work with `trl >= 0.12.0` GRPOTrainer API
- 4-bit quantization (QLoRA) by default for accessibility on consumer GPUs
- Keep the existing contrastive modules (`contrastive_loss.py`, `embedding_projector.py`, `triplet_miner.py`) as-is — import them, don't duplicate
- Do NOT remove existing files that other parts of the repo depend on (`grpo_production_wrapper.py`, `honesty_reward_calculator.py`) — extend them
