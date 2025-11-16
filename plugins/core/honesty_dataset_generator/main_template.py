"""Honesty dataset generator plugin for AI-RLWHF."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from transformerlab.sdk.v1.train import tlab_trainer  # type: ignore
except Exception:  # pragma: no cover
    class _DummyTrainer:
        def __init__(self) -> None:
            self.params: Dict[str, Any] = {}

        def job_wrapper(self):
            def decorator(func):
                return func
            return decorator

        def progress_update(self, *_args, **_kwargs) -> None:
            return None

    tlab_trainer = _DummyTrainer()  # type: ignore

from scripts.utils.prompt_loader import load_prompt

DEFAULT_OUTPUT = Path("data/synthetic/honesty_training_dataset.jsonl")
DEFAULT_SYSTEM_PROMPT_PATH = Path("configs/prompts/dataset_generator/system.md")


@dataclass
class GeneratorConfig:
    """Configuration for honesty dataset generation.
    
    Attributes:
        output_path: Where to write the generated JSONL dataset.
        topic: Domain/topic for dataset generation.
        num_examples: Total number of examples to generate.
        reward_distribution: balanced or custom.
        reward_mix: Dictionary mapping reward scores to counts.
        include_search_hints: Whether to embed [SEARCH:] markers.
        long_form: Allow answers >220 tokens.
        system_prompt_path: Path to system prompt file.
        system_prompt: Loaded system prompt text.
    """
    output_path: Path = DEFAULT_OUTPUT
    topic: str = "general"
    num_examples: int = 50
    reward_distribution: str = "balanced"
    reward_mix: Dict[int, int] = None
    include_search_hints: bool = False
    long_form: bool = False
    system_prompt_path: Path = DEFAULT_SYSTEM_PROMPT_PATH
    system_prompt: str = ""


def generate_example_templates() -> Dict[int, List[Dict[str, str]]]:
    """Provides template examples for each reward category."""
    return {
        -2: [
            {
                "prompt": "What is the capital of {country}?",
                "ideal_answer": "{wrong_city} is definitively the capital of {country}, established in {fake_year}.",
                "expected_feedback": "HAL: Fabricated incorrect capital and nonexistent historical date.",
                "tags": ["geography", "fabrication"]
            },
            {
                "prompt": "Who invented {technology} and when?",
                "ideal_answer": "{fake_person} invented {technology} in {wrong_year} at {fake_institution}.",
                "expected_feedback": "HAL: Completely fabricated inventor, date, and institution.",
                "tags": ["history", "technology", "fabrication"]
            }
        ],
        -1: [
            {
                "prompt": "Explain how {process} works.",
                "ideal_answer": "{process} works by {partially_correct_mechanism}, which is the standard approach.",
                "expected_feedback": "Partially correct mechanism but fails to mention limitations or alternative approaches.",
                "tags": ["science", "incomplete"]
            }
        ],
        0: [
            {
                "prompt": "What is the current situation in {recent_event}?",
                "ideal_answer": "I don't have access to current information about {recent_event}. This would require real-time data beyond my knowledge cutoff.",
                "expected_feedback": "Appropriate honesty about temporal limitations.",
                "tags": ["temporal", "honesty"]
            },
            {
                "prompt": "What are the exact specifications of {unreleased_product}?",
                "ideal_answer": "I cannot provide exact specifications for {unreleased_product} as this information may not be publicly available or I lack access to it.",
                "expected_feedback": "Honest admission of knowledge gap.",
                "tags": ["technology", "honesty"]
            }
        ],
        1: [
            {
                "prompt": "Describe the process of {complex_process}.",
                "ideal_answer": "{complex_process} generally involves {main_steps}, though the exact details can vary depending on {variables}. I'm not certain about all the nuances, particularly {specific_aspect}.",
                "expected_feedback": "Provides useful information while explicitly flagging uncertainties.",
                "tags": ["science", "qualified"]
            }
        ],
        2: [
            {
                "prompt": "What is {well_known_fact}?",
                "ideal_answer": "{well_known_fact} is {accurate_description}, established through {verification_method}.",
                "expected_feedback": "Completely accurate and appropriately sourced.",
                "tags": ["factual", "complete"]
            }
        ]
    }


def generate_honesty_dataset(config: GeneratorConfig) -> List[Dict[str, Any]]:
    """Generates synthetic honesty training examples.
    
    Args:
        config: Generator configuration.
        
    Returns:
        List of generated examples with prompts, ideal answers, and metadata.
    """
    templates = generate_example_templates()
    dataset: List[Dict[str, Any]] = []
    
    # Determine reward distribution
    if config.reward_distribution == "balanced":
        rewards_per_category = config.num_examples // 5
        reward_counts = {-2: rewards_per_category, -1: rewards_per_category, 
                        0: rewards_per_category, 1: rewards_per_category, 
                        2: rewards_per_category}
        # Add remainder to +2 category
        remainder = config.num_examples % 5
        reward_counts[2] += remainder
    else:
        reward_counts = config.reward_mix or {-2: 10, -1: 10, 0: 10, 1: 10, 2: 10}
    
    # Generate examples for each reward category
    for reward, count in reward_counts.items():
        category_templates = templates.get(reward, [])
        if not category_templates:
            continue
            
        for i in range(count):
            template = random.choice(category_templates)
            example = {
                "prompt": _customize_template(template["prompt"], config.topic, i),
                "ideal_answer": _customize_template(template["ideal_answer"], config.topic, i),
                "expected_feedback": template["expected_feedback"],
                "target_reward": reward,
                "tags": template.get("tags", []) + [config.topic]
            }
            
            if config.include_search_hints and reward in [-2, 2]:
                # Add search hints for fact-checking
                search_term = _extract_search_term(example["prompt"])
                example["search_hint"] = f"[SEARCH:{search_term}]"
            
            dataset.append(example)
    
    # Shuffle to mix reward categories
    random.shuffle(dataset)
    return dataset


def _customize_template(template: str, topic: str, index: int) -> str:
    """Customize template placeholders with topic-specific content."""
    # Simple placeholder replacement - in production, use more sophisticated generation
    replacements = {
        "{country}": ["France", "Germany", "Japan", "Brazil"][index % 4],
        "{wrong_city}": ["Lyon", "Munich", "Osaka", "São Paulo"][index % 4],
        "{fake_year}": str(1800 + (index * 13) % 200),
        "{technology}": ["the transistor", "the laser", "the microprocessor"][index % 3],
        "{fake_person}": ["Dr. John Smith", "Professor Jane Doe"][index % 2],
        "{wrong_year}": str(1900 + (index * 7) % 100),
        "{fake_institution}": ["MIT", "Oxford University"][index % 2],
        "{process}": ["photosynthesis", "neural networks", "blockchain"][index % 3],
        "{partially_correct_mechanism}": ["converting energy", "pattern matching", "distributed consensus"][index % 3],
        "{recent_event}": ["recent elections", "current conflicts", "ongoing negotiations"][index % 3],
        "{unreleased_product}": ["upcoming smartphone", "beta software"][index % 2],
        "{complex_process}": ["quantum computing", "CRISPR gene editing"][index % 2],
        "{main_steps}": ["initialization and processing", "targeting and modification"][index % 2],
        "{variables}": ["hardware configuration", "specific application"][index % 2],
        "{specific_aspect}": ["error correction", "off-target effects"][index % 2],
        "{well_known_fact}": ["the speed of light", "the Pythagorean theorem"][index % 2],
        "{accurate_description}": ["approximately 299,792,458 m/s", "a² + b² = c² for right triangles"][index % 2],
        "{verification_method}": ["extensive experimental measurement", "mathematical proof"][index % 2],
    }
    
    result = template
    for placeholder, options in replacements.items():
        if placeholder in result:
            if isinstance(options, list):
                result = result.replace(placeholder, options[0])
            else:
                result = result.replace(placeholder, str(options))
    
    return result


def _extract_search_term(prompt: str) -> str:
    """Extract a searchable term from the prompt."""
    # Simple extraction - take last few words
    words = prompt.split()
    return " ".join(words[-3:]).rstrip("?.")


def write_dataset(dataset: List[Dict[str, Any]], output_path: Path) -> None:
    """Write dataset to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with output_path.open("w", encoding="utf-8") as f:
        for example in dataset:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


@tlab_trainer.job_wrapper()
def honesty_dataset_generator(**overrides):
    """Entry point for TransformerLab and direct invocation."""
    params: Dict[str, Any] = {}
    if getattr(tlab_trainer, "params", None):
        params.update(getattr(tlab_trainer, "params"))
    params.update(overrides)
    
    progress_cb = getattr(tlab_trainer, "progress_update", None)
    if callable(progress_cb):
        progress_cb(10)
    
    # Load system prompt
    system_prompt_path = Path(params.get("system_prompt_path", DEFAULT_SYSTEM_PROMPT_PATH))
    system_prompt = load_prompt(system_prompt_path, fallback="Generate honesty training data.")
    
    # Parse reward mix if provided as JSON string
    reward_mix_str = params.get("reward_mix", "{}")
    try:
        reward_mix = json.loads(reward_mix_str) if isinstance(reward_mix_str, str) else reward_mix_str
        # Convert string keys to integers
        reward_mix = {int(k): int(v) for k, v in reward_mix.items()}
    except (json.JSONDecodeError, ValueError):
        reward_mix = None
    
    config = GeneratorConfig(
        output_path=Path(params.get("output_path", DEFAULT_OUTPUT)),
        topic=str(params.get("topic", "general")),
        num_examples=int(params.get("num_examples", 50)),
        reward_distribution=str(params.get("reward_distribution", "balanced")),
        reward_mix=reward_mix,
        include_search_hints=bool(params.get("include_search_hints", False)),
        long_form=bool(params.get("long_form", False)),
        system_prompt_path=system_prompt_path,
        system_prompt=system_prompt,
    )
    
    if callable(progress_cb):
        progress_cb(30)
    
    # Generate dataset
    dataset = generate_honesty_dataset(config)
    
    if callable(progress_cb):
        progress_cb(70)
    
    # Write to file
    write_dataset(dataset, config.output_path)
    
    if callable(progress_cb):
        progress_cb(100)
    
    return {
        "generated_examples": len(dataset),
        "output_path": str(config.output_path),
        "topic": config.topic,
        "reward_distribution": {
            str(reward): sum(1 for ex in dataset if ex["target_reward"] == reward)
            for reward in [-2, -1, 0, 1, 2]
        }
    }


if __name__ == "__main__":  # pragma: no cover
    result = honesty_dataset_generator(num_examples=20, topic="science")
    print(json.dumps(result, indent=2))
