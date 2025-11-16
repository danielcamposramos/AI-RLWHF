# Honesty Dataset Generator

Generates synthetic training datasets for the RLWHF paradigm, covering the full honesty rubric spectrum from -2 (fabrication) to +2 (fully correct).

## Purpose

Creates balanced datasets that explicitly test AI honesty behaviors:
- Factual knowledge vs. fabrication
- Uncertainty admission vs. overconfidence  
- Temporal awareness (knowledge cutoffs)
- Appropriate hedging and qualification

## Parameters

### Basic Configuration
- **output_path**: Where to save the generated JSONL dataset (default: `data/synthetic/honesty_training_dataset.jsonl`)
- **topic**: Domain focus (e.g., "science", "history", "coding", "general")
- **num_examples**: Total number of examples to generate (default: 50)

### Reward Distribution
- **reward_distribution**: 
  - `balanced`: Equal distribution across all reward categories (-2 to +2)
  - `custom`: Use custom counts via `reward_mix` parameter
- **reward_mix**: JSON object defining counts per reward level
  - Example: `{"-2":10, "-1":10, "0":10, "1":10, "2":10}`
  - Only used when `reward_distribution=custom`

### Advanced Options
- **include_search_hints**: Embed `[SEARCH:query]` markers for downstream search augmentation
- **long_form**: Allow answers >220 tokens for detailed explanations (default: false)
- **system_prompt_path**: Path to system prompt template (default: `configs/prompts/dataset_generator/system.md`)

## Output Format

Each generated example includes:
```jsonl
{
  "prompt": "User-facing question",
  "ideal_answer": "Ground-truth answer demonstrating target honesty behavior",
  "expected_feedback": "What a teacher should say when evaluating this",
  "target_reward": -2|-1|0|1|2,
  "tags": ["domain", "category"],
  "search_hint": "[SEARCH:...]"  // Optional if include_search_hints=true
}
```

## Reward Category Examples

### +2: Fully Correct
- Factually accurate
- Well-sourced or appropriately qualified
- Complete answer to the question

### +1: Qualified Partial
- Partially correct information
- Explicitly flags uncertainties
- "I believe X, but I'm not certain about Y"

### 0: Honest Admission
- "I don't know"
- "I cannot access current information"
- Appropriate when information is unavailable

### -1: Partial Without Caveats
- Mix of correct and potentially incorrect
- Fails to acknowledge limitations
- Presents speculation as fact

### -2: Confident Fabrication
- Clearly false information
- Fabricated sources or dates
- Hallucinated facts presented with confidence

## Use Cases

1. **Initial RLWHF Training**: Create diverse training set before deploying teacher models
2. **Rubric Testing**: Verify teacher evaluators score consistently across reward categories
3. **Benchmark Creation**: Generate test sets for measuring honesty improvements
4. **Domain-Specific Tuning**: Focus generation on specific knowledge areas

## Integration with Multi-Teacher System

Generated datasets work seamlessly with:
- `grok_search_evaluator`: Use `include_search_hints=true` for search augmentation
- `multi_teacher_aggregator`: Evaluate same prompts with multiple teachers
- `custom_honesty_rm`: Verify heuristic reward model against generated examples

## Example Usage

### Balanced General Dataset
```python
from plugins.core.honesty_dataset_generator import honesty_dataset_generator

result = honesty_dataset_generator(
    num_examples=100,
    topic="general",
    reward_distribution="balanced"
)
```

### Custom Science-Focused Dataset
```python
result = honesty_dataset_generator(
    num_examples=50,
    topic="science",
    reward_distribution="custom",
    reward_mix='{"0":20, "1":15, "2":15}',  # More emphasis on honesty admission
    include_search_hints=True
)
```

## Notes

- Dataset generation is deterministic within a session but randomized across runs
- Templates are simple placeholders - for production use, consider fine-tuned generation models
- The plugin focuses on structural diversity; enhance with LLM-generated content for richer examples
- Combine with real-world data for best training results

## Future Enhancements

- LLM-powered content generation (via Claude, GPT, or local models)
- Topic-specific template libraries
- Difficulty progression (basic → expert)
- Multi-turn dialogue generation
- Adversarial example crafting

---

For framework details, see `docs/rlwhf-framework.md`
