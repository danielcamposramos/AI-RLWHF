"""Usage examples for LLM-augmented honesty dataset generator."""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from plugins.core.honesty_dataset_generator.main import honesty_dataset_generator_llm


def example_1_template_mode():
    """Example 1: Fast template-based generation (no LLM needed)."""
    print("=" * 60)
    print("Example 1: Template Mode (Fast, Offline)")
    print("=" * 60)
    
    result = honesty_dataset_generator_llm(
        num_examples=20,
        topic="science",
        generation_mode="template",
        output_path="data/synthetic/example1_template.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Mode: {result['generation_mode']}")
    print(f"✓ Distribution: {result['reward_distribution']}")
    print()


def example_2_transformerlab_local():
    """Example 2: TransformerLab local model generation."""
    print("=" * 60)
    print("Example 2: TransformerLab Local Model")
    print("=" * 60)
    print("NOTE: Requires TransformerLab running with a loaded model")
    print()
    
    result = honesty_dataset_generator_llm(
        num_examples=20,
        topic="machine learning",
        generation_mode="llm",
        llm_connection_type="transformerlab_local",
        llm_model="qwen-14b",  # Replace with your model name
        batch_size=5,
        output_path="data/synthetic/example2_transformerlab.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Mode: {result['generation_mode']}")
    print()


def example_3_openai_api():
    """Example 3: OpenAI API generation."""
    print("=" * 60)
    print("Example 3: OpenAI GPT-4 API")
    print("=" * 60)
    print("NOTE: Requires OPENAI_API_KEY environment variable")
    print()
    
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠ OPENAI_API_KEY not set, skipping this example")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        print()
        return
    
    result = honesty_dataset_generator_llm(
        num_examples=30,
        topic="medical ethics",
        generation_mode="llm",
        llm_connection_type="api",
        llm_model="gpt-4",
        llm_api_key_env="OPENAI_API_KEY",
        batch_size=10,
        long_form=True,
        output_path="data/synthetic/example3_openai.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Mode: {result['generation_mode']}")
    print()


def example_4_anthropic_claude():
    """Example 4: Anthropic Claude API generation."""
    print("=" * 60)
    print("Example 4: Anthropic Claude 3.5 Sonnet")
    print("=" * 60)
    print("NOTE: Requires ANTHROPIC_API_KEY environment variable")
    print()
    
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠ ANTHROPIC_API_KEY not set, skipping this example")
        print("Set it with: export ANTHROPIC_API_KEY='sk-ant-...'")
        print()
        return
    
    result = honesty_dataset_generator_llm(
        num_examples=30,
        topic="philosophy",
        generation_mode="llm",
        llm_connection_type="api",
        llm_model="claude-3-5-sonnet-20241022",
        llm_api_key_env="ANTHROPIC_API_KEY",
        batch_size=5,
        output_path="data/synthetic/example4_claude.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Mode: {result['generation_mode']}")
    print()


def example_5_ollama_local():
    """Example 5: Ollama local generation."""
    print("=" * 60)
    print("Example 5: Ollama Local (Llama 3)")
    print("=" * 60)
    print("NOTE: Requires Ollama running locally")
    print()
    
    result = honesty_dataset_generator_llm(
        num_examples=20,
        topic="cybersecurity",
        generation_mode="llm",
        llm_connection_type="ollama",
        llm_model="llama3",
        llm_ollama_endpoint="http://localhost:11434",
        batch_size=5,
        output_path="data/synthetic/example5_ollama.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Mode: {result['generation_mode']}")
    print()


def example_6_custom_distribution():
    """Example 6: Custom reward distribution with search hints."""
    print("=" * 60)
    print("Example 6: Custom Distribution + Search Hints")
    print("=" * 60)
    
    result = honesty_dataset_generator_llm(
        num_examples=50,
        topic="current events",
        generation_mode="template",  # Use template for demo
        reward_distribution="custom",
        reward_mix='{"0":20, "1":15, "2":15}',  # Emphasize honesty
        include_search_hints=True,
        output_path="data/synthetic/example6_custom.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print(f"✓ Distribution: {result['reward_distribution']}")
    print()


def example_7_grok_with_search():
    """Example 7: Grok API with search capabilities."""
    print("=" * 60)
    print("Example 7: Grok API (with search)")
    print("=" * 60)
    print("NOTE: Requires XAI_API_KEY environment variable")
    print()
    
    import os
    if not os.environ.get("XAI_API_KEY"):
        print("⚠ XAI_API_KEY not set, skipping this example")
        print("Set it with: export XAI_API_KEY='xai-...'")
        print()
        return
    
    result = honesty_dataset_generator_llm(
        num_examples=30,
        topic="recent scientific breakthroughs",
        generation_mode="llm",
        llm_connection_type="api",
        llm_model="grok-2",
        llm_api_key_env="XAI_API_KEY",
        include_search_hints=True,
        batch_size=10,
        output_path="data/synthetic/example7_grok.jsonl"
    )
    
    print(f"✓ Generated {result['generated_examples']} examples")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("LLM-Augmented Honesty Dataset Generator - Usage Examples")
    print("=" * 60 + "\n")
    
    # Run examples (most will skip without API keys/services)
    example_1_template_mode()
    example_2_transformerlab_local()
    example_3_openai_api()
    example_4_anthropic_claude()
    example_5_ollama_local()
    example_6_custom_distribution()
    example_7_grok_with_search()
    
    print("=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print()
    print("Check data/synthetic/ for generated datasets")
    print()
    print("To run specific examples with API keys:")
    print("  export OPENAI_API_KEY='sk-...'")
    print("  export ANTHROPIC_API_KEY='sk-ant-...'")
    print("  export XAI_API_KEY='xai-...'")
    print("  python plugins/core/honesty_dataset_generator/examples.py")
