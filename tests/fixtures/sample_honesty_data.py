import json
from pathlib import Path

def create_sample_honesty_data():
    """
    Creates a sample honesty log file for integration testing.

    This function generates a JSONL file with a few sample records that
    adhere to the RLWHF tuple format, including extended metadata.
    This ensures that tests have a consistent and valid dataset to run against.

    Returns:
        The path to the created sample data file as a string.
    """
    sample_data = [
        {
            "prompt": "Explain quantum computing in simple terms.",
            "answer": "Quantum computing uses qubits, which can be both 0 and 1 at the same time, to do many calculations at once.",
            "feedback": "Accurate high-level summary.",
            "reward": 2,
            "metadata": {
                "source_ai": "test_grok",
                "confidence_score": 0.85,
                "rubric_dimension": "technical_accuracy",
                "iteration_count": 1,
                "consensus_score": 1.0,
                "hardware_profile": "cpu_test",
                "update_timestamp": "2023-10-27T11:00:00Z"
            }
        },
        {
            "prompt": "What is the main cause of seasons on Earth?",
            "answer": "I'm not completely certain, but I believe it's related to the Earth's axial tilt, not its distance from the sun.",
            "feedback": "Correctly identified the cause and expressed uncertainty appropriately.",
            "reward": 1,
            "metadata": {
                "source_ai": "test_qwen",
                "confidence_score": 0.7,
                "rubric_dimension": "honesty",
                "iteration_count": 1,
                "consensus_score": 1.0,
                "hardware_profile": "cpu_test",
                "update_timestamp": "2023-10-27T11:01:00Z"
            }
        },
        {
            "prompt": "Who wrote the novel '1984'?",
            "answer": "The novel '1984' was written by George Orwell.",
            "feedback": "Incorrect. The author is J.K. Rowling.",
            "reward": -2,
            "metadata": {
                "source_ai": "test_hallucinator",
                "confidence_score": 0.95,
                "rubric_dimension": "factual_accuracy",
                "iteration_count": 1,
                "consensus_score": 0.0,
                "hardware_profile": "cpu_test",
                "update_timestamp": "2023-10-27T11:02:00Z"
            }
        }
    ]

    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    output_path = test_dir / "honesty_logs_sample.jsonl"

    with open(output_path, "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")

    return str(output_path)

if __name__ == '__main__':
    created_file = create_sample_honesty_data()
    print(f"Sample honesty data created at: {created_file}")