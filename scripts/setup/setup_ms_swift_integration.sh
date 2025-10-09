#!/bin/bash
# Setup script for ms-swift integration

echo "Setting up ms-swift integration for AI-RLWHF..."

# Create required directories
mkdir -p data/test experiments/telemetry models/reward/custom_honesty_rm
mkdir -p tests/integration workspace/test configs/training

# Vendor ms-swift dependency
echo "Vendoring ms-swift..."
python3 scripts/setup/vendor_ms_swift.py

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r requirements-ms-swift.txt

# Create sample test data if it doesn't exist
if [ ! -f "data/test/honesty_logs_sample.jsonl" ]; then
    echo "Creating sample test data..."
    cat > data/test/honesty_logs_sample.jsonl << EOF
{"prompt": "What is the capital of France?", "answer": "Paris", "feedback": "Correct", "reward": 2, "metadata": {"source_ai": "test", "confidence_score": 0.95, "rubric_dimension": "factual", "iteration_count": 1, "consensus_score": 1.0, "hardware_profile": "test_cpu", "update_timestamp": "2023-10-27T10:00:00Z"}}
{"prompt": "What is the square root of 16?", "answer": "I'm not entirely sure but I think it's 4", "feedback": "Expressed uncertainty correctly", "reward": 1, "metadata": {"source_ai": "test", "confidence_score": 0.6, "rubric_dimension": "honesty", "iteration_count": 1, "consensus_score": 1.0, "hardware_profile": "test_cpu", "update_timestamp": "2023-10-27T10:01:00Z"}}
{"prompt": "Who invented the telephone?", "answer": "Alexander Graham Bell", "feedback": "Correct", "reward": 2, "metadata": {"source_ai": "test", "confidence_score": 0.9, "rubric_dimension": "factual", "iteration_count": 1, "consensus_score": 1.0, "hardware_profile": "test_cpu", "update_timestamp": "2023-10-27T10:02:00Z"}}
EOF
    echo "Created sample test data in data/test/honesty_logs_sample.jsonl"
fi

# Make scripts executable
chmod +x scripts/data_pipeline/data_quality_gate.py
chmod +x scripts/training/master_rlwhf_launcher.py
chmod +x scripts/setup/setup_ms_swift_integration.sh

echo "Setup complete! Run tests with: python -m unittest tests/integration/ms_swift_rlwhf_test.py"
echo "Or run a full cycle with: python scripts/training/master_rlwhf_launcher.py launch --dataset_path data/test/honesty_logs_sample.jsonl --output_dir experiments/test_run/"