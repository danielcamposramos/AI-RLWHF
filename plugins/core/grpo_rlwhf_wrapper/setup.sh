#!/bin/bash
# AI-RLWHF Contrastive Honesty Trainer — TransformerLab setup
set -e

pip install "trl>=0.12.0"
pip install "peft>=0.13.0"
pip install "accelerate>=0.34.0"
pip install "bitsandbytes>=0.44.0"
pip install "datasets>=2.20.0"
pip install sentencepiece
pip install protobuf
