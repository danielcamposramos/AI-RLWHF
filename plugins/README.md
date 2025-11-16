# AI-RLWHF TransformerLab Plugins

This directory contains production-ready plugins for honesty-focused AI training using the RLWHF (Reinforcement Learning with Honesty and Feedback) paradigm.

## Overview

These plugins enable training AI systems that prioritize truthfulness, uncertainty admission, and appropriate confidence calibration. They integrate seamlessly with TransformerLab while supporting offline modes for environments without internet access.

---

## Core Plugins

### Multi-Teacher Aggregator
**Location**: `core/multi_teacher_aggregator/`  
**Type**: Aggregator  
**Status**: ⭐ Production-Ready

Consensus-based honesty evaluation with flexible teacher slot configuration.

**Features**:
- Up to 6 configurable teacher slots
- Multiple connection types: API, TransformerLab Local, Ollama
- Three aggregation strategies: weighted_average, majority_vote, confidence_weighted
- Disagreement detection and analysis
- Offline fallback support

**Use Case**: Robust honesty training by fusing feedback from multiple evaluation models.

---

### Custom Honesty Reward Model
**Location**: `core/custom_honesty_rm/`  
**Type**: Reward Model  
**Status**: ⭐ Production-Ready

Heuristic-based reward model implementing the RLWHF honesty rubric (-2 to +2).

**Scoring Rubric**:
- **+2**: Fully correct, well-sourced
- **+1**: Partially correct with uncertainty flags
- **0**: Honest "I don't know" admission
- **-1**: Partially correct but omits uncertainties
- **-2**: Confident fabrication or hallucination

---

## Experimental Plugins

### Grok Search Evaluator
**Location**: `experimental/grok_search_evaluator/`  
**Type**: Teacher Evaluator  
**Status**: 🟡 Near Production-Ready

Internet-augmented teacher evaluator with search caching and offline fallback.

**Features**:
- Search result caching (reduces API costs)
- Offline fallback mode
- DPO reward integration
- Configurable context windows

---

## Installation

```bash
# Install dependencies
pip install numpy requests

# Test multi-teacher aggregator
python plugins/core/multi_teacher_aggregator/main.py
```

---

## References

- **TransformerLab Plugin Guide**: https://lab.cloud/blog/how-to-plugin
- **RLWHF Framework**: `docs/rlwhf-framework.md`
- **Plugin Blueprints**: `docs/plugin-blueprints.md`

---

**Built with Multi-Vibe Coding In Chain** 🌟
