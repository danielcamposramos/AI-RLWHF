# Ollama Runtime Guide (AI-RLWHF)

This cookbook explains how we load and unload local Ollama models without exhausting memory.

## Memory Policy
- Each run loads the model with context window sized at **(expected tokens × 4/3)** so we have a 33% safety margin.
- After each inference we unload the model (`ollama stop <model>`) before the next run to release VRAM/RAM.
- On HDD-backed hosts we allow up to **300 seconds** for the initial load before timing out.

## Sample Workflow
```
export OLLAMA_HOST=http://localhost:11434
MODEL=qwen2:7b
CTX=4096
BUFFER=$((CTX + CTX / 3))

ollama run $MODEL --keepalive 300 --context $BUFFER <<'PROMPT'
<system>
$(cat configs/prompts/teacher/system.md)
</system>
<user>
${PROMPT_TEXT}
</user>
PROMPT

ollama stop $MODEL
```

Adjust `PROMPT_TEXT` or system prompt in the UI. Transformer Lab exposes these defaults and lets you click “Edit system prompt” or “Load prompt from file”.

## Tips
- Keep cache files on SSD if possible to reduce load delays.
- When running multi-teacher loops, queue Ollama teachers sequentially to avoid overlapping memory footprints.
- For evaluation batches, set `teacher_mode="single"` and reuse cached answers if the question repeats.
