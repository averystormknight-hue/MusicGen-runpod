#!/usr/bin/env bash
set -euo pipefail

# Check and configure caching directories
if [ -d /runpod-volume ]; then
  echo "Using /runpod-volume for caching directories..."
  mkdir -p /runpod-volume/hf /runpod-volume/torch
  export HF_HOME=/runpod-volume/hf
  export TRANSFORMERS_CACHE=/runpod-volume/hf
  export TORCH_HOME=/runpod-volume/torch
else
  echo "Warning: /runpod-volume not found! Using local cache directories..."
  mkdir -p /app/.cache/huggingface /app/.cache/torch
  export HF_HOME=/app/.cache/huggingface
  export TRANSFORMERS_CACHE=/app/.cache/huggingface
  export TORCH_HOME=/app/.cache/torch
fi

echo "Cache directories:"
echo "HF_HOME=$HF_HOME"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "TORCH_HOME=$TORCH_HOME"

# Start the handler
python3 -u /app/handler.py
