#!/usr/bin/env bash
set -euo pipefail

if [ -d /runpod-volume ]; then
  mkdir -p /runpod-volume/hf /runpod-volume/torch
  export HF_HOME=/runpod-volume/hf
  export TRANSFORMERS_CACHE=/runpod-volume/hf
  export TORCH_HOME=/runpod-volume/torch
fi

python3 -u /app/handler.py
