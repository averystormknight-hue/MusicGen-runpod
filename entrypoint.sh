#!/bin/bash
set -e

# Error trap
trap 'echo "❌ Error on line $LINENO. Command exited with status $?"' ERR

echo "🚀 Starting MusicGen entrypoint script..."

# Check and configure caching directories
if [ -d /runpod-volume ]; then
  echo "✅ Using /runpod-volume for caching directories..."
  mkdir -p /runpod-volume/hf /runpod-volume/torch
  export HF_HOME=/runpod-volume/hf
  export TRANSFORMERS_CACHE=/runpod-volume/hf
  export TORCH_HOME=/runpod-volume/torch
else
  echo "⚠️ Warning: /runpod-volume not found! Using local cache directories..."
  mkdir -p /tmp/hf /tmp/torch
  export HF_HOME=/tmp/hf
  export TRANSFORMERS_CACHE=/tmp/hf
  export TORCH_HOME=/tmp/torch
fi

echo "📂 Cache directories:"
echo "  HF_HOME=$HF_HOME"
echo "  TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "  TORCH_HOME=$TORCH_HOME"

# Start the handler
echo "🐍 Checking Python environment..."
python3 --version
python3 -c "import runpod; print('RunPod installed')" || echo "❌ RunPod NOT installed"
python3 -c "import audiocraft; print('Audiocraft installed')" || echo "❌ Audiocraft NOT installed"

echo "🎤 Starting the handler..."
python3 -u /app/handler.py
HANDLER_EXIT=$?
echo "❌ Handler exited with code $HANDLER_EXIT"
exit $HANDLER_EXIT
