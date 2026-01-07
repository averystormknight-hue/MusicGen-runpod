FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/huggingface_cache \
    TRANSFORMERS_CACHE=/tmp/huggingface_cache \
    TORCH_HOME=/tmp/torch_cache

# Install dependencies and make sure git is available
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Verify git installation
RUN which git && git --version

# Install PyTorch with CUDA 11.8 compatibility
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118
    
# Preload the facebook/musicgen-large model and safetensors into the cache
RUN python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/musicgen-large')"

# Set working directory and install Python dependencies
WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Copy application code into the container
COPY . /app
RUN chmod +x /app/entrypoint.sh

# Use exec syntax for CMD
CMD ["/bin/bash", "/app/entrypoint.sh"]
