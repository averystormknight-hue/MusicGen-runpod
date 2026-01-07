FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git curl ffmpeg ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Align torch stack with audiocraft 1.2.0 expectations (2.1.0 + cu121). If you don’t need vision, you can drop torchvision.
RUN pip3 install --no-cache-dir \
    torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN chmod +x /app/entrypoint.sh

CMD ["/app/entrypoint.sh"]
