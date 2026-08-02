# Cohere Transcribe (2B) needs a modern CUDA + torch (cu121) and transformers>=5.4.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (python3.11 to match the tested runtime)
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    ffmpeg curl \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip3 install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the model into the image so startup is fast and offline-capable
# (NAMAA Cohere-Speech-Tashkeel-2B, ~5 GB). Comment out to download at runtime instead.
RUN python3 -c "from transformers import AutoProcessor, CohereAsrForConditionalGeneration as M; \
r='NAMAA-Space/Cohere-Speech-Tashkeel-2B'; AutoProcessor.from_pretrained(r); M.from_pretrained(r)"

# Copy application code (the old Whisper checkpoint under model/ is no longer used)
COPY core/ core/
COPY data/ data/
COPY main.py .

# Create temp storage
RUN mkdir -p temp_storage

# Expose port
EXPOSE 8000

# Health check (PORT is injected by RunPod Serverless; defaults to 8000 for docker compose)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD sh -c "curl -f http://localhost:${PORT:-8000}/health || exit 1"

# Run
CMD ["python3", "main.py"]
