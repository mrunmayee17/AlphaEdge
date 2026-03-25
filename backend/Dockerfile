FROM python:3.13-slim

WORKDIR /app

# System deps for numpy/torch + git for HuggingFace model downloads
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (skip 2GB+ of CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install Python deps
COPY backend/pyproject.toml backend/setup.py* ./backend/
COPY backend/ ./backend/
RUN pip install --no-cache-dir --no-deps -e "./backend[test]" && \
    pip install --no-cache-dir -e "./backend[test]"

# Copy alpha model code (legacy architecture reference)
COPY alpha_model/ ./alpha_model/

# Install the FinCast runtime needed for ffm / LoRA inference.
RUN pip install --no-cache-dir beartype einx utilsforecast CoLT5-attention jax jaxlib && \
    pip install --no-cache-dir "git+https://github.com/vincent05r/FinCast-fts.git"

# HuggingFace model cache directory
ENV HF_HOME=/app/.hf_cache

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
