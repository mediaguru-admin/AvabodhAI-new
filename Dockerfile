# ── Base image ────────────────────────────────────────────────────────────────
# Python 3.11 slim — small size, production ready
FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Apurba"
LABEL description="Avabodh API — Document Intelligence Platform"
LABEL version="1.0.0"

# ── Environment variables ─────────────────────────────────────────────────────
# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8

# ── Set working directory ─────────────────────────────────────────────────────
WORKDIR /app

# ── Install system dependencies ───────────────────────────────────────────────
# These are needed for psycopg2, pdfminer, and other packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Copy requirements first (Docker cache optimization) ───────────────────────
# If requirements.txt doesn't change, Docker reuses cached layer
# This makes rebuilds much faster
COPY requirements.txt .

# ── Install Python dependencies ───────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# ── Pre-download NLTK data needed by `unstructured` for DOCX parsing ──────────
# Without this, the FIRST .docx upload tries to download this data from the
# internet at runtime, mid-request — and if that download fails (network
# restrictions, a flaky connection, whatever), the request hangs for minutes
# before erroring out instead of just working. Baking it into the image here
# means it's always present, with zero network dependency at runtime.
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt_tab punkt

# ── Pre-download fastembed models (SPLADE sparse + cross-encoder reranker) ─────
# Same rationale as the NLTK download above — ~100MB combined, one-time,
# baked into the image so the first real request doesn't pay this cost
# (or fail outright on a restricted-egress network) mid-request.
RUN python -c "from fastembed import SparseTextEmbedding; \
    from fastembed.rerank.cross_encoder import TextCrossEncoder; \
    SparseTextEmbedding(model_name='prithivida/Splade_PP_en_v1'); \
    TextCrossEncoder(model_name='BAAI/bge-reranker-base')"

# NOTE: the two model names above MUST match config/settings.py's
# SPARSE_MODEL / RERANK_MODEL. They drifted once already (this baked in
# bge-reranker-v2-m3 while the app used bge-reranker-base), so the baked-in
# model was never touched and the real one downloaded mid-request instead.

# -- Pre-download the PDF layout + table-structure models --------------------
# unstructured's hi_res strategy loads a YOLOX layout model and a table
# transformer on FIRST USE. Left to runtime that is a multi-hundred-MB
# download landing inside whichever upload happens to be first after a
# deploy. Baked in here; main.py additionally warms them into memory at boot.
RUN python -c "from unstructured_inference.models.base import get_model; get_model()"
RUN python -c "from unstructured_inference.models.tables import load_agent; load_agent()"

# ── Install Playwright browser ────────────────────────────────────────────────
RUN playwright install-deps chromium
RUN playwright install chromium

# ── Copy application code ─────────────────────────────────────────────────────
COPY . .

# ── Create necessary directories ──────────────────────────────────────────────
RUN mkdir -p uploaded_files documents

# ── Expose port ───────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# Docker checks this every 30s — marks container unhealthy if it fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start command ─────────────────────────────────────────────────────────────
# 0.0.0.0 makes it accessible from outside the container
# workers=2 handles multiple requests simultaneously
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
