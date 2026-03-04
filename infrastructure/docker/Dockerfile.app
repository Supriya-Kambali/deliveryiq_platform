# IBM DeliveryIQ — Streamlit Frontend Dockerfile
# ================================================
# WHY A DOCKERFILE?
#   A Dockerfile is a recipe for building a Docker image.
#   It defines EXACTLY what goes into the container:
#   - Which OS/base image
#   - Which Python version
#   - Which dependencies
#   - Which files to copy
#   - Which command to run
#
#   This ensures IBM DeliveryIQ runs identically on:
#   - Your Mac M4 Pro
#   - IBM's Linux servers
#   - Any cloud provider (AWS, Azure, IBM Cloud)
#
# WHY PYTHON 3.11-SLIM?
#   - python:3.11-slim is a minimal Python image (~50MB vs ~900MB for full)
#   - Smaller image = faster deployment, less storage, better security
#   - 3.11 is the latest stable Python with best performance

# ── BASE IMAGE ────────────────────────────────────────────────────
FROM python:3.11-slim

# ── METADATA ──────────────────────────────────────────────────────
LABEL maintainer="IBM DeliveryIQ Team"
LABEL description="IBM DeliveryIQ Streamlit Frontend"
LABEL version="1.0"

# ── ENVIRONMENT VARIABLES ─────────────────────────────────────────
# WHY ENV VARS?
# Configuration should not be hardcoded. ENV vars allow the same
# image to work in dev (local Ollama) and prod (IBM Cloud Ollama).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    OLLAMA_HOST=http://ollama:11434 \
    CHROMADB_HOST=chromadb \
    CHROMADB_PORT=8000

# ── SYSTEM DEPENDENCIES ───────────────────────────────────────────
# WHY THESE PACKAGES?
# - curl: Health checks in Docker Compose
# - build-essential: Compile some Python packages (numpy, etc.)
# - git: Some pip packages install from git
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ── WORKING DIRECTORY ─────────────────────────────────────────────
WORKDIR /app

# ── INSTALL PYTHON DEPENDENCIES ───────────────────────────────────
# WHY COPY requirements.txt FIRST?
# Docker caches each layer. If we copy requirements.txt first and
# install dependencies, Docker won't reinstall them unless
# requirements.txt changes — even if other files change.
# This makes rebuilds MUCH faster.
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── COPY APPLICATION CODE ─────────────────────────────────────────
COPY . .

# ── EXPOSE PORT ───────────────────────────────────────────────────
EXPOSE 8501

# ── HEALTH CHECK ──────────────────────────────────────────────────
# WHY HEALTHCHECK?
# Docker and Kubernetes use health checks to know if the container
# is ready to receive traffic. Without it, requests might go to
# a container that's still starting up.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# ── START COMMAND ─────────────────────────────────────────────────
# WHY STREAMLIT RUN?
# This is the command that starts the Streamlit server.
# --server.address=0.0.0.0: Accept connections from outside container
# --server.port=8501: Standard Streamlit port
CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]