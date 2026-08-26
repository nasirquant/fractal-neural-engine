# Fractal Neural Simulation Engine (FNSE) - Dockerfile
# Multi-stage build for production-ready container

# =============================================================================
# Stage 1: Base Python image with system dependencies
# =============================================================================
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r fnse && useradd -r -g fnse -d /app -s /sbin/nologin fnse

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 2: Dependencies installation
# =============================================================================
FROM base AS deps

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Production image
# =============================================================================
FROM base AS production

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=fnse:fnse . .

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/skills && \
    chown -R fnse:fnse /app

# Switch to non-root user
USER fnse

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - run the API server
CMD ["python", "-m", "uvicorn", "fnse.main:app", "--host", "0.0.0.0", "--port", "8000"]