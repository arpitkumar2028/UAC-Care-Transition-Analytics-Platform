# syntax=docker/dockerfile:1
# ==============================================================================
# MULTI-STAGE PRODUCTION DOCKERFILE
# UAC Care Transition Efficiency & Placement Outcome Analytics
# ==============================================================================

# --- STAGE 1: BUILDER (Dependency Compilation & Wheel Building) ---
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install C compilers & PostgreSQL dev headers needed for wheel compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create isolated virtual environment for clean layer transfer
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- STAGE 2: RUNTIME (Minimal Secure Production Image) ---
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app" \
    SQLITE_DB_PATH="/app/uac_analytics.db"

WORKDIR /app

# Install runtime dependencies only (libpq5 for psycopg2, curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled Python virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root security user for enterprise hardening
RUN addgroup --system uacgroup && \
    adduser --system --ingroup uacgroup uacuser

# Copy project source code
COPY --chown=uacuser:uacgroup . .

# Pre-seed fallback SQLite database, verify ML artifacts, and ensure volume mount dirs exist
RUN mkdir -p /app/data /app/ml/saved_models && \
    python db/init_db.py --reset && \
    python -c "from ml.models import train_and_save_all_models, DEFAULT_FORECASTER_PATH, DEFAULT_CLASSIFIER_PATH; (not DEFAULT_FORECASTER_PATH.exists() or not DEFAULT_CLASSIFIER_PATH.exists()) and train_and_save_all_models()" && \
    chown -R uacuser:uacgroup /app

USER uacuser

# Expose FastAPI (8000) and Streamlit (8501) ports
EXPOSE 8000 8501

# Default healthcheck targets Streamlit unless overridden in docker-compose
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || curl -f http://localhost:8000/ || exit 1

# Default command launches Streamlit Executive Dashboard
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
