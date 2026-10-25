# Production Dockerfile for CIFAR-10 Pure NumPy Classifier
FROM python:3.11-slim as base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY neural_network.py best_model.npz README.md ./
COPY loss_curves.png accuracy_curves.png confusion_matrix.png ./
COPY app/ ./app/
COPY tests/ ./tests/

# Security: Create non-root user
RUN useradd -u 8888 -m appuser && \
    mkdir -p /tmp/matplotlib && \
    chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
