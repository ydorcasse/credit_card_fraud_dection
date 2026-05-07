# ---------- Build stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ---------- Runtime stage ----------
FROM python:3.11-slim

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy only what the API needs (no notebooks, no raw data)
COPY api/ api/
COPY models/ models/

# Create logs directory with correct permissions
RUN mkdir -p logs && chown -R appuser:appuser /app

USER appuser

EXPOSE 5001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "api/app.py"]
