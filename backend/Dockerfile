FROM python:3.13-slim

WORKDIR /app

# Python runtime settings for container logging
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (curl for health checks and wait logic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/

# Create non-root user and set ownership
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Health check: backend /health endpoint depends on Qdrant being reachable
HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Copy entrypoint script for service readiness checks
COPY backend/entrypoint.sh /entrypoint.sh

ENTRYPOINT ["sh", "/entrypoint.sh"]
CMD ["python", "-m", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
