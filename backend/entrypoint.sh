#!/bin/bash
set -e

# Wait for Qdrant to be ready
QDRANT_URL="${QDRANT_URL:-http://qdrant:6333}"
echo "Waiting for Qdrant at ${QDRANT_URL}..."
until curl -sf "${QDRANT_URL}/collections" > /dev/null 2>&1; do
    echo "Qdrant not ready, retrying in 2s..."
    sleep 2
done
echo "Qdrant is ready."

# Wait for Ollama to be ready (API available, not necessarily models)
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://ollama:11434}"
echo "Waiting for Ollama at ${OLLAMA_BASE_URL}..."
until curl -sf "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; do
    echo "Ollama not ready, retrying in 2s..."
    sleep 2
done
echo "Ollama is ready."

# Start the FastAPI application
exec python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000
