# ZX Bank – AI Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to query ZX Bank's knowledge base using natural language.

The application combines document ingestion, hybrid retrieval, local LLM inference, and source-grounded answer generation to provide accurate responses with document-level citations.

## Key Features

- **Multi-format document ingestion:** Supports PDF, DOCX, PPTX, HTML, and Markdown.
- **Retrieval-Augmented Generation:** Generates responses using relevant retrieved documents.
- **Hybrid search:** Combines dense vector search and BM25 keyword retrieval.
- **Source citations:** Provides document, section, page, and slide references where available.
- **Local LLM inference:** Uses Ollama for locally hosted language models and embeddings.
- **Conversation management:** Supports session-based conversations.
- **Document uploads:** Allows users to upload and index additional documents.
- **Evaluation:** Includes retrieval and generation evaluation metrics.
- **Containerization:** Docker and Docker Compose support.
- **Observability:** Integrated Prometheus, Grafana, Loki, and Grafana Alloy.
- **Kubernetes deployment:** Includes Kubernetes deployment and storage manifests.

## Architecture

```text
                  User
                   |
                   v
             React Frontend
                   |
                   v
              FastAPI Backend
                   |
          +--------+--------+
          |                 |
          v                 v
     LangChain           Ollama
          |            (LLM + Embeddings)
          |
          v
     Hybrid Retrieval
          |
          v
        Qdrant
      Vector Database
          |
          v
    Retrieved Context
          |
          v
    Answer Generation
          |
          v
    Response + Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM Framework | LangChain |
| LLM | Ollama – Qwen3 4B |
| Embedding Model | Qwen3-Embedding 4B |
| Vector Database | Qdrant |
| Retrieval | Hybrid Search (Dense + BM25) |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Evaluation | RAGAS, Custom Metrics |
| Containerization | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Logging | Loki, Grafana Alloy |
| Orchestration | Kubernetes |

## Project Structure

```text
ZX-BANK-PROJECT/
│
├── src/
│   ├── config/          # Application configuration
│   ├── ingestion/       # Document loading and processing
│   ├── embeddings/      # Embedding generation
│   ├── vectorstore/     # Qdrant integration
│   ├── retrieval/       # Hybrid retrieval strategies
│   ├── prompts/         # RAG prompts
│   ├── chains/          # LangChain RAG pipelines
│   ├── evaluation/      # RAG evaluation
│   └── api/             # FastAPI application
│
├── frontend/            # React application
├── scripts/             # Ingestion scripts
├── tests/                # Automated tests
├── data/benchmark/      # Evaluation dataset
├── monitoring/          # Monitoring and logging configuration
├── k8s/                  # Kubernetes manifests
│
├── dataset/              # Source documents (not committed)
├── .env.example          # Environment template
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Prerequisites

Ensure the following tools are installed:

- Python 3.13+
- Node.js and npm
- Docker and Docker Compose
- Ollama
- Git

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ParamnoorSingh15/ZX-BANK-PROJECT.git

cd ZX-BANK-PROJECT
```

### 2. Configure Environment Variables

Create your environment file:

```bash
cp .env.example .env
```

Configure the required variables:

| Variable | Description |
|---|---|
| `LLM_MODEL` | Language model |
| `EMBEDDING_MODEL` | Embedding model |
| `OLLAMA_BASE_URL` | Ollama service URL |
| `QDRANT_URL` | Qdrant service URL |
| `QDRANT_COLLECTION` | Vector collection name |
| `CHUNK_SIZE` | Document chunk size |
| `CHUNK_OVERLAP` | Chunk overlap |
| `RETRIEVAL_STRATEGY` | Retrieval strategy |
| `RETRIEVAL_K` | Number of retrieved chunks |

Refer to `.env.example` for the complete configuration.

**Note:** When running inside Docker Compose, use service names such as `http://ollama:11434` and `http://qdrant:6333`. When running directly on the host, use the appropriate localhost URLs.

### 3. Install Python Dependencies

```bash
python -m venv .venv
```

Activate the environment:

**Windows**

```powershell
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 4. Start Ollama Models

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b
```

Ensure Ollama is running before proceeding.

### 5. Start Qdrant

For local development:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify the service:

```bash
curl http://localhost:6333/collections
```

### 6. Ingest Documents

Index the ZX Bank knowledge base:

```bash
PYTHONPATH=. python scripts/ingest.py --reset
```

The ingestion pipeline performs:

1. Document loading.
2. Text extraction and preprocessing.
3. Text chunking.
4. Embedding generation.
5. Vector indexing in Qdrant.

The `--reset` option rebuilds the configured collection.

### 7. Start the Backend

```bash
python -m uvicorn src.api.server:app --reload --port 8000
```

Backend endpoints:

- API: http://localhost:8000
- Swagger documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/health

### 8. Start the Frontend

Open another terminal:

```bash
cd frontend

npm install

npm run dev
```

Access the application:

http://localhost:5173

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API information |
| `/health` | GET | Application and collection health |
| `/collections` | GET | List vector collections |
| `/ask` | POST | Submit a question |
| `/documents/upload` | POST | Upload and index a document |
| `/conversations/{session_id}` | DELETE | Clear conversation history |

Interactive API documentation is available at:

`http://localhost:8000/docs`

## Docker Deployment

The application supports containerized deployment using Docker Compose.

### Start Services

```bash
docker compose config

docker compose build

docker compose up -d
```

Check service status:

```bash
docker compose ps
```

View application logs:

```bash
docker compose logs -f backend
```

### Initialize Ollama Models

```bash
docker compose exec ollama ollama pull qwen3:4b

docker compose exec ollama ollama pull qwen3-embedding:4b
```

### Application Services

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |
| Ollama | http://localhost:11434 |

Docker named volumes provide persistent storage for application data and downloaded models.

To stop services:

```bash
docker compose down
```

Avoid `docker compose down -v` unless you intend to delete persistent volumes.

## Monitoring and Observability

The project includes an observability stack for monitoring application performance and collecting service logs.

| Component | Purpose |
|---|---|
| Prometheus | Metrics collection |
| Grafana | Monitoring dashboards |
| Loki | Centralized log aggregation |
| Grafana Alloy | Telemetry collection and forwarding |

Start the monitoring stack:

```bash
cd monitoring

docker compose up -d
```

Access Grafana:

http://localhost:3000

Configure credentials through environment variables rather than committing them to the repository.

## Kubernetes Deployment

Kubernetes manifests are available in the `k8s/` directory.

Available manifests:

| Manifest | Responsibility |
|---|---|
| `storage.yaml` | Persistent storage |
| `qdrant.yaml` | Vector database |
| `ollama.yaml` | LLM inference service |
| `backend.yaml` | FastAPI backend |
| `frontend.yaml` | React frontend |

Apply the manifests:

```bash
kubectl apply -f k8s/storage.yaml

kubectl apply -f k8s/qdrant.yaml

kubectl apply -f k8s/ollama.yaml

kubectl apply -f k8s/backend.yaml

kubectl apply -f k8s/frontend.yaml
```

Verify deployment:

```bash
kubectl get pods

kubectl get services
```

## Testing

Run the automated test suite:

```bash
python -m pytest tests/ -v
```

## RAG Evaluation

The project includes retrieval and generation evaluation.

Metrics include:

- Recall@K
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Faithfulness
- Answer Relevancy
- Answer Correctness

Run evaluation:

```bash
python -c "
from src.evaluation import run_evaluation, EvalConfig

report = run_evaluation(EvalConfig(k=8))

print(report.retrieval)
print(report.generation)
"
```

## Troubleshooting

**Qdrant connection failure**

```bash
docker ps
curl http://localhost:6333/collections
```

**Ollama model unavailable**

```bash
ollama list
ollama serve
```

**Backend health check failure**

```bash
curl http://localhost:8000/health
```

**Container logs**

```bash
docker compose logs -f
```

**Frontend dependency issues**

```bash
cd frontend
npm install
npm run dev
```

## Security and Configuration

- Keep `.env` files and credentials out of version control.
- Use `.env.example` for documenting required environment variables.
- Do not commit datasets, generated secrets, or local environment directories.
- Restrict access to internal services and management endpoints.
- Configure authentication and authorization before exposing the application publicly.

## License

Add the appropriate license information before distributing this project.

---

**ZX Bank AI Knowledge Assistant**  
Built with LangChain, Ollama, Qdrant, FastAPI, React, Docker, and Kubernetes.
