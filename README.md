# ZX Bank – AI Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application that enables users to interact with ZX Bank's knowledge base through natural language queries.

Built using LangChain, Ollama, Qdrant, FastAPI, and React, the application supports multi-format document ingestion, dynamic PDF uploads, hybrid retrieval, and source-grounded answer generation.

## Key Features

- **Multi-format Document Ingestion:** Supports PDF, DOCX, PPTX, HTML, and Markdown documents.
- **Dynamic PDF Upload:** Upload PDF documents through the application and query their contents.
- **Retrieval-Augmented Generation:** Generates context-aware responses using relevant retrieved documents.
- **Hybrid Search:** Combines dense vector similarity search with BM25 keyword retrieval.
- **Source Citations:** Provides document, section, page, and slide references where available.
- **Local LLM Inference:** Uses Ollama for locally hosted language models and embeddings.
- **Conversation Management:** Supports session-based conversations and document lifecycle management.
- **Evaluation:** Includes retrieval and generation evaluation metrics.
- **Containerization:** Supports Docker and Docker Compose deployments.
- **Observability:** Integrates Prometheus, Grafana, Loki, and Grafana Alloy.
- **Kubernetes Deployment:** Includes Kubernetes deployment and persistent storage manifests.

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
              +----------+----------+
              |                     |
              v                     v
        Document Ingestion       RAG Pipeline
              |                     |
              v                     v
       Text Extraction         Hybrid Retrieval
              |                     |
              v                     v
         Text Chunking            Qdrant
              |                     |
              v                     v
       Ollama Embeddings      Retrieved Context
              |                     |
              +----------+----------+
                         |
                         v
                   Ollama LLM
                         |
                         v
                Answer Generation
                         |
                         v
                 Answer + Sources
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python 3.13+ |
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
├── frontend/            # React frontend
├── scripts/             # Document ingestion scripts
├── tests/               # Automated tests
├── data/benchmark/      # Evaluation dataset
├── dataset/              # Source documents (not committed)
├── monitoring/          # Monitoring and logging configuration
├── k8s/                  # Kubernetes manifests
│
├── .env.example          # Environment template
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Document Ingestion and PDF Upload

The application supports two document ingestion workflows.

### 1. Preloaded Knowledge Base

The initial ZX Bank knowledge base is indexed using the ingestion script.

Supported formats:

- PDF
- DOCX
- PPTX
- HTML
- Markdown

The ingestion pipeline performs:

1. Document loading and text extraction.
2. Content cleaning and preprocessing.
3. Text chunking with configurable overlap.
4. Embedding generation using Ollama.
5. Vector indexing in Qdrant.

Run the ingestion pipeline:

```bash
python scripts/ingest.py --reset
```

### 2. Dynamic PDF Upload

Users can upload PDF documents directly through the React application.

The upload workflow:

1. **Upload:** The PDF is submitted through the frontend to the FastAPI backend.
2. **Extraction:** Text is extracted from the uploaded document.
3. **Chunking:** Extracted content is divided into smaller, overlapping chunks.
4. **Embedding:** Ollama generates embeddings for the document chunks.
5. **Indexing:** Embeddings and metadata are stored in Qdrant.
6. **Retrieval:** Users can ask questions about the uploaded document.
7. **Generation:** The LLM generates responses using the retrieved context and provides source references.

### PDF Upload Data Lifecycle

- Uploaded PDF files are processed temporarily and deleted after indexing.
- Extracted text chunks and embeddings are stored in Qdrant.
- Uploaded documents are indexed into session-scoped collections named `zx_bank_uploads_<session_id>`.
- Clearing a conversation deletes its associated upload collection.

This separates the preloaded ZX Bank knowledge base from dynamically uploaded documents.

## Getting Started

### Prerequisites

Install the following:

- Python 3.13+
- Node.js and npm
- Docker and Docker Compose
- Ollama
- Git

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

Configure the required settings:

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

Use Docker service names for inter-container communication and localhost URLs when running services directly on the host.

### 3. Install Python Dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

**Windows:**

```powershell
.venv\Scripts\activate
```

**Linux / macOS:**

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

Ensure Ollama is running before starting the application.

### 5. Start Qdrant

For local development:

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Verify Qdrant:

```bash
curl http://localhost:6333/collections
```

### 6. Ingest the Knowledge Base

Place the ZX Bank source documents in the `dataset/` directory.

Run:

```bash
python scripts/ingest.py --reset
```

The script loads, processes, chunks, embeds, and indexes the documents into Qdrant.

The `--reset` option rebuilds the configured collection.

### 7. Start the Backend

```bash
python -m uvicorn src.api.server:app --reload --port 8000
```

Backend:

- API: http://localhost:8000
- Swagger Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

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

Interactive API documentation:

http://localhost:8000/docs

## Docker Deployment

The application supports containerized deployment using Docker Compose.

### Build and Start Services

```bash
docker compose config

docker compose build

docker compose up -d
```

Check running services:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs -f backend
```

### Initialize Ollama Models

Models need to be pulled after starting the Ollama container:

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

Docker named volumes provide persistent storage for Qdrant data and downloaded Ollama models.

Stop services:

```bash
docker compose down
```

Avoid `docker compose down -v` unless you intend to delete persistent volumes.

## Monitoring and Observability

The project includes an observability stack for collecting metrics and centralizing application logs.

| Component | Responsibility |
|---|---|
| Prometheus | Metrics collection and storage |
| Grafana | Monitoring dashboards and visualization |
| Loki | Centralized log aggregation |
| Grafana Alloy | Telemetry collection and forwarding |

### Start Monitoring

```bash
cd monitoring

docker compose up -d
```

Access Grafana:

http://localhost:3000

Prometheus:

http://localhost:9090

Loki:

http://localhost:3100

Alloy:

http://localhost:12345

Grafana is configured to use Prometheus and Loki as data sources.

Configure Grafana credentials securely through environment variables.

## Kubernetes Deployment

Kubernetes manifests are available in the `k8s/` directory.

| Manifest | Responsibility |
|---|---|
| `storage.yaml` | Persistent storage |
| `qdrant.yaml` | Vector database deployment |
| `ollama.yaml` | LLM inference deployment |
| `backend.yaml` | FastAPI backend |
| `frontend.yaml` | React frontend |

### Deploy to Kubernetes

```bash
kubectl apply -f k8s/storage.yaml

kubectl apply -f k8s/qdrant.yaml

kubectl apply -f k8s/ollama.yaml

kubectl apply -f k8s/backend.yaml

kubectl apply -f k8s/frontend.yaml
```

### Verify Deployment

```bash
kubectl get pods

kubectl get services

kubectl get pvc
```

## Testing

Run the automated test suite:

```bash
python -m pytest tests/ -v
```

## RAG Evaluation

The application includes retrieval and generation evaluation using RAGAS and custom metrics.

Evaluation metrics include:

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

- Never commit `.env` files, passwords, API keys, or generated secrets.
- Use `.env.example` to document required environment variables.
- Exclude datasets, virtual environments, and generated files from version control.
- Restrict access to internal services and management endpoints.
- Configure authentication and authorization before exposing the application publicly.
- Avoid publishing sensitive document content or credentials in logs.

## Additional Documentation

Refer to `LEARNING_NOTES.md` for detailed explanations of LangChain components, document loaders, text splitting, embeddings, hybrid retrieval, RAG chains, and evaluation concepts.


---

**ZX Bank – AI Knowledge Assistant**

Built with LangChain, Ollama, Qdrant, FastAPI, React, Docker, and Kubernetes.
