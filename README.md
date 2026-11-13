# ZX Bank RAG Application

End-to-end Retrieval-Augmented Generation (RAG) application for ZX Bank using
**LangChain**, **Ollama** (Qwen3 4B LLM + Qwen3-Embedding 4B), **Qdrant**
(vector database), **FastAPI** (backend), and **React** (frontend).

Answers are generated **only** from ZX Bank documents, include source
citations (document, section, page/slide), and are rendered with proper
Markdown tables and lists in the chat UI.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.13+ |
| LLM Framework | LangChain |
| LLM | Ollama — `qwen3:4b` (reasoning enabled) |
| Embeddings | Ollama — `qwen3-embedding:4b` (2560 dimensions) |
| Vector DB | Qdrant |
| Retrieval | Hybrid: dense cosine + BM25 keyword, deduplicated, neighbour-expanded |
| Backend | FastAPI + Uvicorn |
| Frontend | React + Vite + react-markdown |
| Evaluation | RAGAS + custom metrics (Recall@K, Precision@K, MRR, nDCG, faithfulness, answer relevancy, correctness) |
| Container | Docker (Qdrant) |

---

## Project Structure

```
zx-bank-rag/
├── dataset/                 # Existing ZX Bank dataset (preserved, not committed)
│   └── pdf/ html/ docx/ pptx/ md/
├── data/benchmark/          # Benchmark questions for evaluation
├── src/
│   ├── config/              # Environment-driven settings
│   ├── ingestion/           # Loaders (tables included), cleaning, splitting
│   ├── embeddings/          # Ollama embedding client + dimension probing
│   ├── vectorstore/         # Qdrant integration, indexing, reset, dedup
│   ├── retrieval/           # Hybrid, dense, MMR, filtered retrievers
│   ├── prompts/             # Detailed-answer RAG prompt
│   ├── chains/              # LCEL RAG chains + Qwen3 reasoning settings
│   ├── evaluation/          # Retrieval + generation metrics
│   └── api/                 # FastAPI backend
├── frontend/                # React chat interface (Markdown rendering)
├── scripts/ingest.py        # Ingestion CLI (supports --reset)
├── tests/                   # Pytest unit + integration tests
├── requirements.txt
├── .env.example             # Copy to .env and configure
├── LEARNING_NOTES.md        # LangChain concept explanations
└── README.md
```

---

## What Was Fixed for Answer Accuracy

| Problem | Fix |
|---------|-----|
| PowerPoint tables (FD rates, branch lists) were silently dropped | PPTX loader now extracts text frames, nested group shapes, tables, charts and speaker notes |
| HTML tables lost row structure | Tables are converted to pipe-separated rows before text extraction |
| 400-character chunks split headings from their tables | Default chunk size is now 1600 chars with 200 overlap |
| 9% of chunks were title-only fragments | Chunks under 200 chars are merged into a neighbour |
| Duplicate copies from 5 file formats filled the top results | Retrieved chunks are near-deduplicated and capped per document |
| Dense search missed exact terms such as `3.50%`, `IFSC`, branch names | Hybrid retrieval combines cosine similarity with BM25 keyword scoring |
| A hit could be only half of a section | Each top hit is expanded with its neighbouring chunks |
| Qwen3 returned empty answers when reasoning used the whole output budget | `num_ctx=16384`, `num_predict=3072` and an automatic retry |
| Raw `**`, `|` and `#` appeared in the chat | Answers are rendered as Markdown with GFM tables |

---

## Prerequisites

- **Python 3.13+** with a virtual environment
- **Ollama** running at `http://localhost:11434` with:
  ```bash
  ollama pull qwen3:4b
  ollama pull qwen3-embedding:4b
  ```
- **Docker Desktop** for Qdrant

---

## Step-by-Step Setup

### 1. Clone the repository

```bash
git clone https://github.com/ParamnoorSingh15/ZX-BANK-PROJECT.git
cd "ZX Bank Project"
```

### 2. Configure environment

```bash
cp .env.example .env
```

The important settings:

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_MODEL` | `qwen3:4b` | Generation model |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model |
| `LLM_REASONING` | `true` | Keep Qwen3 thinking enabled |
| `LLM_NUM_CTX` | `16384` | Context window |
| `LLM_NUM_PREDICT` | `3072` | Output budget (reasoning + answer) |
| `CHUNK_SIZE` | `1600` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_STRATEGY` | `hybrid` | `hybrid`, `dense`, `mmr` or `filtered` |
| `RETRIEVAL_K` | `8` | Chunks after deduplication |
| `RETRIEVAL_FETCH_K` | `40` | Candidates fetched before re-ranking |
| `RETRIEVAL_KEYWORD_WEIGHT` | `0.35` | BM25 weight in the hybrid score |

### 3. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# or
source .venv/bin/activate       # Mac/Linux

python -m pip install -r requirements.txt
```

### 4. Start Qdrant with persistent storage

```bash
mkdir qdrant_storage
docker run -d --name qdrant -p 6333:6333 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant
```

Verify:

```bash
curl http://localhost:6333/collections
```

> The named volume keeps your vectors across container restarts. On later
> sessions use `docker start qdrant` instead of re-running the command.

### 5. Pull Ollama models

```bash
ollama pull qwen3:4b
ollama pull qwen3-embedding:4b
```

### 6. Ingest documents into Qdrant

First run, or **any time you change chunking / loaders / embedding model**:

```powershell
$env:PYTHONPATH="C:\Users\Param\Desktop\ZX Bank Project"; python scripts/ingest.py --reset
```

Mac/Linux:

```bash
PYTHONPATH="." python scripts/ingest.py --reset
```

What the script reports:

- chunks loaded and split
- chunks by format (`pdf`, `html`, `docx`, `pptx`, `md`)
- total characters
- number of chunks under 120 characters (should be very small)
- indexed progress every 100 chunks

**All five formats are indexed** so no content is lost. Duplicate copies are
removed at search time, not during ingestion.

### 7. Start the FastAPI backend

Terminal 1:

```powershell
$env:PYTHONPATH="C:\Users\Param\Desktop\ZX Bank Project"; python -m uvicorn src.api.server:app --reload --port 8000
```

Verify: <http://localhost:8000/health> should return `"status":"ok"`.

API docs: <http://localhost:8000/docs>

### 8. Start the React frontend

Terminal 2:

```bash
cd frontend
npm install          # first time only
npm run dev
```

Open <http://localhost:5173> and ask a question.

> Answers take roughly 20–60 seconds because Qwen3 first reasons about the
> retrieved context. The UI shows a live “thinking” timer.

---

## Daily Workflow (after first setup)

You do **not** need to re-ingest every time. The vectors persist in
`qdrant_storage`.

| Step | Command |
|------|---------|
| Start Qdrant | `docker start qdrant` |
| Start backend | `python -m uvicorn src.api.server:app --reload --port 8000` |
| Start frontend | `cd frontend && npm run dev` |

Re-ingest with `--reset` only when:

- you add or change documents in the dataset,
- you change `CHUNK_SIZE` / `CHUNK_OVERLAP`,
- you change the embedding model,
- you delete `qdrant_storage`.

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | Qdrant and collection status |
| `/collections` | GET | List Qdrant collections |
| `/ask` | POST | Question → detailed answer + sources |
| `/documents/upload` | POST | Upload and index a document |
| `/conversations/{session_id}` | DELETE | Clear chat history |

Example request:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the interest rates on fixed deposits?", "strategy": "hybrid", "k": 8}'
```

Response shape:

```json
{
  "answer": "ZX Bank offers the following sample fixed deposit rates ...",
  "sources": [
    {
      "title": "ZX Bank Fixed Deposits",
      "format": "md",
      "section": "Sample Interest Rates",
      "page": null,
      "slide": null,
      "content_preview": "..."
    }
  ],
  "strategy": "hybrid",
  "k": 8,
  "session_id": null
}
```

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Evaluation

```bash
python -c "
from src.evaluation import run_evaluation, EvalConfig
report = run_evaluation(EvalConfig(k=8))
print(report.retrieval)
print(report.generation)
"
```

---

## Troubleshooting

**Qdrant status is red / “Backend unreachable”**
- Start Qdrant: `docker start qdrant`
- Start FastAPI on port 8000.
- The frontend proxies `/api/*` to `http://localhost:8000` and strips `/api`
  (see `frontend/vite.config.js`).

**Port 6333 already allocated**
- A Qdrant container already exists. Use `docker ps` and `docker start <name>`
  rather than starting a second container.

**Answers are empty**
- Qwen3 reasoning consumed the output budget. Keep `LLM_NUM_CTX=16384` and
  `LLM_NUM_PREDICT=3072`; the chain also retries automatically.
- Do **not** set `LLM_REASONING=false` on this Ollama build: the raw
  chain-of-thought leaks into the answer text.

**Answer says the data is missing**
- Confirm ingestion finished and `/health` shows the expected chunk count.
- Re-run with `--reset` after changing loaders or chunk sizes.
- Check that the question is about ZX Bank; the model refuses out-of-scope
  questions by design.

**Port 8000 in use**
```powershell
Get-NetTCPConnection -LocalPort 8000 | Select-Object OwningProcess
Stop-Process -Id <pid>
```

---

## Docker Deployment

The application is fully containerized with Docker Compose. Four services run
on a dedicated network: **frontend**, **backend**, **qdrant**, and **ollama**.

### Prerequisites

- **Docker Desktop** (includes Docker Compose)
- **~8 GB RAM** (Qwen3 models are ~2.5 GB each)
- **Git** for cloning the repository

### Required Environment Variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

Key variables (see `.env.example` for all options):

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Ollama service URL (Docker service name) |
| `LLM_MODEL` | `qwen3:4b` | Generation model |
| `EMBEDDING_MODEL` | `qwen3-embedding:4b` | Embedding model |
| `QDRANT_URL` | `http://qdrant:6333` | Qdrant service URL (Docker service name) |
| `QDRANT_COLLECTION` | `zx_bank_knowledge` | Vector collection name |
| `CHUNK_SIZE` | `1600` | Characters per chunk |
| `RETRIEVAL_K` | `8` | Chunks after deduplication |

### Docker Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│  Frontend   │──────│   Backend    │──────│   Qdrant    │
│  Nginx:8080 │      │ FastAPI:8000 │      │ Vector DB:  │
│  (React)    │      │   Python     │      │  6333       │
└─────────────┘      └──────────────┘      └─────────────┘
       │                    │                   ▲
       │                    └───────────────────┘
       │                     Ollama:11434
       ▼                    (LLM + Embeddings)
   Browser
```

- **Frontend**: React + Vite build served by Nginx, proxies `/api` → backend
- **Backend**: FastAPI + Uvicorn, connects to Qdrant and Ollama via service names
- **Qdrant**: Vector database with persistent `qdrant_data` volume
- **Ollama**: LLM server with persistent `ollama_data` volume (models persist)

### Build Commands

```bash
# Validate configuration
docker compose config

# Build all images
docker compose build
```

### Startup Commands

```bash
# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f qdrant
docker compose logs -f ollama
```

### Ollama Model Installation

Models are **not** downloaded automatically on startup. After starting the
containers, pull the required models:

```bash
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama pull qwen3-embedding:4b

# Verify
docker compose exec ollama ollama list
```

Models persist in the `ollama_data` named volume across container restarts.

### Application Access URLs

| Service | URL |
|---------|-----|
| Frontend UI | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Qdrant | http://localhost:6333 |
| Ollama API | http://localhost:11434 |

### Container Management

```bash
# Restart all services
docker compose restart

# Restart a specific service
docker compose restart backend

# Stop all services (preserves data)
docker compose stop

# Stop and remove containers (preserves named volumes)
docker compose down

# Stop and remove everything INCLUDING named volumes (DELETES DATA)
docker compose down -v
```

### Logs and Troubleshooting

```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs -f backend
docker compose logs -f frontend
docker compose logs -f qdrant
docker compose logs -f ollama
```

### Persistent Volumes

| Volume | Purpose | Survives `docker compose down`? | Survives `docker compose down -v`? |
|--------|---------|---------------------------------|------------------------------------|
| `qdrant_data` | Qdrant vectors and collections | Yes | **No** (deleted) |
| `ollama_data` | Ollama models and blobs | Yes | **No** (deleted) |

**Warning:** `docker compose down -v` removes named volumes and will delete
all Qdrant vectors and Ollama models. Use `docker compose down` (without `-v`)
for normal shutdowns to preserve data.

### PDF Upload Behavior

Uploaded PDFs are processed temporarily and **deleted after indexing**. The
extracted text chunks are stored in Qdrant (persistent in `qdrant_data`).
Uploaded documents are indexed into session-scoped collections
(`zx_bank_uploads_<session_id>`) and are deleted when the conversation is
cleared. No persistent upload directory is used.

### Shutdown Instructions

```bash
# Normal shutdown (preserves all data)
docker compose down

# Complete cleanup (DELETES all vectors and models)
docker compose down -v
```

---

## Git Rules

- Do **not** commit `dataset/`, `.env`, `.venv/`, `qdrant_storage/`,
  `frontend/node_modules`, or generated secrets.
- `.gitignore` already excludes these paths.

- Do **not** commit `dataset/`, `.env`, `.venv/`, `qdrant_storage/`,
  `frontend/node_modules`, or generated secrets.
- `.gitignore` already excludes these paths.

---

## Learning Notes

See `LEARNING_NOTES.md` for explanations of LangChain concepts: Document
objects, loaders, table extraction, text splitters, context headers,
embeddings, vector stores, hybrid retrieval, BM25, deduplication, neighbour
expansion, ChatPromptTemplate, MessagesPlaceholder, LCEL, Qwen3 reasoning
budgets, and evaluation metrics.
