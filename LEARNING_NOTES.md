# Learning Notes — LangChain RAG Concepts

These notes explain the LangChain concepts implemented in the ZX Bank RAG application,
what each component does, why it's needed, which class/method is used, how data flows,
what alternatives exist, and how to test it.

---

## 1. Document Objects and Loaders

### What it does
A LangChain `Document` is the core data structure: `page_content` (text) + `metadata` (dict).
Loaders convert files (PDF, HTML, DOCX, PPTX, MD) into `Document` objects.

### Why we need it
RAG starts with unstructured files. Loaders normalize them into a common format so the
rest of the pipeline (split, embed, store) can treat every document uniformly.

### LangChain class/method
- `langchain_core.documents.Document(page_content=..., metadata=...)`
- Custom loader functions in `src/ingestion/loaders.py` (we do **not** use the deprecated
  `langchain_community` loaders; we call `pypdf`, `beautifulsoup4`, `docx2txt`,
  `python-pptx` directly and wrap output in `Document`).

### Information flow
```
file on disk -> library text extraction -> Document(page_content, metadata)
```

### Metadata preserved
`source` (file path), `format`, `title` (filename), `page`/`slide` number, `chunk_index`,
`content_hash`.

### Alternatives
- `langchain_community.document_loaders` (sunset, deprecated — avoided)
- `UnstructuredPDFLoader`, `UnstructuredHTMLLoader`, etc. (heavier dependency)

### How to test
`tests/test_loaders.py` verifies each loader returns Documents with correct metadata and
that HTML tags are stripped.

---

## 2. Text Splitters

### What it does
Splits long documents into smaller overlapping chunks so they fit within the LLM's
context window and embed well.

### Why we need it
Documents can be thousands of tokens. Embedding models and LLMs have input limits.
Chunking also improves retrieval precision: smaller, topically-focused chunks match
queries better than whole documents.

### LangChain class/method
- `langchain_text_splitters.RecursiveCharacterTextSplitter(chunk_size=..., chunk_overlap=...)`
- Used in `src/ingestion/splitter.py`

### Information flow
```
[Document(cleaned text)] -> split_documents() -> [Document(chunk)]
```
Each chunk inherits source metadata and gains a `chunk_index`.

### Split strategy
`RecursiveCharacterTextSplitter` tries separators in order: `["\n\n", "\n", " ", ""]`,
keeping related text together. Better than naive fixed-size splitting because it
respects paragraph/sentence boundaries.

### Alternatives
- `CharacterTextSplitter` (fixed size, less context-aware)
- `TokenTextSplitter` (splits by token count)
- `NLTKTextSplitter`, `SpacyTextSplitter` (language-aware)

### How to test
`tests/test_loaders.py::test_split_documents_adds_chunk_index` verifies chunks get
sequential `chunk_index` and inherit source metadata.

---

## 3. Embeddings

### What it does
Converts text chunks into dense vectors (embeddings) so semantic similarity can be
computed via vector distance.

### Why we need it
Vector search is the core of retrieval. Similar meanings should have similar vectors,
so a query can find relevant chunks even without keyword overlap.

### LangChain class/method
- `langchain_ollama.OllamaEmbeddings(model="qwen3-embedding:4b")`
- Wrapped in `src/embeddings/client.py:get_embeddings()`

### Critical decision: dimension is probed, not hardcoded
`src/embeddings/client.py:probe_dimension()` embeds a probe string and measures
`len(vector)`. Qwen3-Embedding-4B outputs **2560** dimensions (observed at runtime).
We use this to create the Qdrant collection, guaranteeing indexing and query vectors
match.

### Information flow
```
settings (model name) -> OllamaEmbeddings -> embed_query(text) -> vector (2560-dim)
```

### Same model for index and query
Both indexing (`embed_documents`) and querying (`embed_query`) use
`qwen3-embedding:4b`. Using different models would break similarity search.

### Alternatives
- `langchain_openai.OpenAIEmbeddings`, `langchain_huggingface.HuggingFaceEmbeddings`
- SentenceTransformers, Cohere, etc.

### How to test
`src/embeddings/client.py:probe_dimension()` returns 2560. Verified during ingestion.

---

## 4. Vector Store (Qdrant)

### What it does
Stores embedding vectors with their text and metadata, and supports similarity search.

### Why we need it
We need to persist embeddings and retrieve the top-k most similar chunks for a query.
Qdrant also supports metadata filtering, MMR, and hybrid search.

### LangChain class/method
- `langchain_qdrant.QdrantVectorStore` (available but we use `qdrant_client.QdrantClient`
  directly via `src/vectorstore/qdrant_store.py`)

### Why we use QdrantClient directly
Qdrant only builds its HNSW index once a collection crosses `indexing_threshold`
(default 10,000 points). With fewer points, `QdrantVectorStore.similarity_search`
can return empty hits. `QdrantClient.query_points` does brute-force cosine search,
so retrieval works at any collection size.

### Information flow
```
[chunk Documents] -> embed each chunk -> upsert into Qdrant
    (vector, payload = {text, metadata...})
query -> embed query -> QdrantClient.query_points -> [Document]
```

### Duplicate detection
Content hash (SHA-256) of each chunk's text. Deterministic UUID v5 IDs are derived
from the hash, so re-indexing overwrites existing points instead of duplicating.

### Alternatives
- Chroma, FAISS, Pinecone, Weaviate, Milvus

### How to test
`tests/test_pipeline.py` verifies chunk output. Integration: `scripts/ingest.py`
indexes into Qdrant; `scripts/test_retrieve2.py` verifies search returns real docs.

---

## 5. Retrievers

### What it does
A retriever takes a query string and returns relevant `Document` objects.

### Why we need it
It abstracts the search mechanism so the RAG chain can use dense search, MMR, or
filtered search interchangeably.

### LangChain class/method
- `langchain_core.retrievers.BaseRetriever` with `_get_relevant_documents()`
- `src/retrieval/retrievers.py`: `DenseRetriever`, `MMRRetriever`,
  `MetadataFilteredRetriever`

### Strategies
| Strategy | Description |
|----------|-------------|
| Dense | Cosine similarity to query vector |
| MMR | Relevance + diversity (avoids near-duplicate chunks) |
| Filtered | Similarity + Qdrant metadata filter (e.g. by format/source) |

### Information flow
```
user query -> retriever.invoke(query) -> QdrantClient.query_points -> [Document]
```

### How to test
`scripts/test_retrieve2.py` calls `get_retriever("dense", k=4).invoke(query)` and
checks returned Documents have content and metadata.

---

## 6. ChatPromptTemplate & PromptTemplate

### What it does
Structures LLM input as messages (system, human, AI) instead of one raw string.

### Why we need it
Chat models like Qwen3 expect a conversation format. `ChatPromptTemplate` lets us
inject context, question, and chat history as separate message roles.

### LangChain class/method
- `langchain_core.prompts.ChatPromptTemplate.from_messages([...])`
- `langchain_core.prompts.PromptTemplate(input_variables=[...], template=...)`
- `src/prompts/__init__.py`

### MessagesPlaceholder
A special placeholder that injects a variable-length list of messages (e.g. prior
conversation turns) into the prompt. Essential for conversational RAG.

### Information flow
```
{question} + {context} + {chat_history}
    -> ChatPromptTemplate.format() -> [messages]
    -> ChatOllama.invoke(messages) -> answer
```

### Alternatives
- Plain string formatting (f-strings) — loses message structure
- `langchain.prompts.PromptTemplate` (deprecated in favor of core)

### How to test
`src/prompts/__init__.py` exports `get_rag_prompt()` and `get_simple_rag_prompt()`.
Verified by `src/chains` using them.

---

## 7. LCEL (LangChain Expression Language)

### What it does
Composes runnables into a pipeline using `|` (pipe) operator.

### Why we need it
LCEL makes the RAG pipeline declarative: retrieval, formatting, prompt, LLM, output
parsing are chained together into a single callable `Runnable`.

### Key classes
- `RunnableParallel` — runs multiple branches in parallel (e.g. retrieve context +
  pass question through)
- `RunnablePassthrough` — passes a field through unchanged
- `RunnableLambda` — wraps a plain Python function as a runnable
- `StrOutputParser` — extracts string content from an LLM message

### Information flow (basic RAG)
```
{question}
    -> RunnableParallel({
          context: retriever | format_docs,
          question: RunnablePassthrough(),
          chat_history: ...
        })
    -> ChatPromptTemplate
    -> ChatOllama(qwen3:4b)
    -> StrOutputParser
    -> answer string
```

### Alternatives
- Imperative function calls (less composable, harder to inspect)
- LangGraph (for stateful, cyclic workflows — overkill for basic RAG)

### How to test
`src/chains/__init__.py:BasicRAGChain` builds the LCEL chain. Verified end-to-end via
`scripts/test_eval.py` and manual `invoke()` calls.

---

## 8. ChatOllama

### What it does
Calls a chat model served by Ollama (here `qwen3:4b`).

### Why we need it
The LLM generates the final grounded answer from retrieved context.

### LangChain class/method
- `langchain_ollama.ChatOllama(model="qwen3:4b", temperature=0)`
- `temperature=0` for deterministic, factual answers

### Information flow
```
[messages from ChatPromptTemplate] -> ChatOllama.invoke() -> AIMessage
```

### Alternatives
- `ChatOpenAI`, `ChatAnthropic`, `langchain_huggingface.ChatHuggingFace`

### How to test
Ollama must be running with `qwen3:4b` pulled. Test: `ChatOllama(...).invoke("Hi")`.

---

## 9. Conversational RAG

### What it does
Remembers prior turns so follow-up questions can reference earlier context.

### Why we need it
Banking Q&A is multi-turn. A user may ask "What about savings accounts?" then
"What's the minimum balance?" — the second needs the first's context.

### Implementation
- `src/chains/__init__.py:ConversationalRAGChain`
- Prompt uses `MessagesPlaceholder(variable_name="chat_history")`
- History stored as list of `(role, content)` tuples; in FastAPI, keyed by `session_id`

### Information flow
```
[history messages] + [new question] + [retrieved context]
    -> ChatPromptTemplate (with MessagesPlaceholder)
    -> ChatOllama -> answer
    -> append to history
```

### Alternatives
- `langchain.memory.ConversationBufferMemory` (legacy)
- `LangGraph` with checkpointer (more robust, persistent)

### How to test
FastAPI `/ask` with `use_conversation=true` and `session_id`; verify history accumulates.

---

## 10. Evaluation Metrics

### Retrieval metrics (`src/evaluation/metrics.py`)
- **Recall@K**: fraction of relevant docs found in top-K
- **Precision@K**: fraction of top-K retrieved that are relevant
- **MRR**: reciprocal rank of first relevant doc
- **nDCG@K**: discounted cumulative gain normalized by ideal

### Generation metrics
- **Faithfulness**: answer supported by context (heuristic term overlap)
- **Answer relevancy**: answer addresses the question
- **Correctness**: answer matches expected answer

### RAGAS
`try_ragas_evaluation()` wraps RAGAS `faithfulness`, `answer_relevancy`,
`correctness` if RAGAS is installed and configurable.

### Information flow
```
benchmark question + expected doc
    -> retriever -> retrieved ids -> Recall@K/Precision@K/MRR/nDCG
question + context -> RAG chain -> answer -> RAGAS/heuristics
```

### Experimental vs held-out
`EvalConfig` captures chunk size, strategy, k — enabling comparison across configs.
Results are computed live, never fabricated.

### How to test
`scripts/test_eval.py` runs `run_evaluation(EvalConfig(k=4))` and prints metrics.

---

## 11. FastAPI Backend

### What it does
Exposes the RAG pipeline as REST endpoints.

### Endpoints
- `POST /ask` — question -> answer + sources
- `GET /health` — Qdrant/collection status
- `GET /collections` — list Qdrant collections
- `POST /documents/upload` — upload + ingest a document
- `DELETE /conversations/{session_id}` — clear chat history

### Why FastAPI
Automatic OpenAPI/Swagger docs, Pydantic validation, async support, easy CORS.

### Information flow
```
HTTP request -> FastAPI endpoint -> RAG chain -> JSON {answer, sources}
```

### How to test
`uvicorn src.api.server:app --reload --port 8000` then hit `/docs`.

---

## 12. Testing Strategy

- **Unit tests** (`tests/test_loaders.py`): loaders, cleaning, splitting — no external services.
- **Integration tests** (`tests/test_pipeline.py`): full load-clean-split on real dataset.
- **Manual/end-to-end**: `scripts/ingest.py`, `scripts/test_retrieve2.py`, `scripts/test_eval.py`.
- Run all unit tests: `python -m pytest tests/`
