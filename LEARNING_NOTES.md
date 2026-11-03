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
`source` (file path), `format`, `title` (filename), `page`/`slide` number, `chunk_index`.

### Extracting *every* word (tables, charts, notes)
Bank documents store most of their hard facts in tables (FD rates, branch/ATM lists,
eligibility grids). A loader that only reads plain text frames silently drops those
tables, and the retriever then finds a heading like
`Sample Interest Rates (for illustrative purposes)` but never the numbers underneath
it — so the model answers "the context does not specify the rates".

The loaders therefore extract every container that can hold text:

| Format | What is extracted |
|--------|-------------------|
| PPTX | text frames, nested **group shapes**, **tables** (one pipe-separated row per line), **charts** (title, categories, series values), **speaker notes** |
| HTML | visible text, with every `<table>` replaced by pipe-separated rows before extraction so cells stay with their row |
| PDF | page text, with a retry in `extraction_mode="layout"` when normal extraction returns nothing |
| DOCX | all paragraph **and table cell** text (`docx2txt` walks the whole XML) |
| MD | raw markdown, tables intact |

### Alternatives
- `langchain_community.document_loaders` (sunset, deprecated — avoided)
- `UnstructuredPDFLoader`, `UnstructuredHTMLLoader`, etc. (heavier dependency)
- `unstructured` or `camelot` for advanced table parsing (extra system dependencies)

### How to test
`tests/test_loaders.py` verifies each loader returns Documents with correct metadata and
that HTML tags are stripped. To confirm table extraction manually, load a rate table and
check the numbers are present:

```python
from src.ingestion.loaders import load_pptx
docs = load_pptx("dataset/ZX Bank/pptx/ZX Bank Fixed Deposits.pptx")
print(any("3.50%" in d.page_content for d in docs))   # True
```

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
- `langchain_text_splitters.RecursiveCharacterTextSplitter(chunk_size=1600, chunk_overlap=200)`
- Used in `src/ingestion/splitter.py`

### Information flow
```
[Document(cleaned text)] -> split_documents() -> [Document(chunk)]
```
Each chunk inherits source metadata and gains `chunk_index`, `chunk_total` and `section`.

### Split strategy
`RecursiveCharacterTextSplitter` tries separators in order: `["\n\n", "\n", " ", ""]`,
keeping related text together. Better than naive fixed-size splitting because it
respects paragraph/sentence boundaries.

**Why 1600 characters with 200 overlap:** the previous 400-character setting split a
heading away from the table beneath it. A rate table would end up in a different chunk
from its caption, so retrieval returned only the caption. 1600 characters keeps a
heading, its table and the surrounding explanation in one chunk.

### Merging tiny chunks
Headings, slide titles and captions can become chunks of a few characters. They embed
strongly against a question that mentions the heading while carrying none of the facts,
which produces the classic *"the context does not specify"* answer.
`_merge_tiny_chunks()` merges any chunk below `min_chunk_size` (200 chars) into its
neighbour (or the previous chunk for a trailing footer), without crossing documents.

### Context headers
Every chunk is prefixed with:

```
Document: <title>
Section: <heading>

<original chunk text>
```

This puts document identity into the embedded text. A question like *"where is ZX Bank
headquartered?"* now matches the `About ZX Bank Ltd` chunks rather than an arbitrary
branch page, and the LLM gets a ready-made citation label. The heading is also stored as
`section` metadata for UI citations and section-level boosting during retrieval.

### Alternatives
- `CharacterTextSplitter` (fixed size, less context-aware)
- `TokenTextSplitter` (splits by token count)
- `NLTKTextSplitter`, `SpacyTextSplitter` (language-aware)
- Semantic chunking with embeddings (slower, more complex)

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

### Resetting after a chunking change
`reset_collection()` deletes the collection before ingestion. This matters whenever the
chunk size, the loaders or the embedding model change: without a reset, old 400-character
fragments stay in the collection next to the new 1600-character chunks, and retrieval
returns a mix of stale and fresh text. Use it from the CLI:

```powershell
python scripts/ingest.py --reset
```

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
- `src/retrieval/retrievers.py`: `HybridRetriever` (default), `DenseRetriever`,
  `MMRRetriever`, `MetadataFilteredRetriever`

### Strategies
| Strategy | Description |
|----------|-------------|
| Hybrid (default) | Dense cosine search **+ BM25 keyword search**, merged, deduplicated, neighbour-expanded |
| Dense | Cosine similarity to query vector |
| MMR | Relevance + diversity (avoids near-duplicate chunks) |
| Filtered | Similarity + Qdrant metadata filter (e.g. by format/source) |

### Hybrid retrieval (dense + BM25)
Dense embeddings blur exact strings: `3.50%`, `IFSC`, `Mumbai`, a locker number or a
branch name can all end up "close" to each other. BM25 keyword scoring is the opposite —
it is precise on literal terms but blind to paraphrases.

`HybridRetriever` runs both and merges them:

1. Fetch `fetch_k` (40) dense candidates from Qdrant.
2. Score the corpus with BM25 over the same payloads, keep the top 40.
3. Normalise each signal to 0–1 and combine:
   `score = (1 - keyword_weight) * dense + keyword_weight * bm25`
   (`RETRIEVAL_KEYWORD_WEIGHT=0.35` favours dense while still rewarding exact terms).
4. Add a small boost when query terms appear in the chunk's `title` or `section`.

The BM25 index and tokenised corpus are built once per collection from the Qdrant
payloads and cached in memory (`_CORPUS_CACHE`). `reset_corpus_cache()` is called after an
upload or re-ingestion so the keyword index reflects the new points.

### Deduplicating the five formats
The dataset stores each of its 69 documents in five formats (pdf, html, docx, pptx, md).
All formats are indexed so no content is lost, but without dedupe the top-8 slots fill
with five copies of the same paragraph. `_dedupe_select()`:

- drops a candidate whose token set has Jaccard similarity **> 0.85** with an
  already-selected chunk (near-duplicate across formats),
- caps each document title at `RETRIEVAL_MAX_PER_TITLE` (3) chunks.

### Neighbour expansion ("whole context")
A single chunk can still be one half of a section. After selection, each top hit is
expanded with the chunks immediately before/after it in the same source document
(`RETRIEVAL_EXPAND_NEIGHBORS=true`). This gives the LLM the complete section — e.g. the
heading, the rate table and the explanatory note — instead of a fragment.

### Tuning knobs
| Env var | Default | Effect |
|---------|---------|--------|
| `RETRIEVAL_STRATEGY` | `hybrid` | `hybrid`, `dense`, `mmr`, `filtered` |
| `RETRIEVAL_K` | `8` | Chunks returned after dedupe |
| `RETRIEVAL_FETCH_K` | `40` | Candidates fetched before merging |
| `RETRIEVAL_KEYWORD_WEIGHT` | `0.35` | BM25 weight in the hybrid score |
| `RETRIEVAL_EXPAND_NEIGHBORS` | `true` | Add neighbouring chunks |
| `RETRIEVAL_MAX_PER_TITLE` | `3` | Max chunks per document title |

### Information flow
```
user query
    -> embed_query + tokenize
    -> Qdrant dense candidates ─┐
    -> BM25 keyword candidates ─┤
                                -> normalise + merge + title/section boost
                                -> dedupe (Jaccard) + per-title cap
                                -> neighbour expansion
                                -> [Document]  (8 selected + up to 4 neighbours)
```

### How to test
```python
from src.retrieval.retrievers import get_retriever
docs = get_retriever("hybrid", k=8).invoke("What are the interest rates on fixed deposits?")
print([d.metadata.get("title") for d in docs])
```
The returned list should contain several distinct titles (dedupe working) and at least one
chunk whose text contains the actual rate numbers.

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
- `langchain_ollama.ChatOllama(model="qwen3:4b", temperature=0.3, num_ctx=16384, num_predict=3072, reasoning=True)`
- `src/chains/__init__.py:_build_llm()`

### Qwen3 reasoning budgets (important)
Qwen3 is a *thinking* model: it generates reasoning tokens before the final answer.
Two failure modes were observed on this Ollama build:

1. **Empty answers.** With a small `num_predict` (e.g. 160), the reasoning consumes the
   entire output budget and `content` comes back `''` with `done_reason="length"`.
2. **Chain-of-thought leak.** Setting `think=false` / `reasoning=False` does not remove
   the reasoning on this build — it leaks the raw "Hmm, the user is asking…" text into
   the answer instead.

The fix is to keep reasoning **on** and give it room:
`num_ctx=16384`, `num_predict=3072`, `temperature=0.3`.
`_invoke_with_retry()` additionally retries once with `num_predict=6144` and
`num_ctx=32768` if an answer still comes back empty.

Trade-off: answers take roughly 20–60 seconds. The React UI shows a live "thinking"
timer so the wait is visible.

### Information flow
```
[messages from ChatPromptTemplate] -> ChatOllama.invoke()
    -> reasoning tokens (hidden) -> final answer -> AIMessage(content)
```

### Alternatives
- `ChatOpenAI`, `ChatAnthropic`, `langchain_huggingface.ChatHuggingFace`
- Non-reasoning models (e.g. `qwen2.5`) for faster responses at some accuracy cost

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

## 12. React Frontend (Markdown Answers)

### What it does
A Vite + React chat UI that sends questions to `/ask` and renders the returned answer.

### Why Markdown rendering
The detailed-answer prompt asks the model to use headings, bullets and Markdown tables
for things like interest-rate grids. Rendering that as plain text shows literal `**`,
`|` and `##`, which looks broken and is hard to read.

The UI uses:

- `react-markdown` — converts the answer string into React elements
- `remark-gfm` — adds GitHub-Flavored Markdown support (tables, strikethrough, task lists)
- `App.css` `.markdown-body` rules — styled headings, lists, `code`, `blockquote` and
  scrollable tables

Because `react-markdown` does not inject raw HTML by default, model output cannot inject
scripts — a useful safety property for a chat UI.

### Information flow
```
question -> POST /api/ask -> {answer, sources}
    -> React state -> <ReactMarkdown remarkPlugins={[remarkGfm]}>
    -> rendered headings / bullets / tables + source list
```

### UX details
- Default retrieval strategy is **hybrid**; the selector also offers dense and MMR.
- `k` is sent as 8 so deduplication still leaves enough distinct chunks.
- `/health` is polled every 10s, so the status pill recovers when the backend starts.
- A live timer counts seconds while Qwen3 reasons, setting expectations for the
  20–60 second response time.
- Failures show a retry button instead of a silent blank bubble.

### How to test
```bash
cd frontend
npm run dev
```
Ask "What are the interest rates on fixed deposits?" — the answer should show a real
table, not `| ... |` text.

---

## 13. Testing Strategy

- **Unit tests** (`tests/test_loaders.py`): loaders, cleaning, splitting — no external services.
- **Integration tests** (`tests/test_pipeline.py`): full load-clean-split on real dataset.
- **Manual/end-to-end**: `scripts/ingest.py`, retriever checks, the `/ask` endpoint and the
  React UI.
- Run all unit tests: `python -m pytest tests/`
