from src.vectorstore.qdrant_store import get_vector_store
from src.embeddings.client import get_embeddings

emb = get_embeddings()
store = get_vector_store(embeddings=emb)
print("Collection:", store.collection_name)

results = store.similarity_search("ZX Bank headquarters Mumbai SWIFT", k=4)
print("Results:", len(results))
for d in results:
    m = d.metadata or {}
    title = m.get("title")
    text = d.page_content.strip()
    print(f"  title={title!r} content_len={len(text)} preview={text[:60]}")
