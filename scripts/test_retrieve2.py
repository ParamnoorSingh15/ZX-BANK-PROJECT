from src.retrieval.retrievers import get_retriever

r = get_retriever("dense", k=4)
docs = r.invoke("Where is ZX Bank headquartered and what is its SWIFT code?")
print("Results:", len(docs))
for d in docs:
    m = d.metadata or {}
    title = m.get("title")
    text = d.page_content.strip()
    print(f"  title={title!r} content_len={len(text)} preview={text[:80]}")
