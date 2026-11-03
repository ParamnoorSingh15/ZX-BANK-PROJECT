from dotenv import load_dotenv
import os

from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient

load_dotenv()

# 1. Initialize LLM
llm = ChatOllama(
    model=os.getenv("LLM_MODEL"),
    temperature=0
)

# 2. Initialize embedding model
embeddings = OllamaEmbeddings(
    model=os.getenv("EMBEDDING_MODEL")
)

# 3. Initialize Qdrant client
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL")
)

# Test LLM
response = llm.invoke(
    "Explain what a savings account is in two sentences."
)

print("\nLLM RESPONSE:")
print(response.content)

# Test embeddings
vector = embeddings.embed_query(
    "What is a savings account?"
)

print("\nEMBEDDING INFORMATION:")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])

# Test Qdrant
collections = qdrant.get_collections()

print("\nQDRANT CONNECTION:")
print(collections)