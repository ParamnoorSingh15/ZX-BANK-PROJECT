from qdrant_client import QdrantClient
from qdrant_client.http.models import OptimizersConfigDiff

client = QdrantClient(url="http://localhost:6333")
client.update_collection(
    collection_name="zx_bank_knowledge",
    optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
)
print("Updated indexing threshold to 0")
info = client.get_collection("zx_bank_knowledge")
print("Indexed vectors:", info.indexed_vectors_count)
print("Points:", info.points_count)
