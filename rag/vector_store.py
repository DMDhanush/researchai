import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any

class VectorStore:
    def __init__(self, collection_name: str = "research_docs", path: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, 
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: Any):
        ids = [f"{c['paper_id']}_{c['chunk_index']}" for c in chunks]
        metadatas = [{"paper_id": c['paper_id'], "page_number": c['page_number'], "section": c['section']} for c in chunks]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            metadatas=metadatas,
            documents=[c['content'] for c in chunks]
        )

    def query(self, query_embedding: List[float], n_results: int = 5):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
