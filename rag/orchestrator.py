from .ingestion import DocumentIngester
from .chunking import SmartChunker
from .embedding import EmbeddingEngine
from .vector_store import VectorStore

class RAGPipeline:
    def __init__(self):
        self.ingester = DocumentIngester()
        self.chunker = SmartChunker()
        self.embedder = EmbeddingEngine()
        self.vdb = VectorStore()

    def process_document(self, filepath: str, paper_id: int, user_id: int, title: str):
        data = self.ingester.ingest(filepath)
        chunks = self.chunker.chunk_document(data['pages'], paper_id, user_id, title, filepath)
        texts = [c['content'] for c in chunks]
        embeddings = self.embedder.encode(texts)
        self.vdb.add_chunks(chunks, embeddings)
        return len(chunks)
