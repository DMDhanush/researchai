"""
ResearchAI — RAG Retriever Module
Handles semantic retrieval with validation, user/paper isolation, and optional reranking.
"""
import time
import json
from typing import Optional, List, Dict, Any
import numpy as np
from contextlib import contextmanager

from database.database import get_session
from database.models import Paper, PaperChunk
from .vector_store import VectorStore
from .embedding import EmbeddingEngine


class RAGRetriever:
    """Retrieves and validates chunks with strict user/paper isolation."""

    def __init__(self):
        self.vector_store = VectorStore()
        self.embedding_engine = EmbeddingEngine()
        self.similarity_threshold = 0.3  # Minimum cosine similarity

    def retrieve(
        self,
        query: str,
        user_id: int,
        paper_ids: Optional[List[int]] = None,
        top_k: int = 5,
        include_reranking: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant chunks for a query with strict isolation.

        Args:
            query: User's question/query
            user_id: Current user ID (for isolation)
            paper_ids: List of paper IDs to search (None = all user's papers)
            top_k: Number of results to return
            include_reranking: Whether to apply reranking

        Returns:
            {
                'chunks': [list of chunk dicts with metadata],
                'scores': [similarity scores],
                'candidate_count': int,
                'retrieved_count': int,
                'avg_score': float,
                'top_score': float,
                'paper_ids_searched': [list of paper IDs]
            }
        """
        start_time = time.time()

        with get_session() as db:
            # Determine scope: if paper_ids specified, use those; else all user's papers
            if paper_ids:
                search_paper_ids = paper_ids
            else:
                user_papers = db.query(Paper).filter(Paper.user_id == user_id).all()
                search_paper_ids = [p.id for p in user_papers]

            if not search_paper_ids:
                return {
                    'chunks': [],
                    'scores': [],
                    'candidate_count': 0,
                    'retrieved_count': 0,
                    'avg_score': 0.0,
                    'top_score': 0.0,
                    'paper_ids_searched': []
                }

            # Encode query
            query_embedding = self.embedding_engine.encode_single(query)

            # Query vector store
            try:
                results = self.vector_store.query(
                    query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
                    n_results=top_k * 2  # Get more to filter after validation
                )
            except Exception as e:
                # Fallback to NumPy cosine similarity if ChromaDB fails
                return self._fallback_retrieval(
                    query, user_id, search_paper_ids, top_k, db
                )

            # Validate and filter results
            valid_chunks = []
            valid_scores = []

            if results and results['documents'] and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []

                for doc_idx, (doc, metadata) in enumerate(zip(documents, metadatas)):
                    if not metadata:
                        continue

                    paper_id = metadata.get('paper_id')

                    # CRITICAL: Paper ID must be in search scope
                    if paper_id not in search_paper_ids:
                        continue

                    # Convert distance to similarity (ChromaDB uses cosine distance)
                    distance = distances[doc_idx] if doc_idx < len(distances) else 0.0
                    similarity = 1.0 - distance  # Convert distance to similarity

                    if similarity < self.similarity_threshold:
                        continue

                    # Fetch full chunk from DB for validation
                    chunk = db.query(PaperChunk).filter(
                        PaperChunk.paper_id == paper_id,
                        PaperChunk.user_id == user_id,
                        PaperChunk.content == doc
                    ).first()

                    if not chunk:
                        continue

                    # Validate chunk
                    if not self._validate_chunk(chunk, user_id, search_paper_ids):
                        continue

                    # Get paper for title
                    paper = db.query(Paper).filter(
                        Paper.id == paper_id,
                        Paper.user_id == user_id
                    ).first()

                    if not paper:
                        continue

                    chunk_dict = {
                        'id': chunk.id,
                        'paper_id': chunk.paper_id,
                        'paper_title': paper.title or paper.filename,
                        'page_number': chunk.page_number,
                        'section': chunk.section or 'Unknown',
                        'content': chunk.content,
                        'word_count': chunk.word_count,
                        'similarity': float(similarity)
                    }

                    valid_chunks.append(chunk_dict)
                    valid_scores.append(float(similarity))

                    if len(valid_chunks) >= top_k:
                        break

            # Rerank if requested and we have results
            if include_reranking and valid_chunks:
                valid_chunks, valid_scores = self._rerank(
                    valid_chunks, query, top_k
                )

            avg_score = float(np.mean(valid_scores)) if valid_scores else 0.0
            top_score = float(np.max(valid_scores)) if valid_scores else 0.0

            elapsed_ms = (time.time() - start_time) * 1000

            return {
                'chunks': valid_chunks,
                'scores': valid_scores,
                'candidate_count': len(documents) if results and results['documents'] else 0,
                'retrieved_count': len(valid_chunks),
                'avg_score': avg_score,
                'top_score': top_score,
                'paper_ids_searched': search_paper_ids,
                'elapsed_ms': elapsed_ms
            }

    def _validate_chunk(
        self,
        chunk: PaperChunk,
        user_id: int,
        paper_ids: List[int]
    ) -> bool:
        """
        Validate a chunk before including in results.

        Checks:
        1. Chunk has content (not empty)
        2. User ID matches (isolation)
        3. Paper ID is in search scope
        4. Page number is valid (not None)
        5. Not obvious bibliography/references noise
        """
        # Content must exist and be non-empty
        if not chunk.content or len(chunk.content.strip()) < 20:
            return False

        # User isolation
        if chunk.user_id != user_id:
            return False

        # Paper scope isolation
        if chunk.paper_id not in paper_ids:
            return False

        # Page number should exist
        if chunk.page_number is None or chunk.page_number <= 0:
            return False

        # Reject obvious bibliography/reference noise
        if chunk.section and chunk.section.lower() == 'references':
            # Accept references only if explicitly asked for
            return False

        # Reject very short chunks
        if chunk.word_count and chunk.word_count < 10:
            return False

        return True

    def _rerank(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        top_k: int
    ) -> tuple[List[Dict[str, Any]], List[float]]:
        """
        Rerank chunks using cross-encoder style semantic similarity.
        Re-embed chunks and compute similarity to query.
        """
        try:
            chunk_texts = [c['content'] for c in chunks]
            chunk_embeddings = self.embedding_engine.encode(chunk_texts)
            query_embedding = self.embedding_engine.encode_single(query)

            # Compute cosine similarities
            scores = []
            for chunk_emb in chunk_embeddings:
                similarity = float(np.dot(query_embedding, chunk_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)
                ))
                scores.append(similarity)

            # Sort by score descending
            sorted_pairs = sorted(
                zip(chunks, scores),
                key=lambda x: x[1],
                reverse=True
            )

            # Return top k
            reranked_chunks = [c for c, _ in sorted_pairs[:top_k]]
            reranked_scores = [s for _, s in sorted_pairs[:top_k]]

            return reranked_chunks, reranked_scores
        except Exception as e:
            # If reranking fails, return as-is
            return chunks[:top_k], [c['similarity'] for c in chunks[:top_k]]

    def _fallback_retrieval(
        self,
        query: str,
        user_id: int,
        paper_ids: List[int],
        top_k: int,
        db
    ) -> Dict[str, Any]:
        """
        Fallback retrieval using NumPy cosine similarity when ChromaDB unavailable.
        """
        # Fetch all chunks for the user's papers
        chunks = db.query(PaperChunk).filter(
            PaperChunk.user_id == user_id,
            PaperChunk.paper_id.in_(paper_ids)
        ).all()

        if not chunks:
            return {
                'chunks': [],
                'scores': [],
                'candidate_count': 0,
                'retrieved_count': 0,
                'avg_score': 0.0,
                'top_score': 0.0,
                'paper_ids_searched': paper_ids
            }

        # Encode query
        query_embedding = self.embedding_engine.encode_single(query)

        # Encode all chunks and compute similarities
        scores = []
        valid_chunk_dicts = []

        for chunk in chunks:
            try:
                chunk_embedding = self.embedding_engine.encode_single(chunk.content)
                similarity = float(np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                ))

                if similarity >= self.similarity_threshold:
                    paper = db.query(Paper).filter(Paper.id == chunk.paper_id).first()
                    if paper:
                        chunk_dict = {
                            'id': chunk.id,
                            'paper_id': chunk.paper_id,
                            'paper_title': paper.title or paper.filename,
                            'page_number': chunk.page_number,
                            'section': chunk.section or 'Unknown',
                            'content': chunk.content,
                            'word_count': chunk.word_count,
                            'similarity': similarity
                        }
                        valid_chunk_dicts.append(chunk_dict)
                        scores.append(similarity)
            except Exception:
                continue

        # Sort by score descending
        sorted_pairs = sorted(
            zip(valid_chunk_dicts, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top k
        result_chunks = [c for c, _ in sorted_pairs[:top_k]]
        result_scores = [s for _, s in sorted_pairs[:top_k]]

        avg_score = float(np.mean(result_scores)) if result_scores else 0.0
        top_score = float(np.max(result_scores)) if result_scores else 0.0

        return {
            'chunks': result_chunks,
            'scores': result_scores,
            'candidate_count': len(chunks),
            'retrieved_count': len(result_chunks),
            'avg_score': avg_score,
            'top_score': top_score,
            'paper_ids_searched': paper_ids
        }
