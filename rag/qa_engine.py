"""
ResearchAI — QA Engine Module
Full Ask ResearchAI pipeline: intent detection → retrieval → validation → LLM generation → citations.
"""
import time
import json
from typing import Optional, List, Dict, Any
from datetime import datetime

from database.database import get_session
from database.models import ResearchQuestion, RetrievalLog
from .retriever import RAGRetriever
from ai.llm import LLMClient
from ai.intent import IntentDetector
from ai.prompts.qa_prompt import ASK_RESEARCHAI_SYSTEM_PROMPT, format_context_for_qa


class QAEngine:
    """Ask ResearchAI engine with full RAG pipeline."""

    def __init__(self):
        self.retriever = RAGRetriever()
        self.llm_client = LLMClient()
        self.intent_detector = IntentDetector()

    def ask(
        self,
        question: str,
        user_id: int,
        paper_ids: Optional[List[int]] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full Ask ResearchAI pipeline.

        1. Detect intent from question
        2. Construct query (may expand based on intent)
        3. Retrieve candidates
        4. Validate candidates
        5. Rerank if needed
        6. Pass to LLM
        7. Generate grounded answer with citations
        8. Store in database

        Args:
            question: User's natural language question
            user_id: Current user ID
            paper_ids: List of paper IDs to search (None = all user's papers)
            conversation_id: Optional ID for conversation threading

        Returns:
            {
                'answer': str,
                'sources': [list of source dicts],
                'confidence': 'HIGH' | 'MEDIUM' | 'LOW' | 'NO_EVIDENCE',
                'chunks': [retrieved chunks],
                'scores': [similarity scores],
                'intent': detected_intent,
                'latency_ms': float
            }
        """
        pipeline_start = time.time()

        # Step 1: Detect intent
        intent = self.intent_detector.detect_intent(question)

        # Step 2: Construct/expand query based on intent
        query = self._construct_query(question, intent)

        # Step 3-5: Retrieve with validation and reranking
        retrieval_result = self.retriever.retrieve(
            query=query,
            user_id=user_id,
            paper_ids=paper_ids,
            top_k=5,
            include_reranking=True
        )

        chunks = retrieval_result['chunks']
        scores = retrieval_result['scores']
        retrieved_count = retrieval_result['retrieved_count']

        # Step 6: Determine confidence
        confidence = self._determine_confidence(scores, retrieved_count)

        # Generate sources from chunks
        sources = self._generate_citations(chunks)

        # Step 7: Generate answer via LLM if we have evidence
        if chunks:
            context = format_context_for_qa(chunks)
            try:
                answer = self.llm_client.generate(
                    prompt=question,
                    context=context,
                    system_prompt=ASK_RESEARCHAI_SYSTEM_PROMPT
                )
            except Exception as e:
                answer = f"Error generating answer: {str(e)}"
                confidence = "NO_EVIDENCE"
        else:
            answer = "I couldn't find sufficient evidence for this question in the selected papers."
            confidence = "NO_EVIDENCE"

        pipeline_elapsed_ms = (time.time() - pipeline_start) * 1000

        # Step 8: Store in database
        self._store_qa_record(
            user_id=user_id,
            question=question,
            answer=answer,
            sources=sources,
            confidence=confidence,
            paper_ids=paper_ids or [],
            latency_ms=pipeline_elapsed_ms,
            intent=intent,
            conversation_id=conversation_id
        )

        return {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'chunks': chunks,
            'scores': scores,
            'intent': intent,
            'latency_ms': pipeline_elapsed_ms,
            'retrieved_count': retrieved_count,
            'candidate_count': retrieval_result['candidate_count']
        }

    def _construct_query(self, question: str, intent: str) -> str:
        """
        Optionally expand/rewrite query based on detected intent.
        For now, return as-is. Can be enhanced with query expansion.
        """
        return question

    def _determine_confidence(self, scores: List[float], chunk_count: int) -> str:
        """
        Determine confidence level based on retrieval quality.

        HIGH: multiple strong chunks (avg score > 0.7)
        MEDIUM: some relevant chunks (avg score 0.5-0.7 or fewer chunks)
        LOW: weak evidence (avg score 0.3-0.5)
        NO_EVIDENCE: no chunks or very weak (< 0.3)
        """
        if not scores or chunk_count == 0:
            return "NO_EVIDENCE"

        avg_score = sum(scores) / len(scores)

        if chunk_count >= 3 and avg_score > 0.7:
            return "HIGH"
        elif chunk_count >= 2 and avg_score > 0.5:
            return "MEDIUM"
        elif avg_score > 0.3:
            return "LOW"
        else:
            return "NO_EVIDENCE"

    def _generate_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Generate citations from chunks.
        Format: {paper_title, page_number, section}
        """
        citations = []
        seen = set()

        for chunk in chunks:
            key = (chunk['paper_title'], chunk['page_number'])
            if key not in seen:
                citation = {
                    'paper_title': chunk['paper_title'],
                    'page_number': chunk['page_number'],
                    'section': chunk['section']
                }
                citations.append(citation)
                seen.add(key)

        return citations

    def _store_qa_record(
        self,
        user_id: int,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        confidence: str,
        paper_ids: List[int],
        latency_ms: float,
        intent: str,
        conversation_id: Optional[str] = None
    ) -> None:
        """
        Store the Q&A record in database for history and evaluation.
        """
        with get_session() as db:
            qa_record = ResearchQuestion(
                user_id=user_id,
                question=question,
                answer=answer,
                sources=json.dumps(sources),
                confidence=confidence,
                paper_scope=json.dumps(paper_ids),
                intent=intent,
                conversation_id=conversation_id,
                latency_ms=int(latency_ms)
            )
            db.add(qa_record)
            db.commit()
