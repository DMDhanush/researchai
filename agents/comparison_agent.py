"""
ResearchAI — Paper Comparison Agent
Compares multiple papers across key dimensions.
"""
import json
from typing import List
from datetime import datetime

from database.database import get_session
from database.models import Paper, ComparisonRecord
from ai.llm import LLMClient
from rag.retriever import RAGRetriever
from ai.prompts.comparison_prompt import COMPARISON_SYSTEM_PROMPT, format_comparison_result


class ComparisonAgent:
    """Compares multiple papers."""

    def __init__(self):
        self.llm = LLMClient()
        self.retriever = RAGRetriever()

    def compare(self, paper_ids: List[int], user_id: int) -> dict:
        """
        Compare 2-10 papers across key dimensions.

        Args:
            paper_ids: List of paper IDs to compare (2-10)
            user_id: Current user ID (for isolation)

        Returns:
            {
                'comparison_matrix': {...},
                'narrative_synthesis': str,
                'similarities': [...],
                'differences': [...],
                'methodological_differences': [...],
                'findings_differences': [...],
                'limitations_differences': [...]
            }
        """
        try:
            if not paper_ids or len(paper_ids) < 2 or len(paper_ids) > 10:
                return {"error": "Comparison requires 2-10 papers"}

            # Fetch paper metadata
            with get_session() as db:
                papers = db.query(Paper).filter(
                    Paper.id.in_(paper_ids),
                    Paper.user_id == user_id
                ).all()

                if len(papers) != len(paper_ids):
                    return {"error": "Some papers not found or access denied"}

                paper_dict = {p.id: p for p in papers}

            # Build comparison context for each paper
            paper_contexts = {}
            for pid in paper_ids:
                paper = paper_dict[pid]
                chunks_result = self.retriever.retrieve(
                    query="research problem methodology results limitations findings",
                    user_id=user_id,
                    paper_ids=[pid],
                    top_k=20
                )
                chunks = chunks_result.get("chunks", [])
                context = "\n\n".join([c.get("content", "") for c in chunks[:15]])
                paper_contexts[pid] = {
                    "title": paper.title or "Untitled",
                    "context": context
                }

            # Generate comparison
            comparison_data = self._generate_comparison(paper_contexts, paper_ids)

            # Store in database
            self._store_comparison(user_id, paper_ids, comparison_data)

            return comparison_data

        except Exception as e:
            return {"error": f"Comparison failed: {str(e)}"}

    def _generate_comparison(self, paper_contexts: dict, paper_ids: List[int]) -> dict:
        """Generate comprehensive comparison."""
        try:
            # Build context prompt
            context_prompt = "\n\n---\n\n".join([
                f"Paper: {paper_contexts[pid]['title']}\n\nContent:\n{paper_contexts[pid]['context'][:2000]}"
                for pid in paper_ids
            ])

            # Generate comparison prompt
            prompt = f"""Compare these {len(paper_ids)} papers across these dimensions:
1. Research Problem
2. Objectives
3. Research Type
4. Methodology
5. Dataset
6. Models / Techniques
7. Key Findings
8. Limitations
9. Future Work

Context:
{context_prompt}

Provide:
1. Comparison matrix (structured table)
2. Narrative synthesis (how papers relate)
3. Similarities (common themes)
4. Differences (diverging approaches)
5. Methodological differences
6. Findings differences
7. Limitations differences

Format as JSON with these keys."""

            response = self.llm.generate(
                prompt=prompt,
                context="",
                system_prompt=COMPARISON_SYSTEM_PROMPT
            )

            # Parse or structure response
            try:
                result = json.loads(response)
            except:
                result = {
                    "comparison_narrative": response,
                    "comparison_matrix": {},
                    "similarities": [],
                    "differences": [],
                    "methodological_differences": [],
                    "findings_differences": [],
                    "limitations_differences": []
                }

            return result

        except Exception as e:
            return {"error": f"Comparison generation failed: {str(e)}"}

    def _store_comparison(self, user_id: int, paper_ids: List[int], comparison: dict):
        """Store comparison in database."""
        try:
            with get_session() as db:
                record = ComparisonRecord(
                    user_id=user_id,
                    paper_ids=json.dumps(paper_ids),
                    comparison=json.dumps(comparison, default=str)
                )
                db.add(record)
                db.commit()
        except Exception:
            pass
