"""
ResearchAI — Research Gap Finder Agent
Identifies explicit and inferred research gaps from papers.
"""
import json
from typing import List
from datetime import datetime

from database.database import get_session
from database.models import Paper, ResearchGapRecord
from ai.llm import LLMClient
from rag.retriever import RAGRetriever
from ai.prompts.gap_prompt import GAP_SYSTEM_PROMPT, format_gap_result


class GapAgent:
    """Identifies research gaps from papers."""

    def __init__(self):
        self.llm = LLMClient()
        self.retriever = RAGRetriever()

    def find_gaps(self, paper_ids: List[int], user_id: int) -> dict:
        """
        Identify both explicit and inferred research gaps.

        Args:
            paper_ids: List of paper IDs to analyze
            user_id: Current user ID (for isolation)

        Returns:
            {
                'gaps': [
                    {
                        'gap_id': str,
                        'title': str,
                        'category': str,
                        'gap_type': 'explicit' | 'inferred',
                        'description': str,
                        'evidence': str,
                        'why_it_matters': str,
                        'supporting_papers': [paper_ids],
                        'confidence': 'HIGH' | 'MEDIUM' | 'LOW',
                        'research_direction': str
                    }
                ]
            }
        """
        try:
            if not paper_ids:
                return {"error": "No papers provided"}

            # Fetch papers
            with get_session() as db:
                papers = db.query(Paper).filter(
                    Paper.id.in_(paper_ids),
                    Paper.user_id == user_id
                ).all()

                if not papers:
                    return {"error": "Papers not found or access denied"}

                paper_dict = {p.id: p for p in papers}

            # Retrieve gap-related content from all papers
            gap_contexts = {}
            for pid in paper_ids:
                paper = paper_dict[pid]
                # Search for gap-related sections
                chunks_result = self.retriever.retrieve(
                    query="research gap limitation future work open question challenge",
                    user_id=user_id,
                    paper_ids=[pid],
                    top_k=15
                )
                chunks = chunks_result.get("chunks", [])
                context = "\n\n".join([c.get("content", "") for c in chunks[:10]])
                gap_contexts[pid] = {
                    "title": paper.title or "Untitled",
                    "context": context
                }

            # Generate gaps
            gaps = self._generate_gaps(gap_contexts, paper_ids)

            # Store in database
            self._store_gaps(user_id, paper_ids, gaps)

            return gaps

        except Exception as e:
            return {"error": f"Gap identification failed: {str(e)}"}

    def _generate_gaps(self, gap_contexts: dict, paper_ids: List[int]) -> dict:
        """Generate identified gaps."""
        try:
            # Build context
            context_prompt = "\n\n---\n\n".join([
                f"Paper: {gap_contexts[pid]['title']}\n\nRelevant sections:\n{gap_contexts[pid]['context'][:1500]}"
                for pid in paper_ids
            ])

            prompt = f"""Analyze these papers and identify both EXPLICIT and INFERRED research gaps.

Papers analyzed: {len(paper_ids)}

Context:
{context_prompt}

For EXPLICIT gaps: These are gaps the authors explicitly mention as limitations, future work, or open questions.
Label these as gap_type: "explicit"

For INFERRED gaps: These are potential gaps you can reasonably infer from analyzing the work.
Label these as gap_type: "inferred" and prefix title with "[Inferred Potential]"

CRITICAL: Never present an inference as explicit author statement.

For each gap, provide:
1. gap_id: Unique identifier
2. title: Clear gap title (prefix inferred gaps with "[Inferred Potential]")
3. category: research_direction | methodology | dataset | evaluation | application
4. gap_type: "explicit" or "inferred"
5. description: What is the gap and why does it exist?
6. evidence: Supporting quotes or evidence from the paper(s)
7. why_it_matters: Impact and significance
8. supporting_papers: Which paper IDs this gap comes from
9. confidence: HIGH | MEDIUM | LOW
10. research_direction: Suggested research direction to address gap

Return as JSON with key 'gaps' containing array of gap objects."""

            response = self.llm.generate(
                prompt=prompt,
                context="",
                system_prompt=GAP_SYSTEM_PROMPT
            )

            try:
                result = json.loads(response)
            except:
                result = {
                    "gaps": [],
                    "summary": response
                }

            return result

        except Exception as e:
            return {"gaps": [], "error": f"Gap generation failed: {str(e)}"}

    def _store_gaps(self, user_id: int, paper_ids: List[int], gaps_result: dict):
        """Store gaps in database."""
        try:
            with get_session() as db:
                record = ResearchGapRecord(
                    user_id=user_id,
                    paper_ids=json.dumps(paper_ids),
                    gaps=json.dumps(gaps_result, default=str)
                )
                db.add(record)
                db.commit()
        except Exception:
            pass
