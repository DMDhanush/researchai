"""
ResearchAI — Literature Review Agent
Generates structured literature reviews from papers.
"""
import json
from typing import List
from datetime import datetime

from database.database import get_session
from database.models import Paper, LiteratureReviewRecord
from ai.llm import LLMClient
from rag.retriever import RAGRetriever
from ai.prompts.literature_prompt import LITERATURE_SYSTEM_PROMPT, format_literature_result


class LiteratureAgent:
    """Generates structured literature reviews."""

    def __init__(self):
        self.llm = LLMClient()
        self.retriever = RAGRetriever()

    def generate_review(self, paper_ids: List[int], user_id: int,
                       review_type: str = "medium") -> dict:
        """
        Generate structured literature review.

        Args:
            paper_ids: List of paper IDs to review
            user_id: Current user ID (for isolation)
            review_type: 'short' | 'medium' | 'detailed'

        Returns:
            {
                'introduction': str,
                'background': str,
                'research_landscape': str,
                'major_themes': str,
                'methodologies': str,
                'findings': str,
                'benefits': str,
                'challenges': str,
                'research_gaps': str,
                'critical_synthesis': str,
                'future_directions': str,
                'conclusion': str,
                'references': [
                    {'title': str, 'authors': str, 'year': int, 'apa_citation': str}
                ],
                'review_type': str
            }
        """
        try:
            if not paper_ids:
                return {"error": "No papers provided for literature review"}

            # Validate review type
            if review_type not in ["short", "medium", "detailed"]:
                review_type = "medium"

            # Fetch papers
            with get_session() as db:
                papers = db.query(Paper).filter(
                    Paper.id.in_(paper_ids),
                    Paper.user_id == user_id
                ).all()

                if not papers:
                    return {"error": "Papers not found or access denied"}

                paper_dict = {p.id: p for p in papers}

            # Retrieve comprehensive content from all papers
            paper_contents = {}
            for pid in paper_ids:
                paper = paper_dict[pid]
                chunks_result = self.retriever.retrieve(
                    query="research problem methodology findings results implications applications",
                    user_id=user_id,
                    paper_ids=[pid],
                    top_k=25
                )
                chunks = chunks_result.get("chunks", [])
                context = "\n\n".join([c.get("content", "") for c in chunks[:20]])
                paper_contents[pid] = {
                    "title": paper.title or "Untitled",
                    "authors": paper.authors or "Unknown",
                    "year": paper.pub_year or 0,
                    "context": context,
                    "full_text": paper.full_text or ""
                }

            # Generate literature review
            review = self._generate_review_content(paper_contents, paper_ids, review_type)

            # Store in database
            self._store_review(user_id, paper_ids, review_type, review)

            return review

        except Exception as e:
            return {"error": f"Literature review generation failed: {str(e)}"}

    def _generate_review_content(self, paper_contents: dict, paper_ids: List[int],
                                review_type: str) -> dict:
        """Generate literature review sections."""
        try:
            # Build context
            context_prompt = "\n\n---\n\n".join([
                f"Paper {i+1}: {paper_contents[pid]['title']} ({paper_contents[pid]['year']})\n"
                f"Authors: {paper_contents[pid]['authors']}\n\n"
                f"Content:\n{paper_contents[pid]['context'][:1500]}"
                for i, pid in enumerate(paper_ids)
            ])

            length_guidance = {
                "short": "concise (each section 100-150 words)",
                "medium": "balanced (each section 200-300 words)",
                "detailed": "comprehensive (each section 400-500 words)"
            }[review_type]

            prompt = f"""Generate a {length_guidance} literature review from {len(paper_ids)} academic papers.

Papers:
{context_prompt}

Structure the review with these sections:
1. Introduction - Set context and scope
2. Background - Historical development and foundational concepts
3. Research Landscape - Current state of research in this area
4. Major Themes - Key themes and patterns across papers
5. Methodologies - Research methods and approaches used
6. Findings - Key results and discoveries
7. Benefits - Advantages and positive outcomes identified
8. Challenges - Challenges, limitations, and open problems
9. Research Gaps - Identified gaps and opportunities
10. Critical Synthesis - Synthesize and critique the body of work
11. Future Directions - Emerging trends and future research
12. Conclusion - Summary and implications

For References section, provide APA-style citations where metadata is available.

Return as JSON with keys: introduction, background, research_landscape, major_themes,
methodologies, findings, benefits, challenges, research_gaps, critical_synthesis,
future_directions, conclusion, references (array of {title, authors, year, apa_citation})."""

            response = self.llm.generate(
                prompt=prompt,
                context="",
                system_prompt=LITERATURE_SYSTEM_PROMPT
            )

            try:
                result = json.loads(response)
            except:
                # If JSON parsing fails, structure the response
                result = {
                    "introduction": "",
                    "background": "",
                    "research_landscape": "",
                    "major_themes": "",
                    "methodologies": "",
                    "findings": "",
                    "benefits": "",
                    "challenges": "",
                    "research_gaps": "",
                    "critical_synthesis": response,
                    "future_directions": "",
                    "conclusion": "",
                    "references": []
                }

            result["review_type"] = review_type
            return result

        except Exception as e:
            return {
                "error": f"Review generation failed: {str(e)}",
                "review_type": review_type
            }

    def _store_review(self, user_id: int, paper_ids: List[int],
                     review_type: str, review: dict):
        """Store literature review in database."""
        try:
            with get_session() as db:
                record = LiteratureReviewRecord(
                    user_id=user_id,
                    paper_ids=json.dumps(paper_ids),
                    review_type=review_type,
                    content=json.dumps(review, default=str)
                )
                db.add(record)
                db.commit()
        except Exception:
            pass

    def export_markdown(self, review: dict) -> str:
        """Export review as markdown."""
        md = f"# Literature Review\n\n"
        for section in ["introduction", "background", "research_landscape", "major_themes",
                       "methodologies", "findings", "benefits", "challenges",
                       "research_gaps", "critical_synthesis", "future_directions", "conclusion"]:
            if section in review and review[section]:
                title = section.replace("_", " ").title()
                md += f"## {title}\n\n{review[section]}\n\n"

        if "references" in review and review["references"]:
            md += "## References\n\n"
            for ref in review["references"]:
                md += f"- {ref.get('apa_citation', ref.get('title', 'Unknown'))}\n"

        return md

    def export_text(self, review: dict) -> str:
        """Export review as plain text."""
        text = "LITERATURE REVIEW\n"
        text += "=" * 60 + "\n\n"
        for section in ["introduction", "background", "research_landscape", "major_themes",
                       "methodologies", "findings", "benefits", "challenges",
                       "research_gaps", "critical_synthesis", "future_directions", "conclusion"]:
            if section in review and review[section]:
                title = section.replace("_", " ").upper()
                text += f"\n{title}\n"
                text += "-" * 40 + "\n"
                text += f"{review[section]}\n"

        if "references" in review and review["references"]:
            text += "\n" + "=" * 60
            text += "\nREFERENCES\n"
            text += "-" * 40 + "\n"
            for ref in review["references"]:
                text += f"{ref.get('apa_citation', ref.get('title', 'Unknown'))}\n"

        return text
