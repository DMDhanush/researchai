"""
ResearchAI — Research Insights Agent
Generates research analytics from user's paper collection.
"""
import json
from typing import List, Dict
from collections import Counter
from datetime import datetime

from database.database import get_session
from database.models import Paper, AnalysisResult
from ai.llm import LLMClient


class InsightsAgent:
    """Generates research analytics from papers."""

    def __init__(self):
        self.llm = LLMClient()

    def analyze(self, user_id: int) -> dict:
        """
        Analyze all user's papers and generate research insights.

        Args:
            user_id: Current user ID (for isolation)

        Returns:
            {
                'total_papers': int,
                'total_pages': int,
                'publication_timeline': [{year: int, count: int}],
                'research_methodologies': [{methodology: str, count: int}],
                'research_types': [{type: str, count: int}],
                'major_themes': [{theme: str, frequency: int}],
                'ai_tools_mentioned': [{tool: str, count: int}],
                'benefits_identified': [{benefit: str, count: int}],
                'challenges_identified': [{challenge: str, count: int}],
                'paper_count_by_year': [{year: int, count: int}],
                'keywords_frequency': [{keyword: str, frequency: int}],
                'summary': str
            }
        """
        try:
            # Fetch all user's papers
            with get_session() as db:
                papers = db.query(Paper).filter(
                    Paper.user_id == user_id,
                    Paper.status == "ready"
                ).all()

                if not papers:
                    return {
                        "error": "No papers available for analysis",
                        "total_papers": 0
                    }

            # Extract metadata and analyze
            insights = self._extract_insights(papers)

            # Store in database
            self._store_insights(user_id, insights)

            return insights

        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

    def _extract_insights(self, papers: List) -> dict:
        """Extract insights from papers."""
        try:
            total_papers = len(papers)
            total_pages = sum(p.page_count for p in papers)
            total_words = sum(len((p.full_text or "").split()) for p in papers)

            # Publication timeline
            years = [p.pub_year for p in papers if p.pub_year]
            year_counter = Counter(years)
            publication_timeline = sorted(
                [{"year": year, "count": count} for year, count in year_counter.items()],
                key=lambda x: x["year"]
            )

            # Extract methodologies, types, themes from papers
            methodologies = self._extract_methodologies(papers)
            research_types = self._extract_research_types(papers)
            major_themes = self._extract_themes(papers)
            ai_tools = self._extract_ai_tools(papers)
            benefits = self._extract_benefits(papers)
            challenges = self._extract_challenges(papers)

            # Keywords frequency
            all_keywords = []
            for p in papers:
                if p.keywords:
                    kw_list = [k.strip() for k in p.keywords.split(",")]
                    all_keywords.extend(kw_list)
            keywords_counter = Counter(all_keywords)
            keywords_frequency = [
                {"keyword": kw, "frequency": freq}
                for kw, freq in keywords_counter.most_common(20)
            ]

            # Generate summary
            summary = f"""Research Portfolio Analysis:
- Total papers: {total_papers}
- Total pages: {total_pages}
- Total words: {total_words:,}
- Time span: {min(years) if years else "N/A"} - {max(years) if years else "N/A"}
- Average pages per paper: {total_pages // total_papers if total_papers > 0 else 0}
- Most common methodology: {methodologies[0]['methodology'] if methodologies else 'N/A'}
- Most common research type: {research_types[0]['type'] if research_types else 'N/A'}
- Most discussed theme: {major_themes[0]['theme'] if major_themes else 'N/A'}
"""

            return {
                "total_papers": total_papers,
                "total_pages": total_pages,
                "total_words": total_words,
                "publication_timeline": publication_timeline,
                "research_methodologies": methodologies,
                "research_types": research_types,
                "major_themes": major_themes,
                "ai_tools_mentioned": ai_tools,
                "benefits_identified": benefits,
                "challenges_identified": challenges,
                "paper_count_by_year": publication_timeline,
                "keywords_frequency": keywords_frequency,
                "summary": summary,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": f"Insight extraction failed: {str(e)}"}

    def _extract_methodologies(self, papers: List) -> List[Dict]:
        """Extract research methodologies from papers."""
        methodologies = []
        methodology_keywords = [
            "empirical", "experimental", "qualitative", "quantitative",
            "mixed methods", "survey", "case study", "ethnography",
            "action research", "grounded theory", "phenomenology",
            "statistical", "simulation", "modeling", "meta-analysis"
        ]

        methodology_counter = Counter()
        for paper in papers:
            text = (paper.full_text or "").lower()
            for method in methodology_keywords:
                if method in text:
                    methodology_counter[method] += 1

        return [
            {"methodology": method, "count": count}
            for method, count in methodology_counter.most_common(10)
        ]

    def _extract_research_types(self, papers: List) -> List[Dict]:
        """Extract research types."""
        research_types = []
        type_keywords = {
            "Systematic Review": ["systematic review", "meta-analysis"],
            "Literature Review": ["literature review", "scoping review"],
            "Empirical Study": ["empirical study", "experiment", "field study"],
            "Theoretical": ["theoretical framework", "conceptual model"],
            "Case Study": ["case study"],
            "Survey": ["survey", "questionnaire"]
        }

        type_counter = Counter()
        for paper in papers:
            text = (paper.full_text or "").lower()
            for rtype, keywords in type_keywords.items():
                for kw in keywords:
                    if kw in text:
                        type_counter[rtype] += 1
                        break

        return [
            {"type": rtype, "count": count}
            for rtype, count in type_counter.most_common(10)
        ]

    def _extract_themes(self, papers: List) -> List[Dict]:
        """Extract major themes from papers."""
        theme_keywords = {
            "Machine Learning": ["machine learning", "neural network", "deep learning"],
            "Natural Language Processing": ["nlp", "language model", "text processing"],
            "Computer Vision": ["computer vision", "image recognition", "object detection"],
            "Data Analysis": ["data analysis", "data mining", "analytics"],
            "Optimization": ["optimization", "algorithm", "performance"],
            "Security": ["security", "privacy", "encryption"],
            "Distributed Systems": ["distributed", "parallel", "cloud"],
            "Artificial Intelligence": ["artificial intelligence", "ai", "intelligent system"],
            "Software Engineering": ["software engineering", "development", "testing"],
            "Human-Computer Interaction": ["hci", "user interface", "usability"]
        }

        theme_counter = Counter()
        for paper in papers:
            text = (paper.full_text or "").lower()
            for theme, keywords in theme_keywords.items():
                for kw in keywords:
                    if kw in text:
                        theme_counter[theme] += 1
                        break

        return [
            {"theme": theme, "frequency": count}
            for theme, count in theme_counter.most_common(15)
        ]

    def _extract_ai_tools(self, papers: List) -> List[Dict]:
        """Extract AI tools and frameworks mentioned."""
        ai_tools = [
            "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
            "GPT", "BERT", "Transformer", "ResNet", "Spacy",
            "OpenAI", "Hugging Face", "JAX", "MXNet", "ONNX"
        ]

        tool_counter = Counter()
        for paper in papers:
            text = paper.full_text or ""
            for tool in ai_tools:
                if tool.lower() in text.lower():
                    tool_counter[tool] += 1

        return [
            {"tool": tool, "count": count}
            for tool, count in tool_counter.most_common(15)
        ]

    def _extract_benefits(self, papers: List) -> List[Dict]:
        """Extract identified benefits from papers."""
        benefit_keywords = [
            "improves", "better", "efficient", "accurate", "faster",
            "scalable", "robust", "effective", "reliable", "novel",
            "outperforms", "advantages", "benefits", "gains"
        ]

        benefit_counter = Counter()
        for paper in papers:
            text = (paper.full_text or "").lower()
            for benefit in benefit_keywords:
                if benefit in text:
                    benefit_counter[benefit] += 1

        return [
            {"benefit": benefit, "count": count}
            for benefit, count in benefit_counter.most_common(10)
        ]

    def _extract_challenges(self, papers: List) -> List[Dict]:
        """Extract identified challenges from papers."""
        challenge_keywords = [
            "challenge", "limitation", "difficulty", "problem", "issue",
            "scalability", "complexity", "bottleneck", "overhead",
            "expensive", "compute", "data scarcity", "generalization"
        ]

        challenge_counter = Counter()
        for paper in papers:
            text = (paper.full_text or "").lower()
            for challenge in challenge_keywords:
                if challenge in text:
                    challenge_counter[challenge] += 1

        return [
            {"challenge": challenge, "count": count}
            for challenge, count in challenge_counter.most_common(10)
        ]

    def _store_insights(self, user_id: int, insights: dict):
        """Store insights in database."""
        try:
            with get_session() as db:
                result = AnalysisResult(
                    user_id=user_id,
                    analysis_type="insights",
                    content=json.dumps(insights, default=str)
                )
                db.add(result)
                db.commit()
        except Exception:
            pass
