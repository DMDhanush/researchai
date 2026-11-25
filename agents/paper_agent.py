from ai.llm import LLMClient
from database.database import get_session
from database.models import Paper, AnalysisResult
import json

class PaperAgent:
    def __init__(self):
        self.llm = LLMClient()
    
    def summarize(self, paper_id, user_id):
        with get_session() as db:
            paper = db.query(Paper).filter(Paper.id == paper_id, Paper.user_id == user_id).first()
            if not paper: raise ValueError("Paper not found")
            summary = self.llm.generate(
                prompt=f"Summarize this paper: {paper.title}\n\nAbstract: {paper.abstract}",
                context=paper.full_text[:10000] if paper.full_text else ""
            )
            res = AnalysisResult(user_id=user_id, paper_id=paper_id, analysis_type="summary", content=summary)
            db.add(res)
            db.commit()
            return {"summary": summary}
