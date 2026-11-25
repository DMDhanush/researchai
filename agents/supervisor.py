"""
ResearchAI — Supervisor Agent
Orchestrates specialist agents for different analysis tasks.
"""
import json
from datetime import datetime
from typing import Optional

from database.database import get_session
from database.models import AgentLog
from ai.llm import LLMClient
from .paper_agent import PaperAgent
from .comparison_agent import ComparisonAgent
from .gap_agent import GapAgent
from .literature_agent import LiteratureAgent
from .insights_agent import InsightsAgent


class SupervisorAgent:
    """Routes tasks to appropriate specialist agents."""

    def __init__(self):
        self.paper_agent = PaperAgent()
        self.comparison_agent = ComparisonAgent()
        self.gap_agent = GapAgent()
        self.literature_agent = LiteratureAgent()
        self.insights_agent = InsightsAgent()

    def route_task(self, task_type: str, user_id: int, **kwargs) -> dict:
        """
        Route task to specialist agent based on type.

        Args:
            task_type: Type of task (summarize, compare, find_gaps, literature_review, insights)
            user_id: Current user ID
            **kwargs: Task-specific parameters

        Returns:
            Result dict from the specialist agent
        """
        try:
            start_time = datetime.now()

            if task_type == "summarize":
                result = self.paper_agent.summarize(kwargs["paper_id"], user_id)
                agent_name = "PaperAgent"

            elif task_type == "compare":
                result = self.comparison_agent.compare(kwargs["paper_ids"], user_id)
                agent_name = "ComparisonAgent"

            elif task_type == "find_gaps":
                result = self.gap_agent.find_gaps(kwargs["paper_ids"], user_id)
                agent_name = "GapAgent"

            elif task_type == "literature_review":
                review_type = kwargs.get("review_type", "medium")
                result = self.literature_agent.generate_review(kwargs["paper_ids"], user_id, review_type)
                agent_name = "LiteratureAgent"

            elif task_type == "insights":
                result = self.insights_agent.analyze(user_id)
                agent_name = "InsightsAgent"

            else:
                return {"error": f"Unknown task type: {task_type}"}

            # Log agent execution
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            self._log_agent_action(user_id, agent_name, task_type, kwargs, result, latency_ms)

            return result

        except Exception as e:
            return {"error": f"Task execution failed: {str(e)}"}

    def _log_agent_action(self, user_id: int, agent_name: str, action: str,
                         input_data: dict, output_data: dict, latency_ms: float):
        """Log agent action to database."""
        try:
            with get_session() as db:
                log = AgentLog(
                    user_id=user_id,
                    agent_name=agent_name,
                    action=action,
                    input_data=json.dumps(input_data, default=str),
                    output_data=json.dumps(output_data, default=str),
                    latency_ms=int(latency_ms)
                )
                db.add(log)
                db.commit()
        except Exception as e:
            pass  # Silently fail logging to not interrupt main flow
