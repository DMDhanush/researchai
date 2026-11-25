"""
ResearchAI — Intent Detection
Understands user query intent for intelligent routing and context understanding.
"""

import re
from typing import Optional


class IntentDetector:
    """Detects user intent from natural language queries."""

    INTENT_MAPPINGS = {
        "problem": [
            "problem", "motivation", "gap", "purpose", "why", "reason",
            "objective", "goal", "question", "research question", "challenge",
            "issue", "concern", "problem statement", "what is the problem"
        ],
        "limitation": [
            "limitation", "weakness", "constraint", "threat", "limitation",
            "shortcoming", "drawback", "challenge", "difficulty", "problem",
            "restricted", "restricted to", "limited", "limits", "scope",
            "threat to validity", "validity threats"
        ],
        "dataset": [
            "dataset", "data", "sample", "participants", "corpus", "database",
            "data source", "source", "collection", "benchmark", "included studies",
            "primary studies", "survey", "experiment", "study population"
        ],
        "findings": [
            "findings", "results", "found", "discovered", "conclude", "conclusion",
            "outcome", "finding", "result", "discover", "showed", "demonstrate",
            "what did they find", "what were the results"
        ],
        "methodology": [
            "methodology", "method", "approach", "technique", "how", "design",
            "research design", "study design", "framework", "process", "procedure",
            "implementation", "model", "algorithm", "strategy"
        ],
        "comparison": [
            "compare", "difference", "differ", "similar", "similar to",
            "versus", "vs", "contrast", "distinguish", "comparison",
            "how do these differ"
        ],
        "evaluation": [
            "evaluate", "evaluate", "performance", "effectiveness", "success",
            "metric", "measure", "accuracy", "precision", "recall"
        ],
        "future": [
            "future", "next", "further", "beyond", "next step", "forward",
            "improvement", "enhancement", "extension", "further work"
        ]
    }

    @staticmethod
    def detect_intent(query: str) -> str:
        """
        Detect the primary intent of a user query.

        Supports:
        - Natural language paraphrases
        - Synonyms and related terms
        - Incomplete grammar and spelling mistakes
        - Case-insensitive matching

        Args:
            query: User's natural language question

        Returns:
            Intent label (e.g., 'problem', 'limitation', 'dataset') or 'general'
        """
        query_lower = query.lower()

        # Score each intent based on keyword matches
        intent_scores = {}

        for intent, keywords in IntentDetector.INTENT_MAPPINGS.items():
            score = 0
            for keyword in keywords:
                # Exact phrase match (highest priority)
                if f" {keyword} " in f" {query_lower} " or query_lower.startswith(keyword) or query_lower.endswith(keyword):
                    score += 3
                # Word boundary match (word as separate token)
                elif re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                    score += 2
                # Substring match (fuzzy)
                elif keyword in query_lower:
                    score += 1

            if score > 0:
                intent_scores[intent] = score

        # Return the highest-scoring intent, or 'general' if no clear match
        if intent_scores:
            top_intent = max(intent_scores, key=intent_scores.get)
            return top_intent

        return "general"

    @staticmethod
    def detect_multiple_intents(query: str, top_k: int = 3) -> list:
        """
        Detect multiple intents ranked by confidence.

        Args:
            query: User query
            top_k: Number of top intents to return

        Returns:
            List of (intent, confidence_score) tuples
        """
        query_lower = query.lower()
        intent_scores = {}

        for intent, keywords in IntentDetector.INTENT_MAPPINGS.items():
            score = 0
            for keyword in keywords:
                if f" {keyword} " in f" {query_lower} " or query_lower.startswith(keyword) or query_lower.endswith(keyword):
                    score += 3
                elif re.search(rf"\b{re.escape(keyword)}\b", query_lower):
                    score += 2
                elif keyword in query_lower:
                    score += 1
            if score > 0:
                intent_scores[intent] = score

        # Sort by score descending
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_intents[:top_k] if sorted_intents else [("general", 0)]

    @staticmethod
    def should_expand_query(intent: str) -> bool:
        """Determine if query expansion is beneficial for this intent."""
        return intent in ["problem", "limitation", "dataset", "findings"]

    @staticmethod
    def get_query_expansion_terms(intent: str) -> list:
        """Get synonyms to expand query for better retrieval."""
        expansion_map = {
            "problem": ["motivation", "research gap", "challenge", "objective"],
            "limitation": ["weakness", "constraint", "threat to validity", "scope"],
            "dataset": ["data source", "sample", "participants", "corpus"],
            "findings": ["results", "outcomes", "conclusions", "discoveries"],
        }
        return expansion_map.get(intent, [])
