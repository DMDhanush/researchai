"""
ResearchAI — Research Gap Identification Prompts and Utilities
"""

GAP_SYSTEM_PROMPT = """You are an expert academic researcher skilled at identifying research gaps.

Analyze the provided research papers and identify both explicitly stated and inferred research gaps.

For EXPLICIT GAPS: Extract statements where the authors directly mention limitations, future work, or open questions.

For INFERRED GAPS: Use your research knowledge to identify logical gaps not explicitly mentioned (e.g., unexplored populations, missing methodologies, temporal gaps).

For each gap, provide:
1. Title: Concise gap title
2. Category: Explicit or Inferred Potential
3. Description: Clear explanation of the gap
4. Why It Matters: Significance and research potential
5. Supporting Papers: Which papers mention this gap
6. Research Direction: How this gap could be addressed

Be rigorous. Do not invent gaps without grounding.
Distinguish clearly between explicit (stated by authors) and inferred (potential) gaps."""


def format_papers_for_gap_analysis(papers_data: list, chunks_by_paper: dict) -> str:
    """
    Format papers for research gap analysis.

    Args:
        papers_data: List of paper metadata
        chunks_by_paper: Dict mapping paper_id to chunks

    Returns:
        Formatted text for gap analysis
    """
    parts = []

    for paper in papers_data:
        parts.append(f"\n{'='*60}")
        parts.append(f"PAPER: {paper.get('title', 'Unknown')}")
        parts.append(f"{'='*60}")

        if paper.get("authors"):
            parts.append(f"Authors: {paper['authors']}\n")

        # Prioritize sections that mention gaps and limitations
        priority_sections = ["Introduction", "Related Work", "Limitations", "Future Work", "Conclusion"]
        chunks = chunks_by_paper.get(paper.get("id"), [])

        # Sort chunks by section priority
        def section_priority(chunk):
            section = chunk.get("section", "General")
            try:
                return priority_sections.index(section)
            except ValueError:
                return len(priority_sections)

        sorted_chunks = sorted(chunks, key=section_priority)

        for chunk in sorted_chunks[:15]:  # Limit context
            section = chunk.get("section", "General")
            content = chunk.get("content", "").strip()
            if content:
                parts.append(f"[{section}]\n{content}\n")

    return "\n".join(parts)


def parse_gaps_output(gaps_text: str, supporting_papers: list) -> list:
    """
    Parse LLM-generated gaps into structured records.

    Args:
        gaps_text: Raw gap analysis from LLM
        supporting_papers: List of paper IDs that support this analysis

    Returns:
        List of gap dicts
    """
    gaps = []
    current_gap = None

    for line in gaps_text.split("\n"):
        stripped = line.strip()

        # Detect new gap (look for numbered items or gap titles)
        if stripped and (stripped[0].isdigit() or "Gap:" in line or "Research Gap:" in line):
            if current_gap:
                gaps.append(current_gap)
            current_gap = {
                "title": stripped.replace("Gap:", "").replace("Research Gap:", "").strip(),
                "category": "Inferred Potential",
                "description": "",
                "why_it_matters": "",
                "supporting_papers": supporting_papers,
                "research_direction": "",
                "confidence": "MEDIUM"
            }
        elif current_gap and "category:" in line.lower():
            current_gap["category"] = line.split(":", 1)[1].strip()
        elif current_gap and "description:" in line.lower():
            current_gap["description"] = line.split(":", 1)[1].strip()
        elif current_gap and ("why" in line.lower() or "matter" in line.lower()):
            current_gap["why_it_matters"] = line.split(":", 1)[1].strip() if ":" in line else stripped
        elif current_gap and ("research direction" in line.lower() or "address" in line.lower()):
            current_gap["research_direction"] = line.split(":", 1)[1].strip() if ":" in line else stripped
        elif current_gap and stripped:
            # Append to description if no specific field matched
            current_gap["description"] += " " + stripped

    if current_gap:
        gaps.append(current_gap)

    return gaps


def classify_gap_type(gap_text: str) -> str:
    """Classify gap as explicit or inferred."""
    explicit_indicators = [
        "future work", "future research", "suggest", "recommend",
        "further investigation", "limitation", "not addressed",
        "open question", "remains unclear"
    ]

    gap_lower = gap_text.lower()
    if any(indicator in gap_lower for indicator in explicit_indicators):
        return "Explicit"
    return "Inferred Potential"


GAP_CATEGORIES = [
    "Methodological Gap",
    "Temporal Gap",
    "Geographic / Population Gap",
    "Theoretical Gap",
    "Empirical Gap",
    "Technological Gap",
    "Multi-disciplinary Gap",
    "Practical / Applied Gap"
]
