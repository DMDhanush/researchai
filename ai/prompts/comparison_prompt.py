"""
ResearchAI — Paper Comparison Prompts and Utilities
"""

COMPARISON_SYSTEM_PROMPT = """You are an expert academic research analyst.

Compare the provided research papers across multiple dimensions.

Structure your comparison to highlight:
1. Research Problems: How does each paper frame the problem differently?
2. Objectives: What are the specific goals of each study?
3. Research Type: What type of research is each (empirical, survey, meta-analysis, etc.)?
4. Methodology: How do the methods differ?
5. Dataset/Sample: What data or participants did each use?
6. Models/Techniques: What approaches or algorithms did each employ?
7. Key Findings: What were the main results?
8. Limitations: What constraints or weaknesses exist in each?
9. Future Work: What do the authors suggest next?

Be precise and fair. Highlight both similarities and differences.
Do not invent information not present in the papers."""


def format_papers_for_comparison(papers_data: list, chunks_by_paper: dict) -> str:
    """
    Format multiple papers for comparison analysis.

    Args:
        papers_data: List of paper dicts with metadata
        chunks_by_paper: Dict mapping paper_id to list of chunks

    Returns:
        Formatted text for comparison
    """
    parts = []

    for i, paper in enumerate(papers_data, 1):
        parts.append(f"\n{'='*60}")
        parts.append(f"PAPER {i}: {paper.get('title', 'Unknown')}")
        parts.append(f"{'='*60}")

        if paper.get("authors"):
            parts.append(f"Authors: {paper['authors']}")

        if paper.get("pub_year"):
            parts.append(f"Year: {paper['pub_year']}")

        if paper.get("abstract"):
            parts.append(f"Abstract: {paper['abstract']}")

        # Add chunks
        chunks = chunks_by_paper.get(paper.get("id"), [])
        if chunks:
            parts.append("\nContent:")
            for chunk in chunks[:10]:  # Limit to top 10 chunks for context
                section = chunk.get("section", "General")
                content = chunk.get("content", "").strip()
                if content:
                    parts.append(f"[{section}] {content[:500]}")  # Limit chunk size

    return "\n".join(parts)


def parse_comparison_output(comparison_text: str) -> dict:
    """
    Parse LLM-generated comparison into structured sections.

    Args:
        comparison_text: Raw comparison from LLM

    Returns:
        Dict with parsed comparison sections
    """
    sections = {
        "research_problems": "",
        "objectives": "",
        "research_types": "",
        "methodologies": "",
        "datasets": "",
        "models_techniques": "",
        "key_findings": "",
        "limitations": "",
        "future_work": "",
        "summary": ""
    }

    current_section = "summary"
    for line in comparison_text.split("\n"):
        lower_line = line.lower().strip()

        if any(x in lower_line for x in ["research problem", "problem statement"]):
            current_section = "research_problems"
        elif any(x in lower_line for x in ["objective", "goal", "aim"]):
            current_section = "objectives"
        elif any(x in lower_line for x in ["research type", "study design"]):
            current_section = "research_types"
        elif any(x in lower_line for x in ["methodology", "method", "approach"]):
            current_section = "methodologies"
        elif any(x in lower_line for x in ["dataset", "data", "sample"]):
            current_section = "datasets"
        elif any(x in lower_line for x in ["model", "technique", "algorithm"]):
            current_section = "models_techniques"
        elif any(x in lower_line for x in ["finding", "result", "outcome"]):
            current_section = "key_findings"
        elif any(x in lower_line for x in ["limitation", "weakness"]):
            current_section = "limitations"
        elif any(x in lower_line for x in ["future", "next", "further"]):
            current_section = "future_work"
        else:
            if line.strip():
                sections[current_section] += line + "\n"

    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def generate_comparison_matrix(papers_data: list, comparison_analysis: dict) -> dict:
    """
    Generate a comparison matrix for tabular display.

    Args:
        papers_data: List of paper metadata
        comparison_analysis: Parsed comparison output

    Returns:
        Matrix dict for display in UI
    """
    dimensions = [
        "Research Problem",
        "Objectives",
        "Research Type",
        "Methodology",
        "Dataset",
        "Models/Techniques",
        "Key Findings",
        "Limitations",
        "Future Work"
    ]

    matrix = {
        "dimensions": dimensions,
        "papers": [p.get("title", "Unknown") for p in papers_data],
        "data": {}
    }

    # This would be populated by parsing the comparison analysis
    # For now, provide structure
    for dim in dimensions:
        matrix["data"][dim] = ["" for _ in papers_data]

    return matrix


COMPARISON_DIMENSIONS = [
    "Research Problem",
    "Objectives",
    "Research Type",
    "Methodology",
    "Dataset",
    "Models / Techniques",
    "Key Findings",
    "Limitations",
    "Future Work"
]
