"""
ResearchAI — Paper Summarization Prompts and Utilities
"""

SUMMARY_SYSTEM_PROMPT = """You are an expert academic research summarizer.

Analyze the provided research paper content and generate a comprehensive, structured summary.

Extract and clearly distinguish:
- The research problem or motivation
- Research objectives
- Methodology and approach
- Key findings and results
- Limitations and future work

Be precise. Do not invent information not present in the paper.
If a section is not clearly addressed in the paper, state: "Not explicitly reported in the paper."

Format the summary for clarity and readability."""


def format_paper_for_summary(paper_data: dict, chunks: list) -> str:
    """
    Format paper content for summarization.

    Args:
        paper_data: Dict with paper metadata (title, authors, abstract, etc.)
        chunks: List of paper chunks ordered by page/section

    Returns:
        Formatted text for summarization
    """
    parts = []

    if paper_data.get("title"):
        parts.append(f"TITLE: {paper_data['title']}")

    if paper_data.get("authors"):
        parts.append(f"AUTHORS: {paper_data['authors']}")

    if paper_data.get("pub_year"):
        parts.append(f"PUBLICATION YEAR: {paper_data['pub_year']}")

    if paper_data.get("abstract"):
        parts.append(f"ABSTRACT:\n{paper_data['abstract']}")

    parts.append("FULL CONTENT:")
    for chunk in chunks:
        section = chunk.get("section", "General")
        content = chunk.get("content", "").strip()
        if content:
            parts.append(f"[{section}]\n{content}")

    return "\n\n".join(parts)


def parse_summary_output(summary_text: str) -> dict:
    """
    Parse LLM-generated summary into structured sections.

    Args:
        summary_text: Raw summary from LLM

    Returns:
        Dict with parsed sections
    """
    sections = {
        "executive_summary": "",
        "research_problem": "",
        "objectives": "",
        "research_type": "",
        "methodology": "",
        "dataset": "",
        "algorithms": "",
        "experimental_setup": "",
        "results": "",
        "limitations": "",
        "future_work": "",
        "key_contributions": "",
        "keywords": ""
    }

    # Simple section extraction (can be enhanced with more sophisticated parsing)
    current_section = "executive_summary"
    for line in summary_text.split("\n"):
        lower_line = line.lower().strip()

        # Detect section headers
        if any(x in lower_line for x in ["research problem", "motivation", "gap"]):
            current_section = "research_problem"
        elif any(x in lower_line for x in ["objective", "goal", "aim"]):
            current_section = "objectives"
        elif any(x in lower_line for x in ["research type", "study design", "type of study"]):
            current_section = "research_type"
        elif any(x in lower_line for x in ["methodology", "method", "approach", "technique"]):
            current_section = "methodology"
        elif any(x in lower_line for x in ["dataset", "data", "sample", "participants", "corpus"]):
            current_section = "dataset"
        elif any(x in lower_line for x in ["algorithm", "model", "technique"]):
            current_section = "algorithms"
        elif any(x in lower_line for x in ["experimental setup", "experiment", "setup"]):
            current_section = "experimental_setup"
        elif any(x in lower_line for x in ["result", "finding", "outcome"]):
            current_section = "results"
        elif any(x in lower_line for x in ["limitation", "limitation", "weakness", "constraint"]):
            current_section = "limitations"
        elif any(x in lower_line for x in ["future", "next", "further work"]):
            current_section = "future_work"
        elif any(x in lower_line for x in ["contribution", "novelty", "impact"]):
            current_section = "key_contributions"
        elif any(x in lower_line for x in ["keyword", "key word"]):
            current_section = "keywords"
        else:
            # Append to current section
            if line.strip():
                sections[current_section] += line + "\n"

    # Clean up sections
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


SUMMARY_SECTIONS = [
    "Executive Summary",
    "Research Problem",
    "Objectives",
    "Research Type",
    "Methodology",
    "Dataset / Studies / Participants",
    "Algorithms / Models / Techniques",
    "Experimental Setup",
    "Results",
    "Limitations",
    "Future Work",
    "Key Contributions",
    "Keywords"
]
