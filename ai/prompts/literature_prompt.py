"""
ResearchAI — Literature Review Prompts and Utilities
"""

LITERATURE_REVIEW_SYSTEM_PROMPT = """You are an expert academic writer skilled at synthesizing research literature.

Generate a comprehensive literature review from the provided research papers.

Structure the review to include:
1. Introduction: Overview of the research domain
2. Background: Historical context and foundational concepts
3. Research Landscape: Major research areas and trends
4. Major Themes: Key themes across papers
5. Methodologies: Methods employed across studies
6. Findings & Contributions: Synthesis of results
7. Benefits & Strengths: What these papers contribute
8. Challenges & Limitations: Common challenges and weaknesses
9. Research Gaps: Identified gaps in the literature
10. Critical Synthesis: Your analysis of trends and contradictions
11. Future Directions: Promising research directions
12. Conclusion: Summary and implications
13. References: APA-formatted citations

Be scholarly and balanced. Synthesize, don't merely summarize.
Cite papers properly throughout.
Adapt length and depth based on requested review type (short/medium/detailed)."""


def format_papers_for_literature_review(papers_data: list, chunks_by_paper: dict, review_type: str = "medium") -> str:
    """
    Format papers for literature review generation.

    Args:
        papers_data: List of paper metadata
        chunks_by_paper: Dict mapping paper_id to chunks
        review_type: "short", "medium", or "detailed"

    Returns:
        Formatted text for literature review
    """
    parts = []

    # Adjust content depth based on review type
    chunk_limit = {"short": 5, "medium": 10, "detailed": 20}.get(review_type, 10)

    for paper in papers_data:
        parts.append(f"\nPAPER: {paper.get('title', 'Unknown')}")
        parts.append(f"Authors: {paper.get('authors', 'Unknown')}")
        parts.append(f"Year: {paper.get('pub_year', 'Unknown')}")

        if paper.get("abstract"):
            parts.append(f"Abstract: {paper['abstract']}\n")

        chunks = chunks_by_paper.get(paper.get("id"), [])
        for chunk in chunks[:chunk_limit]:
            section = chunk.get("section", "General")
            content = chunk.get("content", "").strip()
            if content:
                parts.append(f"[{section}] {content[:500]}")

    return "\n".join(parts)


def parse_literature_review_output(review_text: str) -> dict:
    """
    Parse LLM-generated literature review into sections.

    Args:
        review_text: Raw review from LLM

    Returns:
        Dict with parsed sections
    """
    sections = {
        "introduction": "",
        "background": "",
        "research_landscape": "",
        "major_themes": "",
        "methodologies": "",
        "findings_contributions": "",
        "benefits_strengths": "",
        "challenges_limitations": "",
        "research_gaps": "",
        "critical_synthesis": "",
        "future_directions": "",
        "conclusion": "",
        "references": ""
    }

    current_section = "introduction"
    section_keywords = {
        "introduction": ["introduction"],
        "background": ["background", "historical context", "foundational"],
        "research_landscape": ["research landscape", "research areas", "trends"],
        "major_themes": ["major theme", "theme", "topic", "area"],
        "methodologies": ["methodology", "method", "approach"],
        "findings_contributions": ["finding", "contribution", "result", "outcome"],
        "benefits_strengths": ["benefit", "strength", "advantage"],
        "challenges_limitations": ["challenge", "limitation", "weakness", "constraint"],
        "research_gaps": ["research gap", "gap", "open question"],
        "critical_synthesis": ["synthesis", "analysis", "trend", "pattern"],
        "future_directions": ["future", "direction", "next", "further research"],
        "conclusion": ["conclusion", "summary"],
        "references": ["reference", "citation"]
    }

    for line in review_text.split("\n"):
        lower_line = line.lower().strip()

        # Check for section headers
        for section, keywords in section_keywords.items():
            if any(kw in lower_line for kw in keywords) and len(line) < 100:
                current_section = section
                break

        # Add line to current section
        if line.strip():
            sections[current_section] += line + "\n"

    # Clean up
    for key in sections:
        sections[key] = sections[key].strip()

    return sections


def generate_apa_citation(paper: dict) -> str:
    """
    Generate APA-formatted citation from paper metadata.

    Args:
        paper: Paper dict with authors, title, pub_year

    Returns:
        APA-formatted citation string
    """
    authors = paper.get("authors", "Unknown").split(";")[0]  # First author
    year = paper.get("pub_year", "n.d.")
    title = paper.get("title", "Unknown")

    # Simplified APA format
    return f"{authors} ({year}). {title}."


def format_for_export(review_data: dict, export_format: str = "markdown") -> str:
    """
    Format literature review for export.

    Args:
        review_data: Parsed review dict with sections
        export_format: "markdown" or "text"

    Returns:
        Formatted export string
    """
    if export_format == "markdown":
        parts = []
        for section, content in review_data.items():
            if content.strip():
                title = section.replace("_", " ").title()
                parts.append(f"## {title}\n\n{content}\n")
        return "\n".join(parts)
    else:  # plain text
        parts = []
        for section, content in review_data.items():
            if content.strip():
                title = section.replace("_", " ").upper()
                parts.append(f"\n{title}\n{'='*len(title)}\n{content}\n")
        return "\n".join(parts)


REVIEW_TYPES = ["Short", "Medium", "Detailed"]

REVIEW_SECTIONS = [
    "Introduction",
    "Background",
    "Research Landscape",
    "Major Themes",
    "Methodologies",
    "Findings & Contributions",
    "Benefits & Strengths",
    "Challenges & Limitations",
    "Research Gaps",
    "Critical Synthesis",
    "Future Directions",
    "Conclusion",
    "References"
]
