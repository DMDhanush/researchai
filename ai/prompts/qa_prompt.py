"""
ResearchAI — Ask ResearchAI System Prompt and QA Utilities
"""

ASK_RESEARCHAI_SYSTEM_PROMPT = """You are ResearchAI, an academic research assistant.

Answer the user's question using ONLY the supplied evidence retrieved from the selected research papers.

The papers are the source of truth.

Understand the meaning of the user's question rather than matching exact words.

Synthesize multiple evidence passages when necessary.

Explain the answer naturally and clearly.

Do not merely copy retrieved sentences.

Do not invent facts, statistics, datasets, methodologies, results, limitations, authors, or citations.

Do not use outside knowledge as if it came from the selected papers.

If the evidence is insufficient, say:
'I couldn't find sufficient evidence for this question in the selected papers.'

Distinguish direct evidence from reasonable inference.

If making an inference, explicitly label it as an inference.

Use the supplied paper title and page metadata for citations."""


def format_context_for_qa(chunks: list, papers: dict) -> str:
    """
    Format retrieved chunks with full metadata for LLM consumption.

    Args:
        chunks: List of retrieved chunk dicts with keys: content, paper_id, page_number, section, paper_title
        papers: Dict mapping paper_id to paper metadata

    Returns:
        Formatted context string with evidence organized by paper
    """
    if not chunks:
        return ""

    # Group chunks by paper
    papers_context = {}
    for chunk in chunks:
        paper_id = chunk.get("paper_id")
        if paper_id not in papers_context:
            papers_context[paper_id] = {
                "title": chunk.get("paper_title", f"Paper {paper_id}"),
                "chunks": []
            }
        papers_context[paper_id]["chunks"].append(chunk)

    # Format output
    formatted_parts = []
    for paper_id, paper_data in papers_context.items():
        formatted_parts.append(f"## {paper_data['title']}")
        for i, chunk in enumerate(paper_data["chunks"], 1):
            page = chunk.get("page_number", "?")
            section = chunk.get("section", "General")
            content = chunk.get("content", "").strip()
            formatted_parts.append(f"### Evidence {i} (Page {page}, {section}):")
            formatted_parts.append(content)
            formatted_parts.append("")

    return "\n".join(formatted_parts)


def format_qa_answer(answer: str, sources: list, confidence: str, key_evidence: list = None) -> dict:
    """
    Format final answer with sources and confidence.

    Args:
        answer: Generated answer from LLM
        sources: List of source dicts with keys: paper_title, page_number, section
        confidence: Confidence level (HIGH, MEDIUM, LOW, NO_EVIDENCE)
        key_evidence: Optional list of key evidence points

    Returns:
        Formatted answer dict with: answer, sources, confidence, key_evidence
    """
    formatted_sources = []
    for source in sources:
        paper_title = source.get("paper_title", "Unknown")
        page = source.get("page_number", "?")
        formatted_sources.append(f"{paper_title} — p. {page}")

    return {
        "answer": answer,
        "sources": formatted_sources,
        "confidence": confidence,
        "key_evidence": key_evidence or [],
        "raw_sources": sources  # Keep raw for internal use
    }


def calculate_confidence(retrieved_chunks: list) -> str:
    """
    Calculate confidence based on retrieval quality.

    Args:
        retrieved_chunks: List of retrieved chunks with 'similarity_score' or 'relevance_score'

    Returns:
        Confidence level: HIGH, MEDIUM, LOW, or NO_EVIDENCE
    """
    if not retrieved_chunks:
        return "NO_EVIDENCE"

    scores = [c.get("similarity_score", c.get("relevance_score", 0)) for c in retrieved_chunks]
    avg_score = sum(scores) / len(scores) if scores else 0

    # Confidence mapping (adjust thresholds as needed)
    if len(retrieved_chunks) >= 3 and avg_score >= 0.7:
        return "HIGH"
    elif len(retrieved_chunks) >= 2 and avg_score >= 0.5:
        return "MEDIUM"
    elif len(retrieved_chunks) >= 1:
        return "LOW"
    else:
        return "NO_EVIDENCE"


def extract_citations_from_chunks(chunks: list) -> list:
    """
    Extract citation metadata from chunks.

    Args:
        chunks: List of chunk dicts with paper metadata

    Returns:
        List of unique citation dicts with paper_title, page_number, section
    """
    citations = {}
    for chunk in chunks:
        key = (chunk.get("paper_title"), chunk.get("page_number"), chunk.get("section"))
        if key not in citations:
            citations[key] = {
                "paper_title": chunk.get("paper_title", "Unknown"),
                "page_number": chunk.get("page_number", "?"),
                "section": chunk.get("section", "General")
            }
    return list(citations.values())
