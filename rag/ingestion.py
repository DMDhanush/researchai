"""
ResearchAI — Document Ingestion Module
Handles PDF, TXT, and MD file ingestion with metadata extraction.
Uses PyMuPDF (fitz) for PDF processing.
"""
import os
import re
from collections import Counter
from pathlib import Path
from typing import Optional
import fitz

class DocumentIngester:
    """Ingests research documents and extracts structured content."""
    SUPPORTED_EXTENSIONS = {'.pdf', '.txt', '.md'}

    @staticmethod
    def validate_file(filepath: str) -> dict:
        path = Path(filepath)
        if not path.exists(): return {"valid": False, "error": "File not found"}
        if not path.is_file(): return {"valid": False, "error": "Not a file"}
        ext = path.suffix.lower()
        if ext not in DocumentIngester.SUPPORTED_EXTENSIONS:
            return {"valid": False, "error": "Unsupported file type"}
        if path.stat().st_size == 0: return {"valid": False, "error": "File empty"}
        return {"valid": True, "error": None}

    @staticmethod
    def extract_pdf(filepath: str) -> dict:
        pages = []
        full_text_parts = []
        doc = fitz.open(filepath)
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                raw = page.get_text("text") or ""
                cleaned = DocumentIngester.clean_text(raw)
                pages.append({"page_number": page_num + 1, "content": cleaned, "word_count": len(cleaned.split())})
                full_text_parts.append(cleaned)
        finally: doc.close()
        full_text = "\n\n".join(full_text_parts)
        return {"pages": pages, "full_text": full_text, "page_count": len(pages), "metadata": DocumentIngester.extract_metadata(full_text, pages), "is_scanned": DocumentIngester.is_scanned_pdf(pages)}

    @staticmethod
    def extract_text_file(filepath: str) -> dict:
        raw = Path(filepath).read_text(encoding="utf-8", errors="replace")
        cleaned = DocumentIngester.clean_text(raw)
        pages = [{"page_number": 1, "content": cleaned, "word_count": len(cleaned.split())}]
        return {"pages": pages, "full_text": cleaned, "page_count": 1, "metadata": DocumentIngester.extract_metadata(cleaned, pages), "is_scanned": False}

    @staticmethod
    def ingest(filepath: str) -> dict:
        val = DocumentIngester.validate_file(filepath)
        if not val["valid"]: raise RuntimeError(val["error"])
        return DocumentIngester.extract_pdf(filepath) if Path(filepath).suffix.lower() == ".pdf" else DocumentIngester.extract_text_file(filepath)

    @staticmethod
    def extract_metadata(full_text: str, pages: list[dict]) -> dict:
        return {"title": DocumentIngester._extract_title(full_text), "authors": DocumentIngester._extract_authors(full_text), "abstract": DocumentIngester._extract_abstract(full_text), "pub_year": DocumentIngester._extract_year(full_text), "keywords": DocumentIngester._extract_keywords(full_text)}

    @staticmethod
    def _extract_title(text: str) -> Optional[str]:
        for line in text.split("\n"):
            s = line.strip()
            if len(s) < 5 or re.match(r"^(page\s*\d+|https?://|\d+\s*$)", s, re.IGNORECASE): continue
            return s[:300]
        return None

    @staticmethod
    def _extract_authors(text: str) -> Optional[str]:
        m = re.search(r"\b(abstract)\b", text, re.IGNORECASE)
        header = text[:m.start() if m else 2000]
        lines = [l.strip() for l in header.split("\n") if l.strip()]
        cands = [l for l in lines[1:] if (sum(c.isalpha() or c.isspace() or c in ",.&-" for c in l)/max(len(l),1) > 0.75 and len(l) > 5)]
        return "; ".join(cands[:3])[:500] if cands else None

    @staticmethod
    def _extract_abstract(text: str) -> Optional[str]:
        p = re.compile(r"\babstract\b\s*[:\-—]?\s*\n?(.*?)(?=\n\s*(?:1[\.\s]|I[\.\s]|introduction|keywords|key\s*words)\b)", re.IGNORECASE | re.DOTALL)
        m = p.search(text)
        if m and len(m.group(1).strip()) > 50: return m.group(1).strip()[:3000]
        return None

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        years = re.findall(r"\b(20[0-3]\d)\b", text[:3000])
        return int(Counter(years).most_common(1)[0][0]) if years else None

    @staticmethod
    def _extract_keywords(text: str) -> Optional[str]:
        m = re.search(r"\b(?:keywords?|key\s*words?)\s*[:\-—]\s*(.+?)(?:\n\s*\n|\n\s*(?:1[\.\s]|I[\.\s]|introduction)\b)", text, re.IGNORECASE | re.DOTALL)
        return re.sub(r"\s+", " ", m.group(1).strip())[:500] if m else None

    @staticmethod
    def is_scanned_pdf(pages: list[dict]) -> bool:
        return sum(p["word_count"] for p in pages) / len(pages) < 20 if pages else True

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return '\n'.join(l.strip() for l in text.split('\n') if not re.match(r'^\d{1,4}$', l.strip())).strip()
