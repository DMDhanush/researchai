import re
from typing import List, Dict

SECTION_PATTERNS = [
    (r"\bAbstract\b", "Abstract"),
    (r"\bIntroduction\b", "Introduction"),
    (r"\bBackground\b", "Background"),
    (r"\bRelated\s+Work\b", "Related Work"),
    (r"\bLiterature\s+Review\b", "Literature Review"),
    (r"\bResearch\s+Gap\b", "Research Gap"),
    (r"\bObjectives\b", "Objectives"),
    (r"\bResearch\s+Questions\b", "Research Questions"),
    (r"\bMethodology\b|\bMethods\b", "Methodology"),
    (r"\bData\b|\bDataset\b", "Dataset"),
    (r"\bParticipants\b", "Participants"),
    (r"\bExperimental\s+Setup\b", "Experimental Setup"),
    (r"\bResults\b|\bFindings\b", "Results"),
    (r"\bDiscussion\b", "Discussion"),
    (r"\bLimitations\b", "Limitations"),
    (r"\bFuture\s+Work\b", "Future Work"),
    (r"\bConclusion\b", "Conclusion"),
    (r"\bReferences\b", "References"),
]

class SmartChunker:
    def __init__(self, chunk_size=800, overlap=150):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def detect_sections(self, pages: List[Dict]) -> List[Dict]:
        processed = []
        sect = "Introduction"
        for p in pages:
            text = p['content']
            for pat, name in SECTION_PATTERNS:
                if re.search(pat, text, re.IGNORECASE): sect = name
            processed.append({'page_number': p['page_number'], 'content': text, 'section': sect})
        return processed

    def chunk_document(self, pages: List[Dict], paper_id: int, user_id: int, paper_title: str, filename: str) -> List[Dict]:
        sections = self.detect_sections(pages)
        chunks, idx = [], 0
        all_toks = [{"word": w, "page_number": s["page_number"], "section": s["section"]} 
                    for s in sections for w in s["content"].split()]
        
        start = 0
        while start < len(all_toks):
            end = min(start + self.chunk_size, len(all_toks))
            ct = all_toks[start:end]
            cnt = " ".join([t["word"] for t in ct])
            if len(ct) >= 50:
                mid = all_toks[min(start + len(ct)//2, len(all_toks)-1)]
                chunks.append({'user_id': user_id, 'paper_id': paper_id, 'paper_title': paper_title, 'filename': filename, 'page_number': mid['page_number'], 'section': mid['section'], 'chunk_index': idx, 'content': cnt, 'word_count': len(ct)})
                idx += 1
            start += (self.chunk_size - self.overlap)
        return chunks
