# tools/citation_picker.py

from typing import List

def pick_citations(docs: List[object], max_items: int = 3) -> List[object]:
    """
    Heuristik sederhana: utamakan dokumen yang punya metadata page dan isi lebih padat.
    """
    scored = []
    for d in docs:
        md = getattr(d, "metadata", {}) or {}
        page = md.get("page") or 0
        content = (getattr(d, "page_content", "") or "")
        score = (1 if page else 0) + min(len(content), 800) / 800.0
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:max_items]]
