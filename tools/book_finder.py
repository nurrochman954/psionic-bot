# tools/book_finder.py

from typing import Optional, Dict

def guess_book_focus(agent, query: str) -> Optional[Dict[str, str]]:
    q = (query or "").lower()
    catalog = agent.list_all_books()
    # exact substring first
    for coll, titles in catalog.items():
        for t in titles:
            if t.lower() in q:
                return {"collection": coll, "title": t}
    # normalized equals
    def norm(s: str) -> str:
        import re
        return re.sub(r"[^a-z0-9]+", "", s.lower())
    qn = norm(q)
    for coll, titles in catalog.items():
        for t in titles:
            if norm(t) == qn and t.strip():
                return {"collection": coll, "title": t}
    return None
