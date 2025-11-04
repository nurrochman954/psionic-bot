from tools.citation_picker import pick_citations

class Doc:
    def __init__(self, text, page=None):
        self.page_content = text
        self.metadata = {"page": page}

def test_pick_citations_prioritizes_page_and_density():
    docs = [
        Doc("x"*100, page=None),           # score ~0.125
        Doc("y"*400, page=2),              # score 1 + 0.5 = 1.5
        Doc("z"*700, page=None),           # score ~0.875
        Doc("w"*50, page=10),              # score 1 + 0.0625 = 1.0625
    ]
    picked = pick_citations(docs, max_items=2)
    assert picked[0].metadata["page"] == 2
    assert len(picked) == 2
