# Menguji util stateless dari psionic_agent tanpa inisialisasi kelas yang berat.
from types import SimpleNamespace
from psionic_agent import _trim_to_chars_by_sentence, PsionicAgent

class DummyDoc:
    def __init__(self, text, meta):
        self.page_content = text
        self.metadata = meta

def test_trim_to_chars_sentence_boundary():
    text = "Kalimat satu. Kalimat dua yang cukup panjang. Tiga."
    out = _trim_to_chars_by_sentence(text, max_chars=25)
    # Harus berhenti di batas kalimat pertama
    assert out.endswith(".")

def test_format_citations_staticmethod():
    d1 = DummyDoc("isi A", {"source": "/x/file.pdf", "page": 3, "book_title": "Book A"})
    d2 = DummyDoc("isi B sangat panjang " * 50, {"source": "/y/b.pdf", "page": 10, "book": "Book B"})
    lines = PsionicAgent.format_citations([d1, d2], max_len=40)
    assert "[book:Book A" in lines[0]
    assert "page:3" in lines[0]
    assert lines[1].endswith("...")
