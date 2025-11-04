import types
from tools.book_finder import guess_book_focus

class DummyAgent:
    def __init__(self, catalog):
        self._catalog = catalog
    def list_all_books(self):
        return self._catalog

def test_guess_book_focus_exact_substring():
    agent = DummyAgent({"psy": ["Existential Psychotherapy"]})
    q = "Tolong jelaskan hal dari Existential Psychotherapy halaman 3"
    out = guess_book_focus(agent, q)
    assert out == {"collection": "psy", "title": "Existential Psychotherapy"}

def test_guess_book_focus_normalized_equals():
    agent = DummyAgent({"psy": ["Man's Search for Meaning"]})
    q = "mans search for meaning"
    out = guess_book_focus(agent, q)
    assert out == {"collection": "psy", "title": "Man's Search for Meaning"}
