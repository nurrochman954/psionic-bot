from agent_brain import _strip_meta

def test_strip_meta_removes_editorial_lines():
    txt = """Terima kasih atas masukan pemeriksa.
Ini tetap ada.
BAB 3
Berikut perbaikan jawaban.
Kalimat normal."""
    out = _strip_meta(txt)
    assert "Terima kasih atas masukan" not in out
    assert "BAB 3" not in out
    assert "perbaikan" not in out.lower()
    assert "Ini tetap ada." in out and "Kalimat normal." in out
