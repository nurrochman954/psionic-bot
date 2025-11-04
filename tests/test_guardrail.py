from tools.guardrail import quick_guardrail

def test_guardrail_flags_general_without_references():
    flags = quick_guardrail("Jawaban ini bersifat umum.")
    assert flags["too_general"] is True

def test_guardrail_detects_reference_phrase():
    flags = quick_guardrail("Rujukan: [book:abc, page:1]")
    assert flags["has_references"] is True
    assert flags["too_general"] is False
