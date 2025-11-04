# tools/guardrail.py

def quick_guardrail(answer_text: str) -> dict:
    text = (answer_text or "").lower()
    flags = {
        "too_general": False,
        "has_references": ("rujukan:" in text),
    }
    if ("jawaban ini bersifat umum" in text) or (not flags["has_references"]):
        flags["too_general"] = True
    return flags
