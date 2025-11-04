# agent_brain.py

from typing import List, Tuple, Optional
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from psionic_agent import PsionicAgent
from tools.book_finder import guess_book_focus
import re

PLANNER_PROMPT = ChatPromptTemplate.from_template(
    """Buat rencana singkat 3–5 langkah untuk menjawab pertanyaan berikut dengan mengandalkan kutipan dari konteks buku (RAG).
Formatkan sebagai bullet, setiap bullet 1 kalimat, tanpa narasi tambahan.

Pertanyaan: {question}
Mode yang diminta: {mode}
"""
)

CRITIC_PROMPT = ChatPromptTemplate.from_template(
    """Anda adalah pemeriksa singkat.
Periksa draf jawaban berikut terhadap konteks sitasi.

Jawab dengan tiga baris "YA/TIDAK: alasan singkat" untuk:
1) Didukung konteks?
2) Rujukan cukup spesifik?
3) Ada klaim di luar konteks?

Draf:
{answer}
"""
)

# Refine yang natural: hindari frasa "perbaikan", langsung jadi jawaban akhir
REFINE_PROMPT = ChatPromptTemplate.from_template(
    """Perhalus jawaban berikut agar hangat, terapeutik, dan langsung ke inti.
Gaya bahasa:
- Gunakan kata ganti inklusif "kita" (hindari "Anda").
- Hindari kalimat meta (mis. 'berikut perbaikan', 'terima kasih atas masukan').

Jika sebagian informasi tidak terdapat dalam kutipan buku, sebutkan singkat bahwa sisanya bersifat umum.
Jaga blok "Rujukan:" bila ada, jangan ubah formatnya.

Jawaban awal:
{answer}

Kritik:
{critique}

Tulis versi akhir yang siap diberikan ke pengguna.
"""
)

RETRY_KEYWORDS = " parasosial identifikasi pembaca empati narrative transportation attachment media psikologi"

# Pola meta language yang harus dihapus
META_REMOVE = [
    r"^\s*terima kasih atas masukan pemeriksa.*",
    r"^\s*saya.*model bahasa.*",
    r"^\s*namun, saya dapat memperbaiki jawaban.*",
    r"^\s*berikut adalah perbaikan.*",
    r"^\s*berikut.*perbaikan jawaban.*",
    r"^\s*mari kita (perbaiki|rapikan).*",
    r"^\s*tentu[, ]+ini perbaikan.*",
    r"^\s*berdasarkan masukan pemeriksa.*",
    r"^\s*PART\s+[IVXLC]+\b",
    r"^\s*BAB\s+\d+\b",
    r"^\s*CHAPTER\s+\d+\b",
    r"^\s*\d+\s*[•\-\u2022]\s+.*$",
]

def _strip_meta(answer: str) -> str:
    if not answer:
        return answer
    lines = answer.splitlines()
    clean = []
    for ln in lines:
        drop = False
        for p in META_REMOVE:
            if re.search(p, ln, flags=re.IGNORECASE):
                drop = True
                break
        if not drop:
            clean.append(ln)
    out = "\n".join(clean).strip()
    return out or answer

class AgentBrain:
    def __init__(self, agent: PsionicAgent, model_name: str = "gemini-2.5-flash"):
        self.agent = agent
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0)

    def plan(self, question: str, mode: str) -> List[str]:
        res = self.llm.invoke(PLANNER_PROMPT.format_messages(question=question, mode=mode)).content
        steps = [ln.strip("-• ").strip() for ln in res.splitlines() if ln.strip()]
        return steps[:5]

    def answer_with_pipeline(
        self,
        user_id: int,
        question: str,
        style: str,
        mode: str,
        history_window: List[Tuple[str, str]],
        memory_summary: Optional[str],
        default_collection: Optional[str] = None,
    ) -> Tuple[str, List[object], dict]:
        # 0) planning untuk mode non-ringan
        do_plan = mode in ("panjang", "banding", "langkah", "definisi")
        plan_steps = self.plan(question, mode) if do_plan else []

        # 1) fokus judul / koleksi
        if default_collection:
            docs = self.agent.retrieve(question, collection=default_collection, k_override=5, use_mmr=False)
        else:
            book_focus = guess_book_focus(self.agent, question)
            if book_focus:
                docs = self.agent.retrieve_by_book(question, book_focus["collection"], book_focus["title"], k_override=12)
            else:
                docs, _, _ = self.agent.smart_retrieve(question)

        # 1b) retry recall-first jika bukti < 3
        if len(docs) < 3:
            aug_q = (question or "") + RETRY_KEYWORDS
            rd = self.agent.retrieve(aug_q, collection=default_collection, k_override=12, use_mmr=False) if default_collection \
                 else self.agent.retrieve(aug_q, k_override=12, use_mmr=False)
            seen = set(); merged = []
            for d in (docs + rd):
                md = getattr(d, "metadata", {}) or {}
                key = (md.get("source"), md.get("page"), md.get("chunk_index"))
                if key not in seen:
                    seen.add(key); merged.append(d)
            docs = merged[: max(self.agent.retrieval_k, 12)]

        # 2) generate jawaban (psionic_agent sudah kompres konteks aman)
        answer = self.agent.answer_from_docs(
            docs,
            question=question,
            style=style,
            history_window=history_window,
            memory_summary=memory_summary,
            mode=mode,
        )

        # 3) kritik & refine bila perlu
        critique = self.llm.invoke(CRITIC_PROMPT.format_messages(answer=answer)).content
        need_refine = "TIDAK" in critique or ("Rujukan:" not in answer)
        if need_refine:
            answer = self.llm.invoke(REFINE_PROMPT.format_messages(answer=answer, critique=critique)).content

        # 4) sanitasi meta/editorial supaya langsung natural
        answer = _strip_meta(answer)

        # 5) fallback lembut jika tanpa rujukan (gaya terapeutik)
        if "Rujukan:" not in answer:
            answer = ("Sepertinya bagian ini tidak dijelaskan secara eksplisit dalam kutipan buku. "
                      "Namun secara umum—dan dari sudut pandang psikologis—\n\n" + answer)

        meta = {
            "plan_steps": plan_steps,
            "critique": critique,
        }
        return answer, docs, meta
