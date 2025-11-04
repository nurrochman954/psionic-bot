# psionic_agent.py

import os
import re
import time
import logging
from typing import List, Optional, Any, Tuple, Dict
from dotenv import load_dotenv

# Matikan telemetri/trace yang tidak perlu
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "noop"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "1"
logging.getLogger("chromadb").setLevel(logging.ERROR)

from chromadb.config import Settings
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate

# =========================
# PROMPTS (gaya "kita")
# =========================

PROMPT_RAG = ChatPromptTemplate.from_template(
    """Kita hanya menjawab dari KONTEN di bawah. Jika tidak memadai, katakan singkat bahwa jawaban berikut bersifat umum di luar kutipan buku.

{history_block}

Konteks:
{context}

Pertanyaan: {question}

Format jawaban yang diminta: {format_hint}

Gaya bahasa:
- Gunakan kata ganti inklusif "kita" (hindari "Anda").
- Hangat, terapeutik, dan langsung ke inti.
- Hindari nada editorial seperti "berikut perbaikan" atau "terima kasih atas masukan".

Struktur:
1) Inti jawaban 2–4 kalimat.
2) Contoh singkat satu kalimat yang relevan.
3) Rujukan: daftar sitasi [book:<judul>, page:<n>] (maksimal 3).

Jangan menyalin panjang-panjang dari konteks. Hindari istilah terlalu kaku bila ada padanannya di Indonesia.
"""
)

PROMPT_REWRITE = ChatPromptTemplate.from_template(
    """Perhalus teks agar terdengar alami sesuai gaya "{style}" tanpa mengubah fakta.
Gunakan kata ganti "kita" (hindari "Anda"). Jangan gunakan frasa meta seperti "berikut perbaikan".
Pertahankan blok "Rujukan:" beserta isinya tanpa perubahan.

Teks:
{draft}
"""
)

PROMPT_SUMMARIZE = ChatPromptTemplate.from_template(
    """Ringkas riwayat dialog berikut menjadi poin kontekstual pendek (maksimum 800 karakter).
Fokus pada tujuan, preferensi gaya, dan istilah yang sudah didefinisikan. Jangan menyimpulkan hal baru.

Riwayat:
{history_text}
"""
)

STYLE_HINTS = {
    "netral": "netral, profesional, langsung ke pokok",
    "hangat": "ramah, empatik, namun tetap ringkas",
    "terapis": "hangat, validatif, reflektif, fokus menenangkan",
    "pengajar": "jelas, bertahap, definisi→contoh",
    "rekan": "akrab seperlunya, non-formal ringan",
}

MODE_HINTS = {
    "ringkas": "Ringkas 3–5 kalimat.",
    "panjang": "Lebih detail 6–10 kalimat, tetap jelas.",
    "bullet": "Gunakan bullet points untuk ide utama.",
    "banding": "Tulis perbandingan A vs B bila relevan; gunakan bullet.",
    "definisi": "Mulai dengan definisi 1–2 kalimat, lalu poin kunci.",
    "langkah": "Berikan langkah-langkah praktis bernomor.",
}

# =========================
# Util pemangkasan aman
# =========================

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return _SENT_SPLIT.split(text)

def _trim_to_chars_by_sentence(text: str, max_chars: int) -> str:
    """Pangkas di batas kalimat agar makna tetap utuh."""
    if not text or len(text) <= max_chars:
        return text or ""
    sents = _split_sentences(text)
    out = []
    total = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if total + len(s) + (1 if out else 0) > max_chars:
            break
        out.append(s)
        total += len(s) + (1 if out else 0)
    joined = " ".join(out).strip()
    return joined if joined else text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."

# =========================
# PsionicAgent
# =========================

class PsionicAgent:
    def __init__(
        self,
        persist_dir: str,
        retrieval_k: int = 5,              # default ringan & cepat
        use_mmr: bool = False,             # default similarity (lebih cepat)
        mmr_lambda: float = 0.5,
        model_name: str = "gemini-2.5-flash",
        embed_name: str = "models/text-embedding-004",
    ) -> None:
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY tidak ditemukan di .env")

        self.persist_dir = os.path.abspath(persist_dir)
        self.retrieval_k = retrieval_k
        self.use_mmr = use_mmr
        self.mmr_lambda = mmr_lambda

        self.client = PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(model=embed_name)
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        self.rewriter = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)

        self.collections = [c.name for c in self.client.list_collections()]
        if not self.collections:
            raise RuntimeError("Tidak ada koleksi di vectorstore. Pastikan persist benar.")

        self._retrievers = {name: self._make_retriever(name) for name in self.collections}
        self._titles_cache: Optional[Dict[str, List[str]]] = None

        # cache retrieval sederhana (TTL 5 menit)
        self._ret_cache: Dict[Tuple[str, str, int, bool], Tuple[float, List[Any]]] = {}
        self._ret_ttl = 300.0

    # ---------- retriever ----------
    def _make_retriever(self, name: str):
        vs = Chroma(
            collection_name=name,
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
            client=self.client,
        )
        if self.use_mmr:
            return vs.as_retriever(
                search_type="mmr",
                search_kwargs={"k": self.retrieval_k, "lambda_mult": self.mmr_lambda},
            )
        return vs.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieval_k})

    def list_collections(self) -> List[str]:
        return list(self.collections)

    # ---------- retrieval + cache ----------
    def _cache_key(self, question: str, collection: Optional[str], k: int, use_mmr: bool) -> Tuple[str, str, int, bool]:
        qn = (question or "").strip().lower()
        coll = collection or "*"
        return (qn, coll, k, use_mmr)

    def _get_cache(self, key):
        now = time.time()
        if key in self._ret_cache:
            ts, docs = self._ret_cache[key]
            if now - ts <= self._ret_ttl:
                return docs
            else:
                self._ret_cache.pop(key, None)
        return None

    def _put_cache(self, key, docs):
        self._ret_cache[key] = (time.time(), docs)

    def retrieve(
        self,
        question: str,
        collection: Optional[str] = None,
        k_override: Optional[int] = None,
        use_mmr: Optional[bool] = None
    ) -> List[Any]:
        """
        Retrieval cepat dengan optional override k & MMR + cache.
        Kompatibel dengan agent_brain yang memanggil k_override/use_mmr.
        """
        k = k_override if k_override is not None else self.retrieval_k
        use_mmr_eff = self.use_mmr if use_mmr is None else use_mmr
        key = self._cache_key(question, collection, k, use_mmr_eff)
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        if collection:
            vs = Chroma(
                collection_name=collection,
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                client=self.client,
            )
            retr = vs.as_retriever(
                search_type="mmr" if use_mmr_eff else "similarity",
                search_kwargs={"k": k, **({"lambda_mult": self.mmr_lambda} if use_mmr_eff else {})},
            )
            try:
                docs = retr.invoke(question)
            except Exception:
                docs = []
            docs = self._dedupe(docs)[:k]
            self._put_cache(key, docs)
            return docs

        # lintas koleksi
        docs_all = []
        for name in self.collections:
            try:
                vs = Chroma(
                    collection_name=name,
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings,
                    client=self.client,
                )
                retr = vs.as_retriever(
                    search_type="mmr" if use_mmr_eff else "similarity",
                    search_kwargs={"k": k, **({"lambda_mult": self.mmr_lambda} if use_mmr_eff else {})},
                )
                docs_all.extend(retr.invoke(question))
            except Exception:
                continue
        docs = self._dedupe(docs_all)[:k]
        self._put_cache(key, docs)
        return docs

    @staticmethod
    def _dedupe(docs: List[Any]) -> List[Any]:
        seen = set()
        out = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            key = (meta.get("source"), meta.get("page"), meta.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    # ---------- title-aware ----------
    @staticmethod
    def _normalize_title(s: str) -> str:
        s = s.lower()
        return re.sub(r"[^a-z0-9]+", "", s)

    def list_books(self, collection_name: str) -> List[str]:
        if collection_name not in self.collections:
            return []
        coll = self.client.get_collection(collection_name)
        total = coll.count()
        titles = set()
        offset = 0
        limit = 1000
        while offset < total:
            batch = coll.get(include=["metadatas"], limit=limit, offset=offset)
            for md in batch.get("metadatas") or []:
                if not md:
                    continue
                title = md.get("book_title") or md.get("book")
                if title:
                    titles.add(str(title).strip())
            offset += limit
        return sorted(titles, key=lambda x: x.lower())

    def list_all_books(self) -> Dict[str, List[str]]:
        return {name: self.list_books(name) for name in self.collections}

    def _all_titles_by_collection(self) -> Dict[str, List[str]]:
        if self._titles_cache is not None:
            return self._titles_cache
        self._titles_cache = self.list_all_books()
        return self._titles_cache

    def _match_title(self, query: str) -> Optional[Tuple[str, str]]:
        q = query.lower()
        q_norm = self._normalize_title(query)
        catalog = self._all_titles_by_collection()
        for coll, titles in catalog.items():
            for t in titles:
                if t and t.lower() in q:
                    return coll, t
        for coll, titles in catalog.items():
            for t in titles:
                if t and self._normalize_title(t) == q_norm:
                    return coll, t
        return None

    def _filter_docs_by_title(self, docs: List[Any], title: str) -> List[Any]:
        t_norm = self._normalize_title(title)
        out = []
        for d in docs:
            md = getattr(d, "metadata", {}) or {}
            bt = md.get("book_title") or md.get("book") or ""
            if bt and (bt == title or self._normalize_title(bt) == t_norm):
                out.append(d)
        return out

    def retrieve_by_book(self, question: str, collection: str, book_title: str, k_override: Optional[int] = None) -> List[Any]:
        k = k_override if k_override is not None else max(self.retrieval_k, 12)
        try:
            vs = Chroma(
                collection_name=collection,
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
                client=self.client,
            )
            retr = vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            docs = self._dedupe(retr.invoke(question))
            filtered = self._filter_docs_by_title(docs, book_title)
            return (filtered or docs)[: self.retrieval_k]
        except Exception:
            return []

    def smart_retrieve(self, question: str) -> Tuple[List[Any], Optional[str], Optional[str]]:
        hit = self._match_title(question)
        if hit:
            coll, title = hit
            docs = self.retrieve_by_book(question, coll, title)
            if docs:
                return docs, coll, title
        return self.retrieve(question), None, None

    # ---------- formatting + kompresi aman ----------
    @staticmethod
    def _cite_line(meta: Dict) -> str:
        import os as _os
        src = meta.get("source", "unknown")
        page = meta.get("page")
        book_title = meta.get("book_title") or meta.get("book") or "unknown"
        return f"[book:{book_title}, source:{_os.path.basename(src)}, page:{page}]"

    def _format_full_block(self, d: Any, char_limit: int) -> str:
        meta = getattr(d, "metadata", {}) or {}
        content = (getattr(d, "page_content", str(d)) or "").strip()
        trimmed = _trim_to_chars_by_sentence(content, char_limit)
        return trimmed + "\n" + self._cite_line(meta)

    def _one_line_summary(self, d: Any, char_limit: int = 280) -> str:
        meta = getattr(d, "metadata", {}) or {}
        text = (getattr(d, "page_content", "") or "").strip().replace("\n", " ")
        if len(text) > char_limit:
            text = text[:char_limit].rstrip() + "..."
        return f"- {text} {self._cite_line(meta)}"

    def format_context_compact(
        self,
        docs: List[Any],
        full_top_n: int = 3,
        full_char_limit: int = 1200,
        tail_summaries_max: int = 3,
        tail_summary_char_limit: int = 280,
    ) -> str:
        """
        2–3 potongan dibawa utuh (dipangkas di batas kalimat),
        sisanya diringkas 1 kalimat/dokumen.
        """
        if not docs:
            return ""
        blocks = []
        top_full = docs[:full_top_n]
        for d in top_full:
            blocks.append(self._format_full_block(d, full_char_limit))
        tail = docs[full_top_n: full_top_n + tail_summaries_max]
        if tail:
            blocks.append("\nCatatan ringkas tambahan:")
            for d in tail:
                blocks.append(self._one_line_summary(d, tail_summary_char_limit))
        return "\n\n---\n\n".join(blocks).strip()

    @staticmethod
    def format_citations(docs: List[Any], max_len: int = 220) -> List[str]:
        import os as _os
        lines = []
        for i, d in enumerate(docs, start=1):
            meta = getattr(d, "metadata", {}) or {}
            src = meta.get("source", "unknown")
            page = meta.get("page")
            book_title = meta.get("book_title") or meta.get("book") or "unknown"
            content = (getattr(d, "page_content", "") or "").strip().replace("\n", " ")
            if len(content) > max_len:
                content = content[:max_len].rstrip() + "..."
            lines.append(f"{i}. [book:{book_title}, file:{_os.path.basename(src)}, page:{page}] — {content}")
        return lines

    # ---------- memory helpers ----------
    def summarize_history(self, pairs: List[Tuple[str, str]]) -> str:
        if not pairs:
            return ""
        text = "\n".join([f"User: {q}\nBot: {a}" for q, a in pairs if q or a])
        res = self.rewriter.invoke(PROMPT_SUMMARIZE.format_messages(history_text=text)).content
        return res.strip()

    def _history_block(self, history_window: List[Tuple[str, str]], memory_summary: Optional[str]) -> str:
        parts = []
        if memory_summary:
            parts.append(f"Ringkasan memori: {memory_summary}")
        if history_window:
            for q, a in history_window:
                q = q.strip().replace("\n", " ")
                a = a.strip().replace("\n", " ")
                parts.append(f"- Q: {q}\n  A: {a}")
        return "Riwayat singkat:\n" + ("\n".join(parts) if parts else "(tidak ada)")

    # ---------- generation ----------
    def answer_from_docs(
        self,
        docs: List[Any],
        question: str,
        style: str = "terapis",  # default warm therapeutic
        history_window: Optional[List[Tuple[str, str]]] = None,
        memory_summary: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        if not docs:
            return "Tidak ditemukan di indeks."
        history_block = self._history_block(history_window or [], memory_summary)
        format_hint = MODE_HINTS.get((mode or "").lower(), "Ikuti format default yang paling jelas.")
        context = self.format_context_compact(docs)  # kompresi aman
        draft = self.llm.invoke(PROMPT_RAG.format_messages(
            history_block=history_block,
            context=context,
            question=question,
            format_hint=format_hint,
        )).content
        style_label = STYLE_HINTS.get(style.lower(), STYLE_HINTS["terapis"])
        refined = self.rewriter.invoke(PROMPT_REWRITE.format_messages(draft=draft, style=style_label)).content
        return refined
