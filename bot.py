# bot.py

import os
import sys
import logging
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# ---- Quiet noisy libs
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "noop"
os.environ["CHROMA_TELEMETRY_DISABLED"] = "true"
os.environ["POSTHOG_DISABLED"] = "1"
logging.getLogger("chromadb").setLevel(logging.ERROR)

import discord
from discord.ext import commands, tasks

from psionic_agent import PsionicAgent
from agent_brain import AgentBrain
from agent_session import SessionManager
import agent_memory as mem

# ================== ENV & BOOT ==================
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./bundle_psionic/vectorstore")

if not DISCORD_TOKEN:
    print("ERROR: DISCORD_TOKEN tidak ditemukan di .env")
    sys.exit(1)

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

# ===== Defaults & per-user state =====
DEFAULT_STYLE = "terapis"      # warm therapeutic default
DEFAULT_MODE = "ringkas"
MEMORY_ON_DEFAULT = True
HISTORY_WINDOW_SIZE = 3
SUMMARY_TRIGGER_TURNS = 8

USER_STYLE: Dict[int, str] = {}
USER_MODE: Dict[int, str] = {}
USER_DM_PREF: Dict[int, bool] = {}
USER_LAST_DOCS: Dict[int, List[object]] = {}
USER_MEMORY_ON: Dict[int, bool] = {}
USER_HISTORY: Dict[int, List[Tuple[str, str]]] = {}
USER_SUMMARY: Dict[int, str] = {}
USER_DEFAULT_COLL: Dict[int, str] = {}

sessions = SessionManager()

def current_style(ctx) -> str:
    return USER_STYLE.get(ctx.author.id, DEFAULT_STYLE)

def current_mode(ctx) -> str:
    return USER_MODE.get(ctx.author.id, DEFAULT_MODE)

def memory_on(user_id: int) -> bool:
    return USER_MEMORY_ON.get(user_id, MEMORY_ON_DEFAULT)

async def safe_send(dest, text: str):
    if not text:
        return
    for i in range(0, len(text), 1900):
        await dest.send(text[i:i+1900])

async def reply_or_dm(ctx, text: str):
    if USER_DM_PREF.get(ctx.author.id, False) and ctx.guild is not None:
        try:
            await safe_send(await ctx.author.create_dm(), text)
            await ctx.reply("Jawaban dikirim ke DM Anda.", mention_author=False)
        except discord.Forbidden:
            await safe_send(ctx.channel, text)
    else:
        await safe_send(ctx.channel, text)

def get_history_window(user_id: int) -> List[Tuple[str, str]]:
    return USER_HISTORY.get(user_id, [])[-HISTORY_WINDOW_SIZE:]

def add_turn_and_maybe_summarize(user_id: int, question: str, answer: str, agent: PsionicAgent):
    pairs = USER_HISTORY.setdefault(user_id, [])
    pairs.append((question, answer))
    if len(pairs) >= SUMMARY_TRIGGER_TURNS:
        try:
            summary = agent.summarize_history(pairs)
            USER_SUMMARY[user_id] = summary
            USER_HISTORY[user_id] = pairs[-HISTORY_WINDOW_SIZE:]
        except Exception:
            USER_HISTORY[user_id] = pairs[-HISTORY_WINDOW_SIZE:]

# ================== Fancy Help (Embed + Pagination) ==================
from discord import Embed, ui

HELP_PAGES = [
    {
        "title": "Psionic Agent — Bantuan",
        "subtitle": "Meta & Sesi",
        "desc": "Perintah dasar untuk memulai & mengelola sesi.",
        "fields": [
            ("`!new [koleksi] [mode=<...>] [style=<...>]`", "Mulai sesi (auto-reply tanpa `!ask`)."),
            ("`!end`", "Akhiri sesi & simpan ringkasan."),
            ("`!topic <judul>`", "Set judul/topik sesi."),
            ("`!status`", "Lihat status konfigurasi & sesi."),
        ],
    },
    {
        "title": "Psionic Agent — Bantuan",
        "subtitle": "Tanya (RAG)",
        "desc": "Bertanya dengan retrieval & pembatasan koleksi.",
        "fields": [
            ("`!ask <pertanyaan>`", "Tanya bebas, title-aware, RAG."),
            ("`!ask_in <koleksi> | <pertanyaan>`", "Batasi retrieval ke koleksi tertentu."),
            ("`!in <koleksi>` / `!in clear`", "Set/Clear koleksi default."),
            ("`!books [koleksi]`", "Daftar judul buku (per koleksi atau semua)."),
            ("`!source` / `!source full <n>`", "Lihat sumber ringkas / kutipan penuh."),
            ("`!why`", "Alasan dokumen terpilih."),
            ("`!collections`", "Lihat daftar koleksi tersedia."),
        ],
    },
    {
        "title": "Psionic Agent — Bantuan",
        "subtitle": "Gaya & Mode",
        "desc": "Atur gaya bahasa dan bentuk jawaban.",
        "fields": [
            ("`!style <netral|hangat|terapis|pengajar|rekan>`", "Ganti persona jawaban."),
            ("`!mode <ringkas|panjang|bullet|banding|definisi|langkah>`", "Format keluaran."),
            ("`!dm on|off`", "Balas via DM atau channel."),
        ],
    },
    {
        "title": "Psionic Agent — Bantuan",
        "subtitle": "Memori & Ringkasan",
        "desc": "Fitur ringkasan harian dan memori singkat.",
        "fields": [
            ("`!today` / `!yesterday`", "Ringkasan harian & kemarin."),
            ("`!recap`", "Ringkasan sesi berjalan."),
            ("`!clear`", "Bersihkan memori singkat."),
        ],
    },
]

def build_help_embed(page_idx: int, author: discord.abc.User) -> Embed:
    page = HELP_PAGES[page_idx]
    emb = Embed(
        title=page["title"],
        description=f"**{page['subtitle']}**\n{page['desc']}",
        color=0x5865F2,
    )
    for name, value in page["fields"]:
        emb.add_field(name=name, value=value, inline=False)
    emb.set_footer(text=f"Halaman {page_idx+1}/{len(HELP_PAGES)} • Diminta oleh {author.display_name}")
    return emb

class HelpSelect(ui.Select):
    def __init__(self, view: "HelpView"):
        options = [discord.SelectOption(label=p["subtitle"], value=str(i)) for i, p in enumerate(HELP_PAGES)]
        super().__init__(placeholder="Lompat ke bagian…", min_values=1, max_values=1, options=options)
        self.view_ref = view

    async def callback(self, interaction: discord.Interaction):
        idx = int(self.values[0])
        self.view_ref.page = idx
        await interaction.response.edit_message(embed=build_help_embed(idx, self.view_ref.author), view=self.view_ref)

class HelpView(ui.View):
    def __init__(self, author: discord.abc.User):
        super().__init__(timeout=120)
        self.page = 0
        self.author = author
        self.add_item(HelpSelect(self))

    async def on_timeout(self):
        for child in self.children:
            if isinstance(child, ui.Button) or isinstance(child, ui.Select):
                child.disabled = True

    @ui.button(emoji="◀️", style=discord.ButtonStyle.secondary)
    async def prev(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.author.id:
            return await interaction.response.defer()
        self.page = (self.page - 1) % len(HELP_PAGES)
        await interaction.response.edit_message(embed=build_help_embed(self.page, self.author), view=self)

    @ui.button(emoji="▶️", style=discord.ButtonStyle.secondary)
    async def next(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.author.id:
            return await interaction.response.defer()
        self.page = (self.page + 1) % len(HELP_PAGES)
        await interaction.response.edit_message(embed=build_help_embed(self.page, self.author), view=self)

# ================== Presence Tips (rotating status) ==================
PRESENCE_TIPS = [
    "Ketik !new untuk memulai sesi",
    "Langsung tanya: !ask <pertanyaan>",
    "Akhiri sesi dengan !end",
    "Lihat bantuan: !help",
]
@tasks.loop(seconds=25)
async def rotate_presence():
    if not bot.is_ready():
        return
    idx = (rotate_presence.current_loop) % len(PRESENCE_TIPS)
    await bot.change_presence(
        activity=discord.Activity(type=discord.ActivityType.watching, name=PRESENCE_TIPS[idx]),
        status=discord.Status.online,
    )

# ================== EVENTS ==================
@bot.event
async def on_ready():
    try:
        bot.agent = PsionicAgent(persist_dir=PERSIST_DIR)
        bot.brain = AgentBrain(bot.agent)
        print("Bot siap. Koleksi:", ", ".join(bot.agent.list_collections()))
    except Exception as e:
        print("Gagal inisialisasi:", e)
        await bot.close()
        return
    # start rotating presence
    if not rotate_presence.is_running():
        rotate_presence.start()

# ================== COMMANDS ==================

@bot.command(name="help")
async def help_cmd(ctx):
    view = HelpView(ctx.author)
    await ctx.reply(embed=build_help_embed(0, ctx.author), view=view, mention_author=False)

@bot.command(name="style")
async def style_cmd(ctx, *, style: str):
    style = style.strip().lower()
    if style not in ["netral","hangat","terapis","pengajar","rekan"]:
        await reply_or_dm(ctx, "Pilihan: netral | hangat | terapis | pengajar | rekan")
        return
    USER_STYLE[ctx.author.id] = style
    await reply_or_dm(ctx, f"Gaya disetel: {style}")

@bot.command(name="mode")
async def mode_cmd(ctx, *, mode: str):
    mode = mode.strip().lower()
    if mode not in ["ringkas","panjang","bullet","banding","definisi","langkah"]:
        await reply_or_dm(ctx, "Pilihan: ringkas | panjang | bullet | banding | definisi | langkah")
        return
    USER_MODE[ctx.author.id] = mode
    await reply_or_dm(ctx, f"Mode disetel: {mode}")

@bot.command(name="dm")
async def dm_cmd(ctx, *, arg: str):
    opt = arg.strip().lower()
    if opt == "on":
        USER_DM_PREF[ctx.author.id] = True
        await reply_or_dm(ctx, "DM mode diaktifkan.")
    elif opt == "off":
        USER_DM_PREF[ctx.author.id] = False
        await reply_or_dm(ctx, "DM mode dinonaktifkan.")
    else:
        await reply_or_dm(ctx, "Gunakan: !dm on  atau  !dm off")

@bot.command(name="in")
async def in_cmd(ctx, *, arg: str):
    arg = arg.strip().lower()
    if arg == "clear":
        USER_DEFAULT_COLL.pop(ctx.author.id, None)
        s = sessions.get(ctx.author.id, ctx.channel.id)
        if s and s.is_on:
            s.default_collection = None
        await reply_or_dm(ctx, "Koleksi default dihapus.")
        return
    cols = bot.agent.list_collections()
    if arg not in cols:
        await reply_or_dm(ctx, "Koleksi tidak dikenal. Gunakan !collections.")
        return
    USER_DEFAULT_COLL[ctx.author.id] = arg
    s = sessions.get(ctx.author.id, ctx.channel.id)
    if s and s.is_on:
        s.default_collection = arg
    await reply_or_dm(ctx, f"Koleksi default disetel: {arg}")

@bot.command(name="collections")
async def collections_cmd(ctx):
    await reply_or_dm(ctx, "Koleksi:\n" + "\n".join(bot.agent.list_collections()))

@bot.command(name="books")
async def books_cmd(ctx, *, collection: str = None):
    try:
        if collection:
            collection = collection.strip()
            if collection not in bot.agent.list_collections():
                await reply_or_dm(ctx, "Koleksi tidak dikenal. Gunakan !collections.")
                return
            titles = bot.agent.list_books(collection)
            if not titles:
                await reply_or_dm(ctx, f'Tidak ada judul di "{collection}".')
                return
            await reply_or_dm(ctx, f'Buku dalam "{collection}":\n- ' + "\n- ".join(titles))
        else:
            mapping = bot.agent.list_all_books()
            parts = []
            for name in bot.agent.list_collections():
                titles = mapping.get(name, [])
                parts.append(f'[{name}]\n- ' + ("\n- ".join(titles) if titles else "(kosong)"))
            await reply_or_dm(ctx, "\n\n".join(parts))
    except Exception as e:
        await reply_or_dm(ctx, f"Gagal memuat daftar buku: {e}")

@bot.command(name="ask")
async def ask_cmd(ctx, *, question: str):
    style = current_style(ctx); mode = current_mode(ctx)
    async with ctx.channel.typing():
        try:
            default_coll = USER_DEFAULT_COLL.get(ctx.author.id)
            hw = get_history_window(ctx.author.id) if memory_on(ctx.author.id) else []
            ms = USER_SUMMARY.get(ctx.author.id) if memory_on(ctx.author.id) else None

            answer, docs, meta = bot.brain.answer_with_pipeline(
                user_id=ctx.author.id,
                question=question,
                style=style,
                mode=mode,
                history_window=hw,
                memory_summary=ms,
                default_collection=default_coll,
            )
            USER_LAST_DOCS[ctx.author.id] = docs
        except Exception as e:
            answer = f"Terjadi kesalahan: {e}"
    await reply_or_dm(ctx, answer)

    if not answer.startswith("Terjadi kesalahan"):
        add_turn_and_maybe_summarize(ctx.author.id, question, answer, bot.agent)
        mem.append_turn(ctx.author.id, question, answer)

@bot.command(name="ask_in")
async def ask_in_cmd(ctx, *, arg: str):
    style = current_style(ctx); mode = current_mode(ctx)
    if "|" not in arg:
        await reply_or_dm(ctx, "Format: !ask_in <nama_koleksi> | <pertanyaan>")
        return
    collection, question = [s.strip() for s in arg.split("|", 1)]
    if collection not in bot.agent.list_collections():
        await reply_or_dm(ctx, "Koleksi tidak dikenal. Gunakan !collections.")
        return

    async with ctx.channel.typing():
        try:
            hw = get_history_window(ctx.author.id) if memory_on(ctx.author.id) else []
            ms = USER_SUMMARY.get(ctx.author.id) if memory_on(ctx.author.id) else None
            docs = bot.agent.retrieve(question, collection=collection)
            USER_LAST_DOCS[ctx.author.id] = docs
            answer = bot.agent.answer_from_docs(docs, question, style, hw, ms, mode)
        except Exception as e:
            answer = f"Terjadi kesalahan: {e}"
    await reply_or_dm(ctx, answer)
    if not answer.startswith("Terjadi kesalahan"):
        add_turn_and_maybe_summarize(ctx.author.id, question, answer, bot.agent)
        mem.append_turn(ctx.author.id, question, answer)

@bot.command(name="source")
async def source_cmd(ctx, *args):
    docs = USER_LAST_DOCS.get(ctx.author.id)
    if not docs:
        await reply_or_dm(ctx, "Belum ada sumber. Lakukan !ask dulu.")
        return
    if len(args) >= 2 and args[0].lower() == "full":
        try:
            idx = int(args[1])
        except ValueError:
            await reply_or_dm(ctx, "Gunakan: !source full <nomor>")
            return
        if idx < 1 or idx > len(docs):
            await reply_or_dm(ctx, f"Nomor tidak valid. 1..{len(docs)}")
            return
        from psionic_agent import PsionicAgent as _PA
        await reply_or_dm(ctx, _PA.format_citations([docs[idx-1]], max_len=1000)[0])
        return
    lines = bot.agent.format_citations(docs, max_len=220)
    await reply_or_dm(ctx, "Sumber terakhir:\n" + "\n".join(lines))

@bot.command(name="why")
async def why_cmd(ctx):
    docs = USER_LAST_DOCS.get(ctx.author.id)
    if not docs:
        await reply_or_dm(ctx, "Belum ada data retrieval. Lakukan !ask.")
        return
    lines = []
    for i, d in enumerate(docs, start=1):
        md = getattr(d, "metadata", {}) or {}
        book = md.get("book_title") or md.get("book") or "unknown"
        page = md.get("page"); src = md.get("source")
        snippet = (getattr(d, "page_content", "") or "").strip().replace("\n"," ")
        if len(snippet) > 180: snippet = snippet[:180].rstrip() + "..."
        lines.append(f"{i}. book={book}; page={page}; source={src}\n   {snippet}")
    await reply_or_dm(ctx, "\n".join(lines))

@bot.command(name="today")
async def today_cmd(ctx):
    data = mem.read_daily(ctx.author.id)
    if not data["turns"] and not data["daily_summary"]:
        await reply_or_dm(ctx, "Belum ada percakapan hari ini.")
        return
    out = []
    if data["daily_summary"]:
        out.append("Ringkasan hari ini:\n" + data["daily_summary"])
    if data["turns"]:
        out.append("\nTopik hari ini (singkat):")
        for i, t in enumerate(data["turns"][-5:], start=max(1, len(data["turns"])-4)):
            out.append(f"{i}. {t['q'][:120]}")
    await reply_or_dm(ctx, "\n".join(out))

@bot.command(name="yesterday")
async def yesterday_cmd(ctx):
    y = mem.yesterday_date_str()
    data = mem.read_daily(ctx.author.id, y)
    if not data["turns"] and not data["daily_summary"]:
        await reply_or_dm(ctx, "Tidak ada data kemarin.")
        return
    out = []
    if data["daily_summary"]:
        out.append(f"Ringkasan {y}:\n" + data["daily_summary"])
    if data["turns"]:
        out.append("\nTopik kemarin (singkat):")
        for i, t in enumerate(data["turns"][-5:], start=max(1, len(data["turns"])-4)):
            out.append(f"{i}. {t['q'][:120]}")
    await reply_or_dm(ctx, "\n".join(out))

@bot.command(name="recap")
async def recap_cmd(ctx):
    pairs = USER_HISTORY.get(ctx.author.id, [])
    if not pairs:
        await reply_or_dm(ctx, "Belum ada riwayat singkat.")
        return
    try:
        summary = bot.agent.summarize_history(pairs[-8:])
        await reply_or_dm(ctx, "Ringkasan percakapan berjalan:\n" + summary)
    except Exception as e:
        await reply_or_dm(ctx, f"Gagal merangkum: {e}")

@bot.command(name="clear")
async def clear_cmd(ctx):
    USER_HISTORY.pop(ctx.author.id, None)
    USER_SUMMARY.pop(ctx.author.id, None)
    await reply_or_dm(ctx, "Memori singkat dibersihkan.")

@bot.command(name="status")
async def status_cmd(ctx):
    s = sessions.get(ctx.author.id, ctx.channel.id)
    coll = USER_DEFAULT_COLL.get(ctx.author.id, "(none)")
    style = current_style(ctx); mode = current_mode(ctx)
    mem_on = "on" if memory_on(ctx.author.id) else "off"
    dm = "on" if USER_DM_PREF.get(ctx.author.id, False) else "off"
    session_line = f"session: {'on' if (s and s.is_on) else 'off'}; topic: {getattr(s,'topic',None)}; turns: {getattr(s,'turns',0)}; session_coll: {getattr(s,'default_collection', None)}"
    await reply_or_dm(ctx,
        f"Status:\n- style: {style}\n- mode: {mode}\n- mem: {mem_on}\n- dm: {dm}\n- default collection: {coll}\n- {session_line}"
    )

# ---- Session controls ----
@bot.command(name="new")
async def new_cmd(ctx, *, args: str = ""):
    default_coll = None; style = current_style(ctx); mode = current_mode(ctx)
    tokens = [a.strip() for a in args.split() if a.strip()] if args else []
    for tok in tokens:
        if tok.startswith("mode="):
            mode = tok.split("=",1)[1].lower()
        elif tok.startswith("style="):
            style = tok.split("=",1)[1].lower()
        else:
            default_coll = tok
    if default_coll and default_coll not in bot.agent.list_collections():
        await reply_or_dm(ctx, "Koleksi tidak dikenal. Gunakan !collections.")
        return
    s = sessions.start(ctx.author.id, ctx.channel.id, default_coll, style, mode)
    await reply_or_dm(ctx, f"Sesi baru dimulai. Mode: {mode}, Style: {style}, Koleksi: {default_coll or '(semua)'}.\nTanyakan apa pun.")

@bot.command(name="topic")
async def topic_cmd(ctx, *, title: str):
    sessions.set_topic(ctx.author.id, ctx.channel.id, title.strip())
    await reply_or_dm(ctx, f"Topik sesi disetel: {title.strip()}")

@bot.command(name="end")
async def end_cmd(ctx):
    s = sessions.end(ctx.author.id, ctx.channel.id)
    if not s:
        await reply_or_dm(ctx, "Tidak ada sesi aktif.")
        return
    pairs = USER_HISTORY.get(ctx.author.id, [])[-8:]
    try:
        daily = bot.agent.summarize_history(pairs) if pairs else ""
        mem.update_daily_summary(ctx.author.id, daily)
        rolling = mem.read_rolling_summary(ctx.author.id)
        combined = (rolling + " " + daily).strip()[:800]
        mem.update_rolling_summary(ctx.author.id, combined)
    except Exception:
        pass
    await reply_or_dm(ctx, "Sesi diakhiri. Ringkasan harian diperbarui.")

# ---- Auto-reply during active session ----
@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    await bot.process_commands(message)  # prioritas command
    if message.content.startswith("!"):
        return
    s = sessions.get(message.author.id, message.channel.id)
    if not s or not s.is_on:
        return

    style = s.style; mode = s.mode
    default_coll = s.default_collection or USER_DEFAULT_COLL.get(message.author.id)

    hw = USER_HISTORY.get(message.author.id, [])[-HISTORY_WINDOW_SIZE:] if memory_on(message.author.id) else []
    ms = USER_SUMMARY.get(message.author.id) if memory_on(message.author.id) else None
    try:
        async with message.channel.typing():
            answer, docs, meta = bot.brain.answer_with_pipeline(
                user_id=message.author.id,
                question=message.content,
                style=style,
                mode=mode,
                history_window=hw,
                memory_summary=ms,
                default_collection=default_coll,
            )
            USER_LAST_DOCS[message.author.id] = docs
        await safe_send(message.channel, answer)
        add_turn_and_maybe_summarize(message.author.id, message.content, answer, bot.agent)
        mem.append_turn(message.author.id, message.content, answer)
        sessions.bump_turn(message.author.id, message.channel.id)
    except Exception as e:
        await safe_send(message.channel, f"Terjadi kesalahan: {e}")

# ================== RUN ==================
if __name__ == "__main__":
    bot.run(DISCORD_TOKEN)
