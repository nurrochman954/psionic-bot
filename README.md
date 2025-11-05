# Psionic-Bot: Agen RAG (Retrieval-Augmented Generation) Discord

Proyek ini adalah implementasi sistem RAG (Retrieval-Augmented Generation) yang canggih, dihadirkan sebagai Discord Bot.

## Latar Belakang

Psionic-Bot dirancang sebagai **agen ahli (expert agent)**, bukan sebagai chatbot percakapan umum. Tujuannya adalah untuk menyediakan jawaban yang akurat, berdasar, dan dapat dilacak sumbernya, yang diambil secara eksklusif dari korpus dokumen tepercaya—dalam hal ini, koleksi buku-buku psikologi, terapi, dan pengembangan diri.

Filosofi utamanya adalah:

1.  **Berbasis Bukti (Evidence-Based)**: Setiap jawaban harus dapat ditelusuri kembali ke kutipan spesifik (`!source`) dari buku-buku yang telah diindeks. Bot ini menghindari "halusinasi" atau memberikan opini di luar konteks yang disediakan.
2.  **Kualitas di Atas Kuantitas**: Melalui alur *Critique & Refine*, bot secara internal mengoreksi draf jawabannya sendiri untuk memastikan akurasi dan kesesuaian dengan nada terapeutik/edukatif yang diminta.
3.  **Spesialisasi**: Bot ini adalah spesialis. Ia hanya mengetahui apa yang ada di dalam `vectorstore`-nya. Ia akan secara eksplisit menyatakan jika sebuah topik tidak ditemukan di dalam indeksnya.

Bot ini menggunakan arsitektur multi-komponen yang matang untuk mencapai tujuan ini, memastikan jawaban tidak hanya relevan tetapi juga akurat, berbasis sumber, dan melalui proses perbaikan mandiri (self-correction).

## Arsitektur & Alur Kerja

Sistem ini dibagi menjadi dua alur kerja utama: Ingesti Data (Offline) dan Pemrosesan Kueri (Online).

### 1. Alur Pengambilan Data

Data (buku dalam format PDF) tidak diproses secara real-time. Data tersebut harus diindeks terlebih dahulu menggunakan pipeline ingesti dari `Chunking_(v2).ipynb`.

1.  **Konfigurasi**: File `books_config.yaml` (dibuat oleh notebook) mendefinisikan kategori (koleksi ChromaDB) dan parameter *chunking* (ukuran & tumpang tindih) yang berbeda untuk setiap domain buku.
2.  **Pemuatan**: PDF dari folder sumber (`data/books/`) dimuat oleh notebook.
3.  **Pemisahan (Chunking)**: Teks dipecah menjadi potongan-potongan (chunks) yang lebih kecil sesuai parameter di konfigurasi.
4.  **Embedding**: Setiap potongan teks diubah menjadi vektor numerik menggunakan model embedding Google (misalnya, `models/text-embedding-004`).
5.  **Penyimpanan**: Vektor dan metadata (judul buku, halaman, dll.) disimpan ke dalam database vektor **ChromaDB** yang persisten di `bundle_psionic/vectorstore/`.

### 2. Alur Pemrosesan Kueri

Ini adalah *pipeline* utama yang dijalankan oleh `AgentBrain` setiap kali pengguna mengajukan pertanyaan.



1.  **Perencanaan (Plan)**: Saat kueri diterima, `AgentBrain` pertama-tama membuat rencana 3-5 langkah tentang cara terbaik untuk menjawab pertanyaan tersebut (`PLANNER_PROMPT`).
2.  **Deteksi Fokus Buku**: `AgentBrain` menggunakan `tools/book_finder.py` untuk menganalisis apakah kueri pengguna menyebutkan *judul buku tertentu* yang ada di dalam database.
3.  **Retrieval (Penarikan)**:
    * Jika fokus buku ditemukan, `PsionicAgent` akan memprioritaskan pencarian di dalam koleksi dan judul buku tersebut.
    * Jika tidak, pencarian dilakukan di semua koleksi.
    * **Retry**: Jika hasil pencarian awal buruk (misalnya, kurang dari 3 dokumen), `AgentBrain` akan otomatis mencoba lagi dengan kueri yang diperluas (`RETRY_KEYWORDS`) untuk mendapatkan hasil yang lebih baik.
4.  **Kompresi Konteks**: Dokumen yang diambil (misalnya, 12 dokumen) tidak dikirim seluruhnya ke LLM. `PsionicAgent` akan "memadatkan" konteks: 3 dokumen teratas disertakan secara penuh, sementara sisanya diringkas menjadi satu baris (`format_context_compact`).
5.  **Generasi Draf**: LLM menghasilkan draf jawaban pertama berdasarkan konteks yang dipadatkan dan riwayat percakapan (`PROMPT_RAG`).
6.  **Kritik (Critique)**: Draf jawaban tersebut *tidak langsung* dikirim ke pengguna. Draf ini dievaluasi secara internal oleh LLM lain menggunakan `CRITIC_PROMPT` untuk memeriksa: (1) Apakah didukung konteks? (2) Apakah rujukan spesifik? (3) Apakah ada klaim di luar konteks?
7.  **Perbaikan (Refine)**: Jika kritik menemukan "TIDAK", jawaban tersebut akan diperbaiki oleh LLM menggunakan `REFINE_PROMPT` untuk memastikan jawaban akhir hangat, terapeutik, dan akurat.
8.  **Pembersihan (Sanitize)**: Teks editorial/meta (seperti "Berikut perbaikan jawaban...") dihapus oleh `_strip_meta` sebelum dikirim.

## Fitur Utama

* **Manajemen Sesi**: Bot dapat masuk ke mode percakapan interaktif (`!new`), di mana ia akan merespons pesan non-perintah secara otomatis, dan dapat diakhiri (`!end`).
* **Kritik & Perbaikan Mandiri**: Alur *Critique & Refine* di `AgentBrain` memastikan jawaban memiliki kualitas yang lebih tinggi dan berani mengatakan "tidak ditemukan" jika tidak ada di konteks.
* **Pencarian Sadar-Buku**: Kemampuan untuk mendeteksi judul buku dalam kueri (`book_finder.py`) memungkinkan jawaban yang jauh lebih relevan.
* **Memori Persisten**: Bot mencatat riwayat percakapan harian dan ringkasan bergulir (rolling summary) ke disk (`storage/memory/`), memungkinkannya mengingat konteks antar sesi dan antar hari.
* **Konfigurasi Pengguna**: Pengguna dapat mengganti persona bot (`!style`) dan format jawaban (`!mode`) kapan saja.
* **Manajemen Status Lengkap**: Bot mengelola status pengguna (mode, gaya, DM, koleksi default) dan riwayat sesi jangka pendek (`USER_HISTORY`, `USER_SUMMARY`).

## Konfigurasi & Menjalankan

### 1. Persiapan Awal

* Pastikan Anda memiliki **Python 3.9** atau yang lebih baru.
* Clone repositori ini dan masuk ke direktorinya.
* Buat dan aktifkan virtual environment (disarankan):
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux/Mac
    # venv\Scripts\activate      # Windows
    ```
* Install dependensi:
    ```bash
    pip install -r requirements.txt
    ```

### 2. Konfigurasi Data (Vector Store)

**Langkah Kritis**: Bot ini tidak akan berfungsi tanpa database vektor.

1.  Unduh file `bundle_psionic.zip` (yang berisi vector store yang sudah jadi) dari tautan berikut:
    * **[Tautan Google Drive: bundle_psionic.zip](https://drive.google.com/file/d/1ixZnMxxzUwA7D-JX3JQUHRKFbmbHi9WI/view?usp=sharing)**
2.  Ekstrak file ZIP tersebut.
3.  Pastikan folder `bundle_psionic` yang telah diekstrak berada di direktori *root* proyek Anda, di level yang sama dengan `bot.py`.

*(Catatan: File `Chunking_(v2).ipynb` di repositori ini hanya sebagai referensi untuk menunjukkan bagaimana `vectorstore` dibuat. Anda tidak perlu menjalankannya jika sudah mengunduh file zip.)*

### 3. Konfigurasi .env

Salin `.env.example` menjadi `.env` dan Isi file `.env` dengan variabel berikut:

* `DISCORD_TOKEN`: Token bot Anda dari Discord Developer Portal.
* `GOOGLE_API_KEY`: Kunci API Google Anda, diperlukan untuk model embedding dan LLM (Gemini).

### 4. Menjalankan Bot

Setelah database vektor (`bundle_psionic/vectorstore`) siap dan `.env` terisi, jalankan bot:

```bash
python bot.py
```

## Daftar Perintah Bot

Berikut adalah daftar lengkap perintah yang tersedia di `bot.py`:

### Manajemen Sesi & Status

* `!new [koleksi] [mode=...] [style=...]`
    Memulai sesi percakapan interaktif. Bot akan otomatis merespons semua pesan Anda (tanpa perlu `!ask`) hingga Anda mengetik `!end`.
* `!end`
    Mengakhiri sesi interaktif. Bot akan berhenti merespons otomatis.
* `!status`
    Menampilkan status konfigurasi Anda saat ini (style, mode, koleksi default, status sesi).
* `!topic <judul>`
    Menetapkan judul/topik untuk sesi saat ini (hanya metadata).

### Pertanyaan & RAG

* `!ask <pertanyaan>`
    Mengajukan satu pertanyaan RAG ke bot.
* `!ask_in <koleksi> | <pertanyaan>`
    Mengajukan pertanyaan RAG, tetapi membatasi pencarian *hanya* ke koleksi yang ditentukan.
* `!source`
    Menampilkan kutipan ringkas (sumber) dari jawaban terakhir.
* `!source full <nomor>`
    Menampilkan kutipan penuh (teks asli) dari dokumen sumber nomor `<n>`.
* `!why`
    Menampilkan alasan dokumen terakhir dipilih (cuplikan mentah dan metadata).

### Konfigurasi

* `!style <netral|hangat|terapis|pengajar|rekan>`
    Mengubah gaya bahasa dan persona jawaban bot.
* `!mode <ringkas|panjang|bullet|banding|definisi|langkah>`
    Mengubah format keluaran jawaban (misalnya, `!mode bullet` untuk jawaban poin-poin).
* `!in <koleksi>`
    Mengatur koleksi default yang akan digunakan untuk semua kueri `!ask` atau sesi `!new`.
* `!in clear`
    Menghapus koleksi default (mencari di semua koleksi).
* `!dm <on|off>`
    Mengalihkan apakah bot harus membalas di channel atau melalui Direct Message (DM).

### Data & Memori

* `!collections`
    Menampilkan daftar semua koleksi (domain buku) yang tersedia di database.
* `!books [koleksi]`
    Menampilkan daftar semua judul buku yang ada di (semua atau spesifik) koleksi.
* `!recap`
    Meminta bot merangkum percakapan dalam sesi jangka pendek saat ini.
* `!clear`
    Membersihkan memori jangka pendek (riwayat dan ringkasan sesi saat ini).
* `!today` / `!yesterday`
    Membaca ringkasan harian dan topik dari memori persisten (`agent_memory.py`).

## Pengujian

Proyek ini dilengkapi dengan rangkaian unit test untuk memvalidasi fungsionalitas setiap komponen secara terisolasi.

* Menjalankan unit tests (memerlukan `pytest`):
    ```bash
    pytest
    ```
* Atau menggunakan `unittest` bawaan:
    ```bash
    python -m unittest discover -v tests
    ```

**Cakupan Pengujian Meliputi:**

* `test_agent_memory.py`: Memastikan penyimpanan dan pembacaan memori harian/bergulir berfungsi.
* `test_agent_brain_utils.py`: Memvalidasi fungsi utilitas seperti `_strip_meta` (penghapusan teks editorial).
* `test_book_finder.py`: Memastikan logika deteksi judul buku (termasuk normalisasi) bekerja.
* `test_citation_picker.py`: Menguji heuristik pemilihan kutipan terbaik.
* `test_guardrail.py`: Memverifikasi logika `quick_guardrail` (pengecekan rujukan).
* `test_psionic_utils.py`: Menguji utilitas pemformatan dan pemotongan teks di `PsionicAgent`.
* `test_session_manager.py`: Memvalidasi alur hidup (lifecycle) sesi (start, bump turn, end).