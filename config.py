"""All settings in one place. Edit paths and model names here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- data -------------------------------------------------------------
DATA_DIR = ROOT / "data"
STUDY_GUIDE_CHUNKS = DATA_DIR / "study_guide_chunks.jsonl"   # from Claude (73 chunks)
SLIDE_CHUNK_FILES = sorted((DATA_DIR / "slides").glob("*.jsonl"))  # from slide_rag_test.py, one per lecture
INDEX_DIR = ROOT / "index"                                   # built by `python rag.py index`

# ---- embeddings (sentence-transformers, runs in your PyTorch env) ------
# multilingual-e5-small: 470 MB, understands EN/ZH/TH queries, ~0.5 GB VRAM or fine on CPU.
# English-only alternative, smaller & faster: "BAAI/bge-small-en-v1.5" (set E5_PREFIXES = False).
EMBED_MODEL = "intfloat/multilingual-e5-small"
E5_PREFIXES = True            # e5 models want "query: " / "passage: " prefixes
EMBED_DEVICE = "cpu"          # "cuda" if you have VRAM to spare next to the LLM
EMBED_BATCH = 32

# ---- retrieval ----------------------------------------------------------
TOP_K = 6                     # chunks sent to the LLM
DENSE_K = 20                  # candidates from embeddings before fusion
BM25_K = 20                   # candidates from keyword search before fusion
RRF_K = 60                    # reciprocal-rank-fusion constant
EXPAND_CITED_SLIDES = 2       # per study-guide hit, add up to N of its cited slide chunks
MIN_BODY_CHARS = 60           # skip near-empty slides (title-only, logos, video placeholders)
SKIP_TITLE_RE = r"^(outline|agenda|contents?|review|practice|acknowledge?ments?|thanks?|q\s*&\s*a)\b"  # outline/recap slides
SKIP_DUPLICATE_SLIDES = True  # drop a slide whose body repeats an earlier slide of the same lecture
# Non-content slides to always skip (title, acknowledgement, outline, recap, "Thanks!").
# Checked against the 7 HEU NLP decks; add numbers here for new decks.
SKIP_SLIDES = {
    "Lecture01": [1, 6, 7, 13, 24, 37, 49, 50],
    "Lecture02": [1, 2, 4, 5, 10, 21, 27, 35, 40],
    "Lecture03": [1, 2, 3, 4, 18, 26, 38, 45, 53],
    "Lecture04": [1, 2, 3, 4, 17, 28, 37, 50, 51, 52, 53],
    "Lecture05": [1, 2, 3, 4, 14, 24, 29, 34, 42],
    "Lecture06": [1, 2, 3, 5, 6, 19, 30, 54],
    "Lecture07": [1, 2, 4, 17, 26, 36, 45, 57, 66, 67],
}
MAX_CONTEXT_CHARS = 9000      # ~2.5k tokens; keep well inside the LLM context window

# ---- generation (llama-server, OpenAI-compatible) ----------------------
LLM_URL = "http://127.0.0.1:8090/v1/chat/completions"
LLM_MODEL = "local"           # llama-server ignores the name, but the field is required
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 700
LLM_TIMEOUT = 300