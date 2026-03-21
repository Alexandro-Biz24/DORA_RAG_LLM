from pathlib import Path
import os
from dotenv import load_dotenv

# =========================
# ENV
# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# PATHS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdf"
MARKDOWN_DIR = DATA_DIR / "markdown"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"

FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.parquet"
BM25_PATH = VECTOR_STORE_DIR / "bm25.pkl"

# =========================
# MODELS
# =========================

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# =========================
# RETRIEVAL PARAMS
# =========================

TOP_K_VECTOR = 8
TOP_K_BM25 = 8
TOP_K_FINAL = 10

USE_BM25 = True

# =========================
# CHUNK PARAMS (rappel)
# =========================

MAX_CHUNK_CHARS = 2500
OVERLAP_CHARS = 350

# =========================
# GENERATION PARAMS
# =========================

MAX_CONTEXT_CHARS = 12000
TEMPERATURE = 0.2

# =========================
# REPORT PARAMS
# =========================

REPORT_MAX_TOKENS = 2000

# =========================
# DEBUG
# =========================

DEBUG = True