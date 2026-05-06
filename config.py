"""Central configuration loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent

# ---- LLM (LiteLLM-compatible model strings) ----
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

if LLM_PROVIDER == "ollama":
    LLM_MODEL = f"ollama/{os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}"
    LLM_API_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
elif LLM_PROVIDER == "openai":
    LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    LLM_API_BASE = None
elif LLM_PROVIDER == "anthropic":
    LLM_MODEL = f"anthropic/{os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-5')}"
    LLM_API_BASE = None
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

def _resolve(p: str) -> Path:
    p = p.lstrip("./")
    return ROOT / p

# ---- Paths ----
DB_PATH = _resolve(os.getenv("DB_PATH", "close.db"))
VECTOR_DB_PATH = _resolve(os.getenv("VECTOR_DB_PATH", "chroma_db"))
REPORTS_DIR = _resolve(os.getenv("REPORTS_DIR", "reports_out"))
POLICY_DIR = ROOT / "data" / "policies"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Workflow ----
MAX_NARRATIVE_RETRIES = int(os.getenv("MAX_NARRATIVE_RETRIES", "2"))
RAG_CONFIDENCE_THRESHOLD = float(os.getenv("RAG_CONFIDENCE_THRESHOLD", "0.35"))
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))

# ---- RAG ----
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RAG_TOP_K = 4

# ---- Canonical schemas ----
TB_CANONICAL_COLS = ["period", "entity", "account", "account_name", "debit", "credit"]
TB_REQUIRED_COLS = ["period", "account", "debit", "credit"]

GL_CANONICAL_COLS = [
    "period", "entity", "txn_date", "journal_id", "account",
    "account_name", "description", "debit", "credit",
]
GL_REQUIRED_COLS = ["period", "txn_date", "account", "debit", "credit"]
