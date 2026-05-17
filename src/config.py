from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"
REPORTS_DIR = PROJECT_ROOT / "reports"

CSV_PATH = DATA_DIR / "trades.csv"

OLLAMA_CHAT_MODEL = "llama3.2:3b"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
