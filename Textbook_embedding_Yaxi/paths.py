"""Repository path helpers (portable across machines)."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE_DIR = Path(__file__).resolve().parent
DATA_DIR = MODULE_DIR / "data"
OUTPUTS_DIR = MODULE_DIR / "outputs"
TEXTBOOK_DIR = REPO_ROOT / "textbook"

SEARCH_FAISS_SCRIPT = MODULE_DIR / "search_faiss.py"
FAISS_INDEX = TEXTBOOK_DIR / "biomaterials_index.faiss"
METADATA_CSV = TEXTBOOK_DIR / "biomaterials_metadata.csv"
EXPORT_TEXTBOOK_CSV = TEXTBOOK_DIR / "export_textbook.csv"

SCQ_BANK_JSON = DATA_DIR / "scq_bank.json"
OPEN_ENDED_BANK_JSON = DATA_DIR / "question_bank_open_ended.json"


def fetch_rag_context(query: str) -> str:
    import subprocess
    import sys

    results = subprocess.check_output(
        [sys.executable, str(SEARCH_FAISS_SCRIPT), query],
        text=True,
    )
    chunks = [line for line in results.strip().split("\n") if line]
    return "\n\n".join(chunks)
