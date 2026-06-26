import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
TEXTBOOK = os.path.join(REPO_ROOT, "textbook")
DATA_DIR = os.path.join(REPO_ROOT, "Textbook_embedding_Yaxi", "data")
OUTPUTS = os.path.join(REPO_ROOT, "Textbook_embedding_Yaxi", "outputs")

# turn methods on/off here. "baseline" alone works with nothing else enabled.
ACTIVE_METHODS = ["baseline", "advanced", "concept_map"]

# let the LLM ignore retrieved context if it's irrelevant, and self-report
# whether it used the context or its own knowledge
ALLOW_CONTEXT_OPTIONAL = True
TAG_ANSWER_SOURCE = True

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"

RELEVANCE_THRESHOLD = 0.45

# prefer the page-tracked index, fall back to the original
def _pick(a, b):
    return a if os.path.exists(a) else b

INDEX_PATH = _pick(os.path.join(TEXTBOOK, "biomaterials_index_v2.faiss"),
                   os.path.join(TEXTBOOK, "biomaterials_index.faiss"))
METADATA_PATH = _pick(os.path.join(TEXTBOOK, "biomaterials_metadata_v2.csv"),
                      os.path.join(TEXTBOOK, "biomaterials_metadata.csv"))

SCQ_BANK = os.path.join(DATA_DIR, "scq_bank.json")
OPEN_BANK = next((p for p in [
    os.path.join(DATA_DIR, "20_open_ended_reference_bank.json"),
    os.path.join(REPO_ROOT, "20_open_ended_reference_bank.json"),
] if os.path.exists(p)), None)

METHODS = {
    "baseline":    {"retriever": "dense", "k": 5, "rerank": False},
    "advanced":    {"retriever": "dense", "k_initial": 20, "k_final": 5, "rerank": True},
    "concept_map": {"retriever": "graph", "k_initial": 20, "k_final": 5,
                    "rerank": True, "hops": 1, "max_expand": 15},
}
