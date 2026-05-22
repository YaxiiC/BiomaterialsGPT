# BiomaterialsGPT

RAG-assisted evaluation of biomaterials question banks using textbook embeddings and local or API LLMs.

Related repo: [YaxiiC/BiomaterialsGPT](https://github.com/YaxiiC/BiomaterialsGPT) (includes `Search_Paper_jiayu` for PubMed abstract search).

## Repository layout

```
.
├── Textbook_embedding_Yaxi/   # Main scripts, question banks, experiment outputs
│   ├── data/                  # scq_bank.json, open-ended bank, prompts
│   └── outputs/               # Generated CSVs (gitignored)
├── textbook/                  # PDFs + FAISS index (built locally, not in git)
└── local/                     # Word drafts and personal files (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Add textbook PDFs under `textbook/` (see `textbook/README.md`).
2. Build the index: `python Textbook_embedding_Yaxi/textbook_embedding.py`
3. Install [Ollama](https://ollama.com/) and pull models used in scripts (e.g. `llama3`, `qwen3`).
4. For GPT scripts: `export OPENAI_API_KEY=...`

## Common commands

Run from repo root:

```bash
# Interactive RAG + Ollama
python Textbook_embedding_Yaxi/faiss_ollama.py

# Single-choice questions (Llama + RAG)
python Textbook_embedding_Yaxi/ask_scq_to_llama_RAG.py

# Open-ended questions
python Textbook_embedding_Yaxi/ask_openend_to_llama_RAG.py

# FAISS search only
python Textbook_embedding_Yaxi/search_faiss.py "What is a hydrogel?"
```

Convert Word question banks (source `.docx` files live in `local/`):

```bash
python Textbook_embedding_Yaxi/question_bank_convert_scq.py
python Textbook_embedding_Yaxi/question_bank_convert_open_ended.py
```

## Uploading to GitHub

This folder is structured to match `Textbook_embedding_Yaxi` on the upstream repo. After committing here, push to your fork or open a PR against [YaxiiC/BiomaterialsGPT](https://github.com/YaxiiC/BiomaterialsGPT).

Do not commit: PDFs, FAISS indices, metadata CSVs, or files under `local/`.
