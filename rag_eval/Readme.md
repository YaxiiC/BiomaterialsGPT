# rag_eval

evaluation comparing three retrieval methods for the
biomaterials RAG system. Any subset of methods can be run; baseline works
alone with no API key or anything else enabled.

## Methods

- `baseline`    dense retrieval, top 5
- `advanced`    dense top 20 -> cross-encoder rerank -> top 5
- `concept_map` dense seed -> term co-occurrence graph expansion -> rerank -> top 5

Toggle in `config.py` to `ACTIVE_METHODS`.

## Setup

```
pip install sentence-transformers faiss-cpu groq pandas numpy
$env:GROQ_API_KEY = "your-key"          
$env:PYTHONUTF8 = "1"

 line 2 can be changed and adapted to the llm of your choice
```

Index path is auto-detected: prefers `biomaterials_index_v2.faiss`, falls back
to `biomaterials_index.faiss`. The folder must sit in the repo root next to
`textbook/` and `Textbook_embedding_Yaxi/`.

## Run

```
python rag_eval/run_retrieval.py   # retrieval metrics only, no llm / no api key
python rag_eval/run_mcq.py         # mcq accuracy, needs scq_bank.json
python rag_eval/run_open.py        # open-ended key-point coverage
```

Outputs are written to `Textbook_embedding_Yaxi/outputs/`:
`eval_retrieval.csv`, `eval_mcq.csv`, `eval_open.csv`.

## Evaluation layers

The three runners measure different things on purpose:

- `run_retrieval.py` is the only llm-free comparison. It checks whether each
  method retrieves chunks matching the known answer (Hit@1/3/5, MRR). Model
  knowledge cannot contaminate it, so this is the fair method-vs-method test.
- `run_mcq.py` measures end-to-end accuracy and tags each failure as a
  retrieval failure (right chunk never retrieved) or generation failure
  (right chunk retrieved, llm still wrong).
- `run_open.py` scores open-ended answers by how many of each question's
  key_evaluation_points they cover.

## Config flags

- `ACTIVE_METHODS`         methods to run
- `ALLOW_CONTEXT_OPTIONAL` lets the llm ignore irrelevant context and answer
                           from its own knowledge
- `TAG_ANSWER_SOURCE`      llm self-reports SOURCE: CONTEXT or OWN, logged per
                           answer to show when retrieval was actually used
- `RELEVANCE_THRESHOLD`    similarity cutoff for a chunk to count as relevant
                           in run_retrieval.py (default 0.45)

## Notes

Failed llm calls retry 3x, then the question is skipped rather than scored
zero, so one network error does not distort the mean.