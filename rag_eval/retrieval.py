import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

import config
import graph

_embed = None
_reranker = None


def get_embed():
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(config.EMBED_MODEL)
    return _embed


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker


class Corpus:
    def __init__(self):
        self.index = faiss.read_index(config.INDEX_PATH)
        self.meta = pd.read_csv(config.METADATA_PATH)
        self.texts = [str(t).replace("\n", " ") for t in self.meta["text"]]
        self.neighbours = None
        # precompute normalized chunk embeddings for relevance scoring
        self.chunk_embs = get_embed().encode(
            self.texts, normalize_embeddings=True, show_progress_bar=False)

    def ensure_graph(self):
        if self.neighbours is None:
            self.neighbours = graph.build(self.texts)

    def page_of(self, idx):
        if "page" in self.meta.columns:
            return self.meta.iloc[idx].get("page", "?")
        return "?"


def _dense_idxs(corpus, query, k):
    vec = get_embed().encode([query])
    _, idxs = corpus.index.search(np.array(vec).astype("float32"), k)
    return [int(i) for i in idxs[0]]


def _rerank(corpus, query, cand_idxs, k_final):
    pairs = [(query, corpus.texts[i]) for i in cand_idxs]
    scores = get_reranker().predict(pairs)
    order = np.argsort(scores)[::-1]
    return [cand_idxs[i] for i in order[:k_final]]


def retrieve(corpus, query, method):
    cfg = config.METHODS[method]
    kind = cfg["retriever"]

    if kind == "dense" and not cfg.get("rerank"):
        idxs = _dense_idxs(corpus, query, cfg["k"])
    elif kind == "dense":
        idxs = _rerank(corpus, query, _dense_idxs(corpus, query, cfg["k_initial"]),
                       cfg["k_final"])
    elif kind == "graph":
        corpus.ensure_graph()
        seeds = _dense_idxs(corpus, query, cfg["k_initial"])
        expanded = graph.expand(seeds, corpus.neighbours,
                                hops=cfg["hops"], max_expand=cfg["max_expand"])
        idxs = _rerank(corpus, query, expanded, cfg["k_final"])
    else:
        raise ValueError(kind)

    return [{"idx": i, "page": corpus.page_of(i), "text": corpus.texts[i]}
            for i in idxs]


def context_str(chunks, limit=600):
    return "\n\n".join(c["text"][:limit] for c in chunks)
