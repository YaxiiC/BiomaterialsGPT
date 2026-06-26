import numpy as np

import config
import llm


def retrieval_ok(chunks, reference_text):
    # keyword overlap check, used for failure-mode tagging
    kws = [w for w in reference_text.lower().split()
           if len(w) > 5 and w.isalpha()][:6]
    combined = " ".join(c["text"].lower() for c in chunks)
    return sum(1 for k in kws if k in combined) >= 2


def grade_coverage(question, answer, points):
    # checklist grading against the reference key points. returns None if the
    # grading call fails so the caller can skip rather than score a zero.
    if not answer:
        return None
    numbered = "\n".join(f"{i+1}. {p}" for i, p in enumerate(points))
    prompt = (
        "Grade the answer against the checklist. For each point decide if the "
        "answer covers it.\n\n"
        f"QUESTION: {question}\n\nANSWER:\n{answer}\n\nCHECKLIST:\n{numbered}\n\n"
        'For each point output one line: "<number>: YES" or "<number>: NO". '
        "Output only those lines."
    )
    resp = llm.call(prompt)
    if not resp:
        return None
    import re
    covered, detail = 0, []
    for i in range(len(points)):
        m = re.search(rf"{i+1}\s*:\s*(YES|NO)", resp, re.IGNORECASE)
        hit = bool(m and m.group(1).upper() == "YES")
        covered += int(hit)
        detail.append("Y" if hit else "N")
    return covered, len(points), "".join(detail)


def relevance_metrics(corpus, chunks, gold_text):
    # llm-free: similarity of each retrieved chunk to the gold answer
    from retrieval import get_embed
    gold_emb = get_embed().encode([gold_text], normalize_embeddings=True)[0]
    rels = [float(np.dot(corpus.chunk_embs[c["idx"]], gold_emb)) for c in chunks]
    flags = [r >= config.RELEVANCE_THRESHOLD for r in rels]
    hit1 = int(any(flags[:1]))
    hit3 = int(any(flags[:3]))
    hit5 = int(any(flags[:5]))
    mrr = 0.0
    for rank, f in enumerate(flags, start=1):
        if f:
            mrr = 1.0 / rank
            break
    return {"hit1": hit1, "hit3": hit3, "hit5": hit5, "mrr": mrr,
            "top_rel": max(rels) if rels else 0.0}
