import os
import sys
import io
import json
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
import llm
import prompts
import parsing
import scoring
from retrieval import Corpus, retrieve, context_str


def main():
    with open(config.SCQ_BANK, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"loaded {len(questions)} mcq questions")
    print(f"active methods: {config.ACTIVE_METHODS}")

    corpus = Corpus()
    print(f"index: {corpus.index.ntotal} vectors\n")

    methods = config.ACTIVE_METHODS
    fields = ["id", "number", "section", "question", "correct"]
    for m in methods:
        fields += [f"{m}_pred", f"{m}_correct", f"{m}_source",
                   f"{m}_failure", f"{m}_chunks"]

    scores = {m: 0 for m in methods}
    total = 0
    out_path = os.path.join(config.OUTPUTS, "eval_mcq.csv")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for item in questions:
            q = item["question"]
            options = item["options"]
            correct = item["answer"].lower()
            expl = item.get("explanation") or ""
            total += 1

            row = {"id": item["id"], "number": item["number"],
                   "section": item.get("section", ""), "question": q,
                   "correct": correct}
            line = f"q{item['number']:>3} correct={correct}"

            for m in methods:
                chunks = retrieve(corpus, q, m)
                resp = llm.call(prompts.mcq(q, options, context_str(chunks)))
                pred = parsing.extract_letter(resp, options)
                source = parsing.extract_source(resp)
                ok = scoring.retrieval_ok(chunks, expl)
                is_correct = pred == correct
                if is_correct:
                    scores[m] += 1
                failure = ("correct" if is_correct
                           else "retrieval" if not ok else "generation")
                row[f"{m}_pred"] = pred
                row[f"{m}_correct"] = is_correct
                row[f"{m}_source"] = source
                row[f"{m}_failure"] = failure
                row[f"{m}_chunks"] = " | ".join(
                    f"p{c['page']}:{c['text'][:80]}" for c in chunks)
                line += f" | {m[:4]}={pred}{'OK' if is_correct else 'X'}"

            w.writerow(row)
            print(line)

    print(f"\n{'='*56}\nmcq accuracy ({total} questions)\n{'='*56}")
    for m in methods:
        print(f"  {m:<14} {scores[m]}/{total} = {scores[m]/total*100:.1f}%")

    df = pd.read_csv(out_path)
    if config.TAG_ANSWER_SOURCE:
        print("\nanswer source (self-reported):")
        for m in methods:
            counts = df[f"{m}_source"].value_counts().to_dict()
            print(f"  {m}: {counts}")

    print("\nfailure modes:")
    for m in methods:
        counts = df[~df[f"{m}_correct"]][f"{m}_failure"].value_counts().to_dict()
        print(f"  {m}: {counts}")

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
