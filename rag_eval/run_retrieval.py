import os
import sys
import io
import json
import csv

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

import config
import scoring
from retrieval import Corpus, retrieve


# retrieval-only quality, no llm. measures whether each method pulls chunks
# that match the known answer.
def main():
    if not config.OPEN_BANK:
        print("no reference bank found")
        return

    with open(config.OPEN_BANK, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"loaded {len(questions)} questions (threshold {config.RELEVANCE_THRESHOLD})")

    corpus = Corpus()
    methods = config.ACTIVE_METHODS
    agg = {m: {"hit1": 0, "hit3": 0, "hit5": 0, "mrr": 0.0, "top_rel": 0.0}
           for m in methods}
    n = len(questions)

    fields = ["id", "number", "question"]
    for m in methods:
        fields += [f"{m}_hit1", f"{m}_hit3", f"{m}_hit5", f"{m}_mrr", f"{m}_top_rel"]

    out_path = os.path.join(config.OUTPUTS, "eval_retrieval.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for item in questions:
            q = item["question"]
            gold = item["answer"]
            row = {"id": item["id"], "number": item["number"], "question": q}
            line = f"q{item['number']:>2}"

            for m in methods:
                chunks = retrieve(corpus, q, m)
                mt = scoring.relevance_metrics(corpus, chunks, gold)
                for key in ("hit1", "hit3", "hit5", "mrr", "top_rel"):
                    agg[m][key] += mt[key]
                row[f"{m}_hit1"] = mt["hit1"]
                row[f"{m}_hit3"] = mt["hit3"]
                row[f"{m}_hit5"] = mt["hit5"]
                row[f"{m}_mrr"] = round(mt["mrr"], 3)
                row[f"{m}_top_rel"] = round(mt["top_rel"], 3)
                line += f" | {m[:4]} h5={mt['hit5']} mrr={mt['mrr']:.2f}"

            w.writerow(row)
            print(line)

    print(f"\n{'='*60}\nretrieval quality (no llm)\n{'='*60}")
    print(f"{'method':<14}{'Hit@1':>8}{'Hit@3':>8}{'Hit@5':>8}{'MRR':>8}{'TopRel':>9}")
    for m in methods:
        a = agg[m]
        print(f"{m:<14}{a['hit1']/n*100:>7.0f}%{a['hit3']/n*100:>7.0f}%"
              f"{a['hit5']/n*100:>7.0f}%{a['mrr']/n:>8.3f}{a['top_rel']/n:>9.3f}")

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
