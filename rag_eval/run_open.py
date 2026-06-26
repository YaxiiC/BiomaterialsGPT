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
    if not config.OPEN_BANK:
        print("no open-ended reference bank found. put "
              "20_open_ended_reference_bank.json in Textbook_embedding_Yaxi/data/")
        return

    with open(config.OPEN_BANK, encoding="utf-8") as f:
        questions = json.load(f)
    print(f"loaded {len(questions)} open-ended questions")
    print(f"active methods: {config.ACTIVE_METHODS}")

    corpus = Corpus()
    print(f"index: {corpus.index.ntotal} vectors\n")

    methods = config.ACTIVE_METHODS
    fields = ["id", "number", "group", "question", "reference"]
    for m in methods:
        fields += [f"{m}_answer", f"{m}_coverage", f"{m}_points", f"{m}_source"]

    sums = {m: 0.0 for m in methods}
    counts = {m: 0 for m in methods}

    out_path = os.path.join(config.OUTPUTS, "eval_open.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for item in questions:
            q = item["question"]
            ref = item["answer"]
            points = item["key_evaluation_points"]

            row = {"id": item["id"], "number": item["number"],
                   "group": item.get("group", ""), "question": q,
                   "reference": ref}
            line = f"q{item['number']:>2} [{item.get('group','')[:18]:<18}]"

            for m in methods:
                chunks = retrieve(corpus, q, m)
                resp = llm.call(prompts.open_ended(q, context_str(chunks)))
                source = parsing.extract_source(resp)
                answer = parsing.strip_tags(resp)
                graded = scoring.grade_coverage(q, answer, points)

                if graded is None:
                    row[f"{m}_answer"] = answer
                    row[f"{m}_coverage"] = "SKIP"
                    row[f"{m}_points"] = ""
                    row[f"{m}_source"] = source
                    line += f" | {m[:4]}=SKIP"
                else:
                    cov, tot, detail = graded
                    sums[m] += cov / tot if tot else 0
                    counts[m] += 1
                    row[f"{m}_answer"] = answer
                    row[f"{m}_coverage"] = f"{cov}/{tot}"
                    row[f"{m}_points"] = detail
                    row[f"{m}_source"] = source
                    line += f" | {m[:4]}={cov}/{tot}"

            w.writerow(row)
            print(line)

    print(f"\n{'='*56}\nmean key-point coverage\n{'='*56}")
    for m in methods:
        c = counts[m]
        pct = sums[m] / c * 100 if c else 0
        print(f"  {m:<14} {pct:.1f}%  (scored on {c})")

    df = pd.read_csv(out_path)
    print("\nby group:")
    for grp in df["group"].unique():
        sub = df[df["group"] == grp]
        line = f"  {str(grp)[:28]:<28}"
        for m in methods:
            valid = sub[sub[f"{m}_coverage"] != "SKIP"][f"{m}_coverage"]
            if len(valid):
                fr = valid.apply(lambda s: int(s.split("/")[0]) / int(s.split("/")[1])).mean()
                line += f" {m[:4]}={fr*100:.0f}%"
            else:
                line += f" {m[:4]}=--"
        print(line)

    if config.TAG_ANSWER_SOURCE:
        print("\nanswer source (self-reported):")
        for m in methods:
            print(f"  {m}: {df[f'{m}_source'].value_counts().to_dict()}")

    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
