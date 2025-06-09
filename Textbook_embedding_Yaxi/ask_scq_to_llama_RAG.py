import re
import json
import csv
import uuid
import subprocess

def fetch_context(query):
    results = subprocess.check_output(
        ["python", "/home/yaxi/biomaterialsGPT/search_faiss.py", query],
        text=True
    )
    chunks = [line for line in results.strip().split("\n") if line]
    return "\n\n".join(chunks)

def ask_llm(question_with_opts, context):
    # Build the prompt string
    prompt = f"""You are a biomaterials assistant.

CONTEXT:
{context}

QUESTION:
{question_with_opts}

Please provide:
The best single-choice option (a, b, c, or d). You must choose one of the options.
no explaination needed.

Format:
1. <option> (a, b, c, or d, without any other text)
"""
    # Run `ollama run llama3`, feeding prompt on stdin
    try:
        proc = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        print("Error calling LLM:", e)
        return ""

def parse_response(resp_text):
    lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
    option = ""
    explanation = ""
    
    for ln in lines:
        # Robust pattern matching
        if re.match(r"^1\.\s*([a-dA-D])\s*$", ln):
            option = ln.split(".")[1].strip().lower()
        elif re.match(r"^[a-dA-D]\s*$", ln):
            option = ln.strip().lower()
        elif re.search(r"\b[a-dA-D]\b", ln):
            match = re.search(r"\b([a-dA-D])\b", ln)
            option = match.group(1).lower()

    if option not in {"a", "b", "c", "d"}:
        option = ""  # fallback if not found
    return option, explanation


def main():
    with open("scq_bank.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open("scq_with_llm_withRAG.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "section", "number", "question",
            "correct_answer", "predicted_option", "explanation"
        ])
        writer.writeheader()

        for item in questions:
            q_text = item["question"]
            opts = item["options"]
            opts_lines = "\n".join(f"{letter}) {text}" for letter, text in sorted(opts.items()))
            q_block = f"{q_text}\n{opts_lines}"

            # 1) RAG context
            ctx = fetch_context(q_text)

            # 2) Query the model
            resp = ask_llm(q_block, ctx)

            # 3) Extract predicted option
            pred_opt, exp = parse_response(resp)

            if not pred_opt:
                print(f"⚠️ Warning: No valid prediction for Q{item['number']}")
                pred_opt = "?"

            # 4) Write CSV row
            writer.writerow({
                "id":               item["id"],
                "section":          item.get("section", ""),
                "number":           item.get("number", ""),
                "question":         q_text,
                "correct_answer":   item.get("answer", ""),
                "predicted_option": pred_opt,
                "explanation":      exp
            })

            print(f"Q{item['number']} → predicted {pred_opt}, actual {item.get('answer')}")

    print("Saved results to scq_with_llm_withRAG.csv")

if __name__ == "__main__":
    main()