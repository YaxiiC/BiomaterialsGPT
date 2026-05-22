import json
import csv
import subprocess

from paths import OPEN_ENDED_BANK_JSON, OUTPUTS_DIR, fetch_rag_context

def ask_llm_open_ended(question, context):
    prompt = f"""You are a knowledgeable assistant in the field of biomaterials.

QUESTION:
{question}

Answer the question clearly and directly, with no extra commentary, internal thoughts, or preambles. Please limit your answer to 400 words.
"""
    try:
        proc = subprocess.run(
            ["ollama", "run", "llama3"],
            input=prompt,
            text=True,
            capture_output=True,
            check=True
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("Error calling LLM:", e)
        return ""

def main():
    with open(OPEN_ENDED_BANK_JSON, "r", encoding="utf-8") as f:
        questions = json.load(f)

    out_path = OUTPUTS_DIR / "open_qa_with_llm_withoutRAG.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "unit", "part", "number", "question", "llm_answer"
        ])
        writer.writeheader()

        for item in questions:
            q_text = item["question"]

            # 1) RAG context
            ctx = fetch_rag_context(q_text)

            # 2) Query the model
            llm_answer = ask_llm_open_ended(q_text, ctx)

            # 3) Write to CSV
            writer.writerow({
                "id": item["id"],
                "unit": item.get("unit", ""),
                "part": item.get("part", ""),
                "number": item.get("number", ""),
                "question": q_text,
                "llm_answer": llm_answer
            })

            print(f"Q{item['number']} → answer written")

    print(f"✅ Saved results to {out_path}")

if __name__ == "__main__":
    main()