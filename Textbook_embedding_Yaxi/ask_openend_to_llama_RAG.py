import json
import csv
import subprocess

def fetch_context(query):
    results = subprocess.check_output(
        ["python", "/home/yaxi/biomaterialsGPT/search_faiss.py", query],
        text=True
    )
    chunks = [line for line in results.strip().split("\n") if line]
    return "\n\n".join(chunks)

def ask_llm_open_ended(question, context):
    prompt = f"""You are a knowledgeable assistant in the field of biomaterials.

CONTEXT (if relevant):
{context}

QUESTION:
{question}

Please provide a clear and informative open-ended answer mainly based on the context. Please limit your answer to 400 words.
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
    with open("question_bank_open_ended.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open("open_qa_with_llm_withRAG.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "unit", "part", "number", "question", "llm_answer"
        ])
        writer.writeheader()

        for item in questions:
            q_text = item["question"]

            # 1) RAG context
            ctx = fetch_context(q_text)

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

    print("✅ Saved results to open_qa_with_llm_withRAG.csv")

if __name__ == "__main__":
    main()