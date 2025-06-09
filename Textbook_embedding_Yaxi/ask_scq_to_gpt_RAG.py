import re
import json
import csv
import subprocess
import os
import time
import unicodedata
from openai import OpenAI
import sys
import io

# Force stdout/stderr encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
os.environ["PYTHONIOENCODING"] = "utf-8"

def clean_text(text: str) -> str:
    # 1) Normalize to NFKC
    text = unicodedata.normalize("NFKC", text)
    # 2) Replace any remaining curly quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # 3) Strip out any character above U+007F
    return text.encode("ascii", "ignore").decode("ascii")

# Initialize OpenAI client
client = OpenAI()  # API key should be set via OPENAI_API_KEY env variable

def normalize_unicode(text):
    """Normalize Unicode text to avoid smart quote and encoding issues."""
    return unicodedata.normalize("NFKC", text)

def fetch_context(query):
    query = clean_text(query)
    try:
        results = subprocess.check_output(
            ["python", "/home/yaxi/biomaterialsGPT/search_faiss.py", query],
            text=True,
            encoding='utf-8',  # <<< enforce UTF-8 decoding here
            errors='replace' 
        )
        chunks = [line for line in results.strip().split("\n") if line]
        return clean_text("\n\n".join(chunks))
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fetching context: {e}")
        return ""

def ask_llm(question_with_opts: str, context: str, retries: int = 3) -> str:
    # Clean both question and context one last time
    question_with_opts = clean_text(question_with_opts)
    context = clean_text(context)

    prompt = f"""You are a biomaterials assistant.

CONTEXT:
{context}

QUESTION:
{question_with_opts}

Please provide:
The best single-choice option (a, b, c, or d). You must choose one of the options.
No explanation needed.

Format:
1. <option> (a, b, c, or d, without any other text)
"""
    # Clean the final prompt to strip any accidental non-ASCII
    prompt = clean_text(prompt)

    # (Optional) DEBUG: check if there’s any non-ASCII left
    nonascii = [ord(c) for c in prompt if ord(c) > 127]
    if nonascii:
        print("⚠️ WARNING: prompt still has non-ASCII codepoints:", nonascii)

    print("\n🧾 Final prompt being sent to GPT-4o:\n" + "-" * 40)
    print(prompt)
    print("-" * 40)

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system",  "content": clean_text("You are a biomaterials assistant.")},
                    {"role": "user",    "content": prompt}
                ],
                temperature=0
            )
            # The API will return a string in response.choices[0].message.content
            return response.choices[0].message.content

        except Exception as e:
            print(f"⚠️ GPT-4o call failed (attempt {attempt+1}): {e}")
            time.sleep(2)

    # If all retries fail, return an empty string
    return ""

def parse_response(resp_text):
    lines = [ln.strip() for ln in resp_text.splitlines() if ln.strip()]
    option = ""
    explanation = ""

    for ln in lines:
        if re.match(r"^1\.\s*([a-dA-D])\s*$", ln):
            option = ln.split(".")[1].strip().lower()
        elif re.match(r"^[a-dA-D]\s*$", ln):
            option = ln.strip().lower()
        elif re.search(r"\b[a-dA-D]\b", ln):
            match = re.search(r"\b([a-dA-D])\b", ln)
            option = match.group(1).lower()

    if option not in {"a", "b", "c", "d"}:
        option = ""
    return option, explanation

def main():
    with open("scq_bank.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open("scq_with_gpt_withRAG.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "id", "section", "number", "question",
            "correct_answer", "predicted_option", "explanation"
        ])
        writer.writeheader()

        for item in questions:
            q_text = clean_text(item["question"])
            opts = item["options"]
            opts_lines = "\n".join(f"{letter}) {clean_text(text)}"
                                for letter, text in sorted(opts.items()))
            q_block = f"{q_text}\n{opts_lines}"

            print(f"🔍 Received query: '{q_text}'")

            ctx = fetch_context(q_text)
            if not ctx.strip():
                print(f"⚠️ No context found for Q{item['number']}")

            
            resp = ask_llm(q_block, ctx)
            pred_opt, exp = parse_response(resp)

            if not pred_opt:
                print(f"⚠️ Warning: No valid prediction for Q{item['number']}")
                pred_opt = "?"

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

    print("✅ Saved results to scq_with_gpt_withRAG.csv")

if __name__ == "__main__":
    main()