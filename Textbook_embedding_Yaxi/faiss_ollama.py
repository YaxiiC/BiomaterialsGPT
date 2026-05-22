import ollama

from paths import fetch_rag_context

query = input("❓ Your question: ")
context = fetch_rag_context(query)

prompt = f"""You are a biomaterials assistant.

CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": prompt}
])

print("\n🧠", response['message']['content'])