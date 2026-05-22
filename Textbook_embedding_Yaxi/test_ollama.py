import ollama

query = "What are the mechanical properties of collagen?"
context = "Collagen-based biomaterials provide high tensile strength and support cellular adhesion."

prompt = f"""CONTEXT:
{context}

QUESTION:
{query}

ANSWER:"""

response = ollama.chat(model="llama3", messages=[
    {"role": "user", "content": prompt}
])
print(response['message']['content'])