import sys
#print("✅ FAISS subprocess started")

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    import pandas as pd
except Exception as e:
    print("❌ Import error:", e)
    sys.exit(1)

#print("✅ All modules imported")

query = " ".join(sys.argv[1:]).strip()
print(f"🔍 Received query: {query!r}", file=sys.stderr)

if not query:
    print("❌ No query provided.", file=sys.stderr)
    sys.exit(1)


try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("biomaterials_index.faiss")
    metadata = pd.read_csv("biomaterials_metadata.csv")

    embedding = model.encode([query])
    D, I = index.search(np.array(embedding).astype("float32"), 5)

    # output top-5 chunks
    for idx in I[0]:
        print(metadata.iloc[idx]["text"].replace("\n", " "))
except Exception as e:
    print("❌ Runtime error:", e, file=sys.stderr)
    sys.exit(1)