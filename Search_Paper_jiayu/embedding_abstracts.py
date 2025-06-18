# 从CSV文件中读取abstract并embedding，完成后存faiss库 (使用BCEmbedding)
import os
import pandas as pd
import glob
import faiss
import numpy as np
from typing import List
import time
from sentence_transformers import SentenceTransformer

# === CONFIG ===
CSV_DIR = "/remote-home/jiayuguo/RAG-search/results/"  # CSV文件目录
CHUNK_SIZE = 1000  # abstract通常较短，可以设置smaller chunk size
FAISS_INDEX_PATH = "/remote-home/jiayuguo/RAG-search/abstracts_index.faiss"
METADATA_PATH = "/remote-home/jiayuguo/RAG-search/abstracts_metadata.csv"
BATCH_SIZE = 10  # 批处理大小

print("🔧 Configuration loaded.")
print(f"CSV directory: {CSV_DIR}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Batch size: {BATCH_SIZE}\n")

# === INIT EMBEDDING MODEL ===
model = SentenceTransformer("maidalun1020/bce-embedding-base_v1")
embedding_dim = model.get_sentence_embedding_dimension()
print(f"✅ BCEmbedding model loaded. Dimension: {embedding_dim}")

# === FUNCTIONS ===
def extract_abstracts_from_csv(csv_path: str) -> List[dict]:
    try:
        df = pd.read_csv(csv_path)
        abstracts_data = []

        for idx, row in df.iterrows():
            abstract = str(row.get('Abstract', '')).strip()
            if abstract and abstract != 'nan' and len(abstract) > 50:
                abstracts_data.append({
                    'pmid': row.get('PMID', ''),
                    'title': row.get('Title', ''),
                    'abstract': abstract,
                    'journal': row.get('Journal', ''),
                    'date': row.get('Date', ''),
                    'source_file': os.path.basename(csv_path)
                })

        print(f"✅ 从 {csv_path} 提取了 {len(abstracts_data)} 个有效abstract")
        return abstracts_data

    except Exception as e:
        print(f"❌ Error reading {csv_path}: {e}")
        return []

def chunk_text(text: str, max_length: int = 1000) -> List[str]:
    if len(text) <= max_length:
        return [text]
    chunks = []
    while len(text) > max_length:
        split_idx = text.rfind('.', 0, max_length)
        if split_idx == -1:
            split_idx = max_length
        chunks.append(text[:split_idx+1].strip())
        text = text[split_idx+1:].strip()
    if text:
        chunks.append(text)
    return chunks

# === FIND ALL CSV FILES ===
print("\n📄 Finding CSV files...")
csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
print(f"✅ Found {len(csv_files)} CSV files.")

if not csv_files:
    print("❌ No CSV files found in results directory!")
    exit(1)

# === INIT FAISS ===
index = faiss.IndexFlatL2(embedding_dim)
metadata = []

# === PROCESS CSV FILES ===
total_abstracts = 0
for csv_file in csv_files:
    print(f"\n📄 Processing CSV: {csv_file}")
    abstracts_data = extract_abstracts_from_csv(csv_file)

    if not abstracts_data:
        print("⚠️ No valid abstracts found. Skipping.")
        continue

    texts_to_embed = []
    current_metadata = []

    for data in abstracts_data:
        abstract = data['abstract']
        chunks = chunk_text(abstract, max_length=CHUNK_SIZE)

        for i, chunk in enumerate(chunks):
            texts_to_embed.append(chunk)
            current_metadata.append({
                'pmid': data['pmid'],
                'title': data['title'],
                'text': chunk,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'journal': data['journal'],
                'date': data['date'],
                'source_file': data['source_file']
            })

    if not texts_to_embed:
        continue

    print(f"✂️  Created {len(texts_to_embed)} text chunks for embedding.")

    try:
        all_embeddings = []
        for i in range(0, len(texts_to_embed), BATCH_SIZE):
            batch_texts = texts_to_embed[i:i+BATCH_SIZE]
            batch_num = i//BATCH_SIZE + 1
            total_batches = (len(texts_to_embed) + BATCH_SIZE - 1)//BATCH_SIZE
            print(f"📦 Processing batch {batch_num}/{total_batches}")

            batch_embeddings = model.encode(batch_texts)
            all_embeddings.extend(batch_embeddings.tolist())
            print(f"✅ Batch {batch_num} completed. Total embeddings so far: {len(all_embeddings)}")

        if len(all_embeddings) != len(texts_to_embed):
            print(f"⚠️ Warning: Expected {len(texts_to_embed)} embeddings, got {len(all_embeddings)}")

        if all_embeddings:
            embeddings_array = np.array(all_embeddings).astype("float32")
            index.add(embeddings_array)
            metadata.extend(current_metadata[:len(all_embeddings)])
            total_abstracts += len(abstracts_data)
            print(f"📌 Added {len(all_embeddings)} embeddings to FAISS.")

    except Exception as e:
        print(f"❌ Embedding failed for {csv_file}: {e}")
        continue

print(f"\n🎉 Total processed: {total_abstracts} abstracts from {len(csv_files)} files")
print(f"📊 Total chunks in FAISS index: {len(metadata)}")

# === SAVE FAISS INDEX AND METADATA ===
print("\n💾 Saving FAISS index and metadata...")
faiss.write_index(index, FAISS_INDEX_PATH)

metadata_df = pd.DataFrame(metadata)
metadata_df.to_csv(
    METADATA_PATH,
    index=False,
    escapechar='\\',
    quoting=1
)

print(f"\n✅ FAISS index saved to: {FAISS_INDEX_PATH}")
print(f"✅ Metadata saved to: {METADATA_PATH}")
print(f"📈 Index contains {index.ntotal} vectors")
