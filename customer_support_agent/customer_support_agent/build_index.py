"""Optional: build an authentic FAISS index from chunks.json with MiniLM embeddings."""
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
p=Path(__file__).parent
chunks=json.loads((p/'chunks.json').read_text(encoding='utf-8'))
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
emb=model.encode([x['text'] for x in chunks],normalize_embeddings=True,convert_to_numpy=True).astype('float32')
index=faiss.IndexFlatIP(384)
index.add(emb)
faiss.write_index(index,str(p/'faiss.index'))
print(f'Built {p / "faiss.index"} with {index.ntotal} vectors')
