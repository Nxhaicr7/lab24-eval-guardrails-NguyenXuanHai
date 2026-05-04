"""
Run all 4 chunking strategies on the data/ documents and save results to chunks/.
"""
import json
import os
from src.m1_chunking import (
    load_documents, chunk_basic, chunk_semantic,
    chunk_hierarchical, chunk_structure_aware, compare_strategies
)

OUT_DIR = "chunks"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load documents ────────────────────────────────────────
docs = load_documents()
print(f"\nLoaded {len(docs)} document(s):")
for d in docs:
    print(f"  - {d['metadata']['source']} ({len(d['text']):,} chars)")

if not docs:
    print("No documents found in data/. Run ocr_pdfs.py first.")
    exit(1)

# ── Run all strategies ────────────────────────────────────
all_chunks = {"basic": [], "semantic": [], "hierarchical_parents": [],
              "hierarchical_children": [], "structure": []}

for doc in docs:
    text = doc["text"]
    meta = doc["metadata"]

    all_chunks["basic"].extend(chunk_basic(text, metadata=meta))

    print(f"\n  [semantic] Embedding sentences from {meta['source']} via OpenAI...")
    all_chunks["semantic"].extend(chunk_semantic(text, metadata=meta))

    parents, children = chunk_hierarchical(text, metadata=meta)
    all_chunks["hierarchical_parents"].extend(parents)
    all_chunks["hierarchical_children"].extend(children)

    all_chunks["structure"].extend(chunk_structure_aware(text, metadata=meta))

# ── Save to JSON ──────────────────────────────────────────
def chunks_to_json(chunks):
    return [{"text": c.text, "metadata": c.metadata, "parent_id": c.parent_id}
            for c in chunks]

for name, chunks in all_chunks.items():
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(chunks_to_json(chunks), f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(chunks):>4} chunks → {path}")

# ── Print comparison table ────────────────────────────────
print("\n")
compare_strategies(docs)

print(f"\nAll chunks saved to ./{OUT_DIR}/")
