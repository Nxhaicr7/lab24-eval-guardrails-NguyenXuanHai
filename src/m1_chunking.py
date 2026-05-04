"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

LlamaParse được dùng để OCR các file PDF trong data/.

Test: pytest tests/test_m1.py
"""

import os
import sys
import glob
import re
import asyncio
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: Optional[str] = None


# ─── Document Loading with LlamaParse OCR ────────────────


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """
    Load documents from data/:
    - .md / .txt files: read directly
    - .pdf files: OCR via LlamaParse API
    Returns list of {"text": str, "metadata": {"source": filename}}
    """
    docs = []

    # Load plain text / markdown files
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md")) +
                     glob.glob(os.path.join(data_dir, "*.txt"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({
                "text": f.read(),
                "metadata": {"source": os.path.basename(fp)}
            })

    # Load PDF files via LlamaParse OCR
    pdf_files = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    if pdf_files:
        docs.extend(_parse_pdfs_with_llamaparse(pdf_files))

    return docs


def _parse_pdfs_with_llamaparse(pdf_paths: list[str]) -> list[dict]:
    """
    Use LlamaParse to OCR PDF files and return extracted text as documents.
    Reads LLAMA_CLOUD_API_KEY from environment (or config).
    """
    try:
        from llama_parse import LlamaParse
    except ImportError:
        print("  [WARNING] llama-parse not installed. Skipping PDF OCR.")
        print("  Run: pip install llama-parse")
        return []

    # API key: prefer env var, fall back to hardcoded for lab use
    api_key = os.getenv("LLAMA_CLOUD_API_KEY", "llx-DdhsSTV7keHghBiQXlWmwIoXYdFazkJDpPtjeDHVUSapbeIZ")
    if not api_key:
        print("  [WARNING] LLAMA_CLOUD_API_KEY not set. Skipping PDF OCR.")
        return []

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",   # Get structured markdown output
        language="vi",            # Vietnamese documents
        verbose=False,
    )

    docs = []
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f"  [LlamaParse] OCR-ing: {filename} ...")
        try:
            # LlamaParse returns a list of Document objects
            documents = parser.load_data(pdf_path)
            full_text = "\n\n".join(doc.text for doc in documents if doc.text.strip())
            if full_text.strip():
                docs.append({
                    "text": full_text,
                    "metadata": {
                        "source": filename,
                        "ocr": "llamaparse",
                        "pages": len(documents),
                    }
                })
                print(f"  [LlamaParse] Done: {filename} → {len(full_text)} chars, {len(documents)} pages")
            else:
                print(f"  [LlamaParse] WARNING: No text extracted from {filename}")
        except Exception as e:
            print(f"  [LlamaParse] ERROR on {filename}: {e}")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "basic"}
            ))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            metadata={**metadata, "chunk_index": len(chunks), "strategy": "basic"}
        ))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.

    Args:
        text: Input text.
        threshold: Cosine similarity threshold. Dưới threshold → tách chunk mới.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects grouped by semantic similarity.
    """
    metadata = metadata or {}

    # 1. Split text into sentences (handle Vietnamese punctuation + newlines)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n\n', text) if s.strip()]
    if not sentences:
        return []

    # Need at least 2 sentences to compare; if only 1, return as single chunk
    if len(sentences) == 1:
        return [Chunk(
            text=sentences[0],
            metadata={**metadata, "chunk_index": 0, "strategy": "semantic"}
        )]

    # 2. Encode sentences with a lightweight multilingual model
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, good enough for similarity
        embeddings = model.encode(sentences, show_progress_bar=False)

        def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
            """Cosine similarity between two vectors."""
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # 3. Group sentences: start new chunk when similarity drops below threshold
        chunks: list[Chunk] = []
        current_group: list[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            sim = cosine_sim(embeddings[i - 1], embeddings[i])
            if sim < threshold:
                # Flush current group as a chunk
                chunks.append(Chunk(
                    text=" ".join(current_group),
                    metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
                ))
                current_group = []
            current_group.append(sentences[i])

        # Don't forget the last group
        if current_group:
            chunks.append(Chunk(
                text=" ".join(current_group),
                metadata={**metadata, "chunk_index": len(chunks), "strategy": "semantic"}
            ))

        return chunks

    except ImportError:
        # Fallback: no sentence-transformers → treat each paragraph as a chunk
        print("  [WARNING] sentence-transformers not available. Using paragraph fallback.")
        return [
            Chunk(
                text=s,
                metadata={**metadata, "chunk_index": i, "strategy": "semantic_fallback"}
            )
            for i, s in enumerate(sentences)
        ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Args:
        text: Input text.
        parent_size: Chars per parent chunk.
        child_size: Chars per child chunk.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    parents: list[Chunk] = []
    children: list[Chunk] = []

    # 1. Split text into parent chunks by accumulating paragraphs up to parent_size
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    current_text = ""

    for para in paragraphs:
        # If adding this paragraph would exceed parent_size, flush current parent
        if len(current_text) + len(para) > parent_size and current_text:
            p_index = len(parents)
            pid = f"parent_{p_index}"
            parents.append(Chunk(
                text=current_text.strip(),
                metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
            ))
            current_text = ""
        current_text += para + "\n\n"

    # Flush remaining text as last parent
    if current_text.strip():
        p_index = len(parents)
        pid = f"parent_{p_index}"
        parents.append(Chunk(
            text=current_text.strip(),
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid}
        ))

    # 2. Split each parent into children using a sliding window of child_size
    for parent in parents:
        pid = parent.metadata["parent_id"]
        parent_text = parent.text
        start = 0
        c_index = 0

        while start < len(parent_text):
            end = start + child_size
            child_text = parent_text[start:end].strip()
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={
                        **metadata,
                        "chunk_type": "child",
                        "child_index": c_index,
                    },
                    parent_id=pid,
                ))
                c_index += 1
            start = end  # non-overlapping; use start = start + child_size // 2 for overlap

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.

    Args:
        text: Markdown text.
        metadata: Metadata gắn vào mỗi chunk.

    Returns:
        List of Chunk objects, mỗi chunk = 1 section (header + content).
    """
    metadata = metadata or {}

    # 1. Split by markdown headers (H1–H3), keeping the delimiters
    sections = re.split(r'(^#{1,3}\s+.+$)', text, flags=re.MULTILINE)

    chunks: list[Chunk] = []
    current_header = ""
    current_content = ""

    for part in sections:
        if re.match(r'^#{1,3}\s+', part):
            # Flush previous section before starting a new one
            if current_content.strip():
                chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "section": current_header.strip(),
                        "strategy": "structure",
                        "chunk_index": len(chunks),
                    }
                ))
            current_header = part.strip()
            current_content = ""
        else:
            current_content += part

    # Don't forget the last section
    if current_content.strip():
        chunk_text = f"{current_header}\n{current_content}".strip() if current_header else current_content.strip()
        chunks.append(Chunk(
            text=chunk_text,
            metadata={
                **metadata,
                "section": current_header.strip(),
                "strategy": "structure",
                "chunk_index": len(chunks),
            }
        ))

    # Edge case: no headers found → treat entire text as one chunk
    if not chunks and text.strip():
        chunks.append(Chunk(
            text=text.strip(),
            metadata={**metadata, "section": "", "strategy": "structure", "chunk_index": 0}
        ))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all 4 strategies on documents and compare stats.

    Returns:
        {
            "basic":        {"num_chunks": int, "avg_length": float, "min_length": int, "max_length": int},
            "semantic":     {...},
            "hierarchical": {"num_parents": int, "num_children": int, "avg_child_length": float, ...},
            "structure":    {...},
        }
    """
    results: dict[str, dict] = {}

    all_basic: list[Chunk] = []
    all_semantic: list[Chunk] = []
    all_parents: list[Chunk] = []
    all_children: list[Chunk] = []
    all_structure: list[Chunk] = []

    for doc in documents:
        text = doc["text"]
        meta = doc.get("metadata", {})

        all_basic.extend(chunk_basic(text, metadata=meta))
        all_semantic.extend(chunk_semantic(text, metadata=meta))
        parents, children = chunk_hierarchical(text, metadata=meta)
        all_parents.extend(parents)
        all_children.extend(children)
        all_structure.extend(chunk_structure_aware(text, metadata=meta))

    def _stats(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"num_chunks": 0, "avg_length": 0.0, "min_length": 0, "max_length": 0}
        lengths = [len(c.text) for c in chunks]
        return {
            "num_chunks": len(chunks),
            "avg_length": round(sum(lengths) / len(lengths), 1),
            "min_length": min(lengths),
            "max_length": max(lengths),
        }

    results["basic"] = _stats(all_basic)
    results["semantic"] = _stats(all_semantic)
    results["hierarchical"] = {
        **_stats(all_children),
        "num_parents": len(all_parents),
        "num_children": len(all_children),
        "avg_parent_length": round(
            sum(len(p.text) for p in all_parents) / max(len(all_parents), 1), 1
        ),
    }
    results["structure"] = _stats(all_structure)

    # Print comparison table
    print(f"\n{'Strategy':<16} | {'Chunks':>7} | {'Avg Len':>8} | {'Min':>6} | {'Max':>6}")
    print("-" * 55)
    for name in ["basic", "semantic", "structure"]:
        s = results[name]
        print(f"{name:<16} | {s['num_chunks']:>7} | {s['avg_length']:>8.0f} | {s['min_length']:>6} | {s['max_length']:>6}")
    h = results["hierarchical"]
    print(f"{'hierarchical':<16} | {h['num_parents']:>3}p/{h['num_children']:>3}c | "
          f"{h['avg_length']:>8.0f} | {h['min_length']:>6} | {h['max_length']:>6}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    if docs:
        results = compare_strategies(docs)
        print("\nStrategy stats:")
        for name, stats in results.items():
            print(f"  {name}: {stats}")
    else:
        print("No documents found in data/. Add .md, .txt, or .pdf files.")
