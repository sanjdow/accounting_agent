"""ChromaDB-backed policy vector store with incremental ingestion."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, POLICY_DIR, RAG_TOP_K


_CLIENT = None
_COLLECTION = None
_COLLECTION_NAME = "policies"


def _client():
    global _CLIENT
    if _CLIENT is None:
        VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
        _CLIENT = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    return _CLIENT


def _collection():
    global _COLLECTION
    if _COLLECTION is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        _COLLECTION = _client().get_or_create_collection(
            name=_COLLECTION_NAME, embedding_function=ef
        )
    return _COLLECTION


def _chunk(text: str, size: int = 600, overlap: int = 80) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks


def _doc_id(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}:{idx}:{h}"


def ingest_policies(extra_files: List[Path] | None = None) -> Dict[str, int]:
    """Ingest every .txt and .md file in POLICY_DIR (plus any extras).

    Returns a dict of {filename: chunks_added}. Idempotent via content hash.
    """
    col = _collection()
    existing = set(col.get()["ids"])
    added: Dict[str, int] = {}

    files: List[Path] = []
    if POLICY_DIR.exists():
        files.extend(sorted(POLICY_DIR.glob("*.txt")))
        files.extend(sorted(POLICY_DIR.glob("*.md")))
    if extra_files:
        files.extend(extra_files)

    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        chunks = _chunk(text)
        ids, docs, metas = [], [], []
        for i, c in enumerate(chunks):
            _id = _doc_id(f.stem, i, c)
            if _id in existing:
                continue
            ids.append(_id)
            docs.append(c)
            metas.append({"source": f.name, "chunk": i})
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
            added[f.name] = len(ids)
    return added


def query_policies(query: str, k: int = RAG_TOP_K) -> Dict:
    col = _collection()
    res = col.query(query_texts=[query], n_results=k)
    # Return flat structure with confidence (1 - distance)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res.get("distances") else [1.0] * len(docs)
    hits = []
    for d, m, dist in zip(docs, metas, dists):
        hits.append({
            "text": d,
            "source": m.get("source", "unknown"),
            "chunk": m.get("chunk", -1),
            "confidence": max(0.0, round(1.0 - float(dist), 3)),
        })
    avg_conf = sum(h["confidence"] for h in hits) / len(hits) if hits else 0.0
    return {"hits": hits, "avg_confidence": round(avg_conf, 3)}


def collection_size() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0
