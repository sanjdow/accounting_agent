"""ChromaDB-backed policy vector store with incremental ingestion.

Supports .txt, .md, and .pdf accounting-standard documents.
Embeds with a multilingual sentence-transformers model so German (HGB)
and English (IFRS) policy text are both handled well.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, POLICY_DIR, RAG_TOP_K


_CLIENT = None
_COLLECTION = None
_COLLECTION_NAME = "policies"
# Sidecar file recording which embedding model built the current index.
_MODEL_FINGERPRINT = VECTOR_DB_PATH / ".embedding_model"


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


def check_model_consistency() -> Dict[str, object]:
    """Detect whether the index was built with a different embedding model.

    Vectors from different models are NOT comparable. If the model changed,
    the caller must reset the index before querying.
    """
    if not _MODEL_FINGERPRINT.exists():
        return {"consistent": True, "stored_model": None, "current_model": EMBEDDING_MODEL}
    stored = _MODEL_FINGERPRINT.read_text(encoding="utf-8").strip()
    return {
        "consistent": stored == EMBEDDING_MODEL,
        "stored_model": stored,
        "current_model": EMBEDDING_MODEL,
    }


def _write_fingerprint() -> None:
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    _MODEL_FINGERPRINT.write_text(EMBEDDING_MODEL, encoding="utf-8")


def reset_index() -> None:
    """Drop the collection entirely. Required after an embedding-model change."""
    global _COLLECTION
    try:
        _client().delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    _COLLECTION = None
    if _MODEL_FINGERPRINT.exists():
        _MODEL_FINGERPRINT.unlink()


def _chunk(text: str, size: int = 600, overlap: int = 80) -> List[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    i = 0
    while i < len(text):
        chunk = text[i : i + size].strip()
        if chunk:
            chunks.append(chunk)
        i += size - overlap
    return chunks


def _doc_id(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(content.encode("utf-8")).hexdigest()[:8]
    return f"{source}:{idx}:{h}"


def _extract_text(path: Path) -> str:
    """Read a policy document into plain text. Supports .txt, .md, .pdf."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "pypdf is required to ingest PDF policy documents. "
                "Run: pip install pypdf"
            ) from e
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                pages.append(txt)
        return "\n\n".join(pages)
    raise ValueError(f"Unsupported policy file type: {path.name} ({suffix})")


def ingest_policies(extra_files: List[Path] | None = None) -> Dict[str, int]:
    """Ingest every .txt, .md, and .pdf file in POLICY_DIR (plus any extras).

    Returns {filename: chunks_added}. Idempotent via content hash.
    Records the embedding-model fingerprint so a later model change is detected.
    """
    col = _collection()
    existing = set(col.get()["ids"])
    added: Dict[str, int] = {}

    files: List[Path] = []
    if POLICY_DIR.exists():
        for pattern in ("*.txt", "*.md", "*.pdf"):
            files.extend(sorted(POLICY_DIR.glob(pattern)))
    if extra_files:
        files.extend(extra_files)

    for f in files:
        try:
            text = _extract_text(f)
        except (ImportError, ValueError) as e:
            added[f.name] = -1  # signal an error for this file
            print(f"[ingest] skipped {f.name}: {e}")
            continue

        chunks = _chunk(text)
        ids, docs, metas = [], [], []
        for i, c in enumerate(chunks):
            _id = _doc_id(f.stem, i, c)
            if _id in existing:
                continue
            ids.append(_id)
            docs.append(c)
            metas.append({"source": f.name, "chunk": i, "filetype": f.suffix.lower()})
        if ids:
            col.add(ids=ids, documents=docs, metadatas=metas)
            added[f.name] = len(ids)

    _write_fingerprint()
    return added


def query_policies(query: str, k: int = RAG_TOP_K) -> Dict:
    col = _collection()
    res = col.query(query_texts=[query], n_results=k)
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res.get("distances") else [1.0] * len(docs)
    hits = []
    for d, m, dist in zip(docs, metas, dists):
        hits.append({
            "text": d,
            "source": m.get("source", "unknown"),
            "chunk": m.get("chunk", -1),
            "filetype": m.get("filetype", ""),
            "confidence": max(0.0, round(1.0 - float(dist), 3)),
        })
    avg_conf = sum(h["confidence"] for h in hits) / len(hits) if hits else 0.0
    return {"hits": hits, "avg_confidence": round(avg_conf, 3)}


def collection_size() -> int:
    try:
        return _collection().count()
    except Exception:
        return 0
