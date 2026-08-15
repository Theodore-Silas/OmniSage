"""
FAISS-based vector memory for search result caching and similar question recall.
"""

import json
import os
import time
import numpy as np
from typing import List, Optional, Tuple

import faiss


class VectorMemory:
    """Stores search query embeddings + results for fast semantic recall."""

    def __init__(self, dim: int = 384, index_path: str = None):
        self.dim = dim
        self.index_path = index_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            ".searchagent_memory",
        )
        self.index: Optional[faiss.IndexFlatIP] = None
        self.records: List[dict] = []        # [{query, answer, sources, ts}]
        self._model = None
        self._model_available = True

    @property
    def model(self):
        if self._model is None and self._model_available:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model_available = False
                return None
        return self._model

    def _ensure_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dim)
            self._load()

    def _load(self):
        idx_file = os.path.join(self.index_path, "index.faiss")
        rec_file = os.path.join(self.index_path, "records.json")
        if os.path.exists(idx_file):
            self.index = faiss.read_index(idx_file)
        if os.path.exists(rec_file):
            with open(rec_file, "r", encoding="utf-8") as f:
                self.records = json.load(f)

    def _save(self):
        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.index_path, "index.faiss"))
        with open(os.path.join(self.index_path, "records.json"), "w", encoding="utf-8") as f:
            json.dump(self.records[-500:], f, ensure_ascii=False)  # keep last 500

    def search_similar(self, query: str, threshold: float = 0.75, top_k: int = 3) -> List[dict]:
        """Find similar past queries. Returns matching records."""
        self._ensure_index()
        if self.index.ntotal == 0 or self.model is None:
            return []

        emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(emb.astype(np.float32), min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.records) and score >= threshold:
                results.append({**self.records[idx], "similarity": float(score)})
        return results

    def store(self, query: str, answer: str, sources_count: int = 0):
        """Store a query-result pair for future recall."""
        self._ensure_index()
        if self.model is None:
            return  # skip if model unavailable
        emb = self.model.encode([query], normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self.records.append({
            "query": query,
            "answer": answer[:2000],
            "sources_count": sources_count,
            "ts": time.time(),
        })
        self._save()

    def get_recommendations(self, query: str, top_k: int = 3) -> List[str]:
        """Get similar past queries as recommendations."""
        similar = self.search_similar(query, threshold=0.6, top_k=top_k)
        return [r["query"] for r in similar if r["query"] != query]

    def size(self) -> int:
        self._ensure_index()
        return self.index.ntotal
