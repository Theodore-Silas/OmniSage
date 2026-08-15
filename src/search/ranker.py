"""
Hybrid result ranker: BM25 lexical ranking.
Semantic ranking is optional and only enabled if sentence-transformers model is locally cached.
"""

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    _HAS_NUMPY = False

from typing import List
import os

from src.search.base import SearchResult


class ResultRanker:
    """Ranks search results using BM25 (+ optional semantic if model is cached)."""

    def __init__(self):
        self._model = None
        self._model_tried = False

    def _try_load_model(self):
        if self._model_tried:
            return self._model
        self._model_tried = True
        try:
            cache = os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface", "hub",
                "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
            )
            if os.path.isdir(cache):
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    local_files_only=True,
                )
        except Exception:
            pass
        return self._model

    def rank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results

        texts = [f"{r.title} {r.snippet}"[:300] for r in results]

        # BM25-like lexical scores
        query_terms = set(query.lower().split())
        if _HAS_NUMPY:
            bm25_scores = np.array([
                sum(1 for t in query_terms if t in text.lower()) / max(len(query_terms), 1)
                for text in texts
            ])
        else:
            bm25_scores = [
                sum(1 for t in query_terms if t in text.lower()) / max(len(query_terms), 1)
                for text in texts
            ]

        # Try semantic (only if locally cached)
        model = self._try_load_model()
        if model is not None and _HAS_NUMPY:
            try:
                query_emb = model.encode([query], normalize_embeddings=True)
                doc_embs = model.encode(texts, normalize_embeddings=True)
                semantic_scores = (doc_embs @ query_emb.T).flatten()
                hybrid = 0.6 * semantic_scores + 0.3 * bm25_scores + 0.1 * np.array([r.score for r in results])
            except Exception:
                hybrid = self._combine_scores(bm25_scores, results)
        else:
            hybrid = self._combine_scores(bm25_scores, results)

        ranked = sorted(zip(results, hybrid), key=lambda x: x[1], reverse=True)
        for r, score in ranked:
            r.score = float(score)
        return [r for r, _ in ranked]

    def _combine_scores(self, bm25_scores, results: List[SearchResult]):
        """Combine BM25 scores with original result scores. Works with or without numpy."""
        if _HAS_NUMPY:
            return 0.7 * bm25_scores + 0.3 * np.array([r.score for r in results])
        else:
            return [0.7 * b + 0.3 * r.score for b, r in zip(bm25_scores, results)]
