"""
Academic Vector Store & Hybrid Retrieval Engine (Dense + BM25)
Project: Academic Doubt-Solving RAG System
Author: Lanka Jeeshita | Guide: Dr. P. Sudhakar
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .preprocessor import AcademicChunk


class BM25Retriever:
    """
    Academic Lexical Search using the Okapi BM25 algorithm.
    Optimized for formula names, acronyms, and verbatim course definitions.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_len: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        # Lowercase, retain alphanumeric tokens and key academic symbols
        tokens = re.findall(r"\b[a-zA-Z0-9_\-\.]{2,}\b", text.lower())
        return tokens

    def fit(self, chunks: List[AcademicChunk]):
        self.corpus = [self.tokenize(c.text) for c in chunks]
        self.num_docs = len(self.corpus)
        if self.num_docs == 0:
            return

        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_doc_len = sum(self.doc_lengths) / max(1, self.num_docs)

        self.doc_freqs = {}
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.idf = {}
        for term, freq in self.doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.num_docs - freq + 0.5) / (freq + 0.5))

    def score(self, query: str) -> np.ndarray:
        if self.num_docs == 0:
            return np.array([])

        q_tokens = self.tokenize(query)
        scores = np.zeros(self.num_docs, dtype=float)

        for token in q_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for i, doc in enumerate(self.corpus):
                freq = doc.count(token)
                if freq == 0:
                    continue
                num = freq * (self.k1 + 1.0)
                denom = freq + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / max(1.0, self.avg_doc_len)))
                scores[i] += idf_val * (num / denom)

        return scores


class AcademicVectorEngine:
    """
    Hybrid Vector Store with Dual Retrieval:
    - Dense Embeddings (Google Gemini text-embedding-004 or local normalized TF-IDF)
    - Sparse BM25 Lexical Matching
    - Reciprocal Rank Fusion (RRF) for robust multi-modal retrieval.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.chunks: List[AcademicChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.bm25 = BM25Retriever()
        self.vocabulary: Dict[str, int] = {}
        self.use_api_embeddings = bool(self.api_key)

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
        self.use_api_embeddings = bool(self.api_key)

    def _compute_local_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Built-in zero-dependency normalized TF-IDF dense representation.
        Ensures 100% offline functionality if no API key is provided.
        """
        tokenized = [BM25Retriever.tokenize(t) for t in texts]

        # Build vocabulary from corpus if empty
        if not self.vocabulary:
            freq_dict = {}
            for doc in tokenized:
                for token in doc:
                    freq_dict[token] = freq_dict.get(token, 0) + 1
            # Top 2048 distinctive vocabulary words
            sorted_vocab = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:2048]
            self.vocabulary = {item[0]: idx for idx, item in enumerate(sorted_vocab)}

        vocab_size = max(1, len(self.vocabulary))
        matrix = np.zeros((len(texts), vocab_size), dtype=np.float32)

        for i, doc in enumerate(tokenized):
            for token in doc:
                if token in self.vocabulary:
                    matrix[i, self.vocabulary[token]] += 1.0

            # L2 normalize
            norm = np.linalg.norm(matrix[i])
            if norm > 0:
                matrix[i] = matrix[i] / norm

        return matrix

    def _compute_gemini_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """
        Fetches high-dimensional dense embeddings using Google GenAI SDK.
        """
        if not self.api_key:
            return None
        try:
            from google import genai
            client = genai.Client(api_key=self.api_key)
            embeddings_list = []

            # Batch requests in groups of 20 to respect rate limits
            batch_size = 20
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                # Embed each text
                for t in batch:
                    res = client.models.embed_content(
                        model="text-embedding-004",
                        contents=t,
                    )
                    # res.embedding or res.embeddings[0]
                    emb_vals = res.embedding.values if hasattr(res, "embedding") and res.embedding else res.embeddings[0].values
                    embeddings_list.append(emb_vals)

            arr = np.array(embeddings_list, dtype=np.float32)
            # Normalize vectors
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return arr / norms
        except Exception as e:
            print(f"[Warning] Gemini Embedding API call failed: {e}. Falling back to local vector representation.")
            return None

    def build_index(self, chunks: List[AcademicChunk]):
        """
        Builds the hybrid search index from academic chunks.
        """
        self.chunks = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [c.text for c in chunks]

        # 1. Fit Sparse BM25
        self.bm25.fit(chunks)

        # 2. Dense Embeddings
        dense_embs = None
        if self.use_api_embeddings:
            dense_embs = self._compute_gemini_embeddings(texts)

        if dense_embs is None:
            dense_embs = self._compute_local_tfidf_embeddings(texts)

        self.embeddings = dense_embs

    def _embed_query(self, query: str) -> np.ndarray:
        if self.use_api_embeddings and self.embeddings is not None and self.embeddings.shape[1] > 2048:
            try:
                from google import genai
                client = genai.Client(api_key=self.api_key)
                res = client.models.embed_content(
                    model="text-embedding-004",
                    contents=query
                )
                emb_vals = res.embedding.values if hasattr(res, "embedding") and res.embedding else res.embeddings[0].values
                q_vec = np.array(emb_vals, dtype=np.float32)
                norm = np.linalg.norm(q_vec)
                return q_vec / (norm if norm > 0 else 1.0)
            except Exception:
                pass

        # Fallback to local vector
        matrix = self._compute_local_tfidf_embeddings([query])
        return matrix[0]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 4,
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        Performs Hybrid Search using Reciprocal Rank Fusion (RRF)
        between Dense Cosine Similarity and Sparse BM25 scoring.
        """
        if not self.chunks or self.embeddings is None:
            return []

        num_docs = len(self.chunks)

        # 1. Dense Cosine Similarity
        q_vec = self._embed_query(query)
        if q_vec.shape[0] != self.embeddings.shape[1]:
            # Fallback if dimension mismatch
            q_vec = self._compute_local_tfidf_embeddings([query])[0]

        dense_scores = np.dot(self.embeddings, q_vec)
        dense_ranking = np.argsort(-dense_scores)

        # 2. Sparse BM25 Scoring
        bm25_scores = self.bm25.score(query)
        if len(bm25_scores) > 0 and np.max(bm25_scores) > 0:
            bm25_ranking = np.argsort(-bm25_scores)
        else:
            bm25_ranking = dense_ranking

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_k = 60
        rrf_scores = np.zeros(num_docs, dtype=float)

        for rank_idx, doc_idx in enumerate(dense_ranking):
            rrf_scores[doc_idx] += dense_weight * (1.0 / (rrf_k + rank_idx + 1))

        for rank_idx, doc_idx in enumerate(bm25_ranking):
            rrf_scores[doc_idx] += bm25_weight * (1.0 / (rrf_k + rank_idx + 1))

        # Sort by final combined score
        top_indices = np.argsort(-rrf_scores)[:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            results.append({
                "chunk": chunk,
                "score": float(rrf_scores[idx]),
                "dense_score": float(dense_scores[idx]) if idx < len(dense_scores) else 0.0,
                "bm25_score": float(bm25_scores[idx]) if idx < len(bm25_scores) else 0.0,
            })

        return results

    def save_to_disk(self, directory: str = "vector_db"):
        """Saves chunks, vocabulary, and embeddings to disk."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "chunks_metadata.json")
        embs_path = os.path.join(directory, "embeddings.npy")
        vocab_path = os.path.join(directory, "vocab.json")

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocabulary, f, indent=2)

        if self.embeddings is not None:
            np.save(embs_path, self.embeddings)

    def load_from_disk(self, directory: str = "vector_db") -> bool:
        """Loads index from disk if available."""
        meta_path = os.path.join(directory, "chunks_metadata.json")
        embs_path = os.path.join(directory, "embeddings.npy")
        vocab_path = os.path.join(directory, "vocab.json")

        if not (os.path.exists(meta_path) and os.path.exists(embs_path)):
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.chunks = [AcademicChunk(**item) for item in data]

            if os.path.exists(vocab_path):
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocabulary = json.load(f)

            self.embeddings = np.load(embs_path)
            self.bm25.fit(self.chunks)
            return True
        except Exception as e:
            print(f"[Error] Failed to load index from disk: {e}")
            return False
