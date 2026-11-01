"""
Clause-Aware Vector Retrieval Engine and Semantic Index.
Maintains session/tenant isolation and evaluates retrieval performance (Recall@K, MRR).
"""

from __future__ import annotations
import math
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any

from src.ingest.parser import ParsedDocument
from src.data.schema import ClauseAnnotation


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    text: str
    category: Optional[str]
    page_number: int
    section_title: Optional[str]
    char_start: int
    char_end: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClauseAwareVectorStore:
    """Session-isolated, clause-aware vector index using BM25/TF-IDF and cosine similarity."""

    def __init__(self, session_id: str = "default_session"):
        self.session_id = session_id
        # Isolated storage per doc_id: doc_id -> list of chunks
        self._doc_chunks: Dict[str, List[RetrievedChunk]] = {}
        # Precomputed term statistics per doc_id: doc_id -> {chunk_idx: {word: tfidf}}
        self._doc_tfidf: Dict[str, List[Dict[str, float]]] = {}
        self._doc_idf: Dict[str, Dict[str, float]] = {}

    def index_document(self, parsed_doc: ParsedDocument, clauses: Optional[List[ClauseAnnotation]] = None) -> int:
        """Indexes parsed document sections and identified clauses."""
        chunks: List[RetrievedChunk] = []
        doc_id = parsed_doc.doc_id

        # 1. Add identified clauses as high-priority chunks
        if clauses:
            for idx, clause in enumerate(clauses):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{doc_id}_cl_{idx:03d}",
                        doc_id=doc_id,
                        text=clause.text,
                        category=clause.category,
                        page_number=clause.page_number or 1,
                        section_title=clause.section_title or "Identified Provision",
                        char_start=clause.start_char,
                        char_end=clause.end_char,
                        score=0.0,
                    )
                )

        # 2. Add document sections/paragraphs to ensure full coverage
        for sec in parsed_doc.sections:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", sec.content) if len(p.strip()) > 40]
            for p_idx, para in enumerate(paragraphs):
                # Avoid duplicate if matches an existing clause
                if any(para in c.text or c.text in para for c in chunks):
                    continue
                chunks.append(
                    RetrievedChunk(
                        chunk_id=f"{doc_id}_sec_{sec.section_id}_{p_idx}",
                        doc_id=doc_id,
                        text=para,
                        category=None,
                        page_number=sec.page_number,
                        section_title=sec.title,
                        char_start=sec.char_start,
                        char_end=sec.char_end,
                        score=0.0,
                    )
                )

        self._doc_chunks[doc_id] = chunks
        self._build_tfidf_index(doc_id, chunks)
        return len(chunks)

    def retrieve(self, doc_id: str, query: str, top_k: int = 3, threshold: float = 0.05) -> List[RetrievedChunk]:
        """Retrieves top-k relevant clauses for a query within isolated document index."""
        if doc_id not in self._doc_chunks:
            return []

        chunks = self._doc_chunks[doc_id]
        if not chunks:
            return []

        tfidf_list = self._doc_tfidf[doc_id]
        idf = self._doc_idf[doc_id]
        query_vec = self._vectorize_query(query, idf)

        scored_chunks: List[Tuple[float, RetrievedChunk]] = []
        for idx, chunk in enumerate(chunks):
            doc_vec = tfidf_list[idx]
            sim = self._cosine_similarity(query_vec, doc_vec)
            # Bonus boost only if there is already a semantic match and query mentions key category word
            cat_words = self._tokenize(chunk.category) if chunk.category else []
            if sim > 0.05 and any(w in query.lower() for w in cat_words):
                sim += 0.25

            if sim >= threshold:
                c_copy = RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    text=chunk.text,
                    category=chunk.category,
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    score=round(sim, 4),
                )
                scored_chunks.append((sim, c_copy))

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored_chunks[:top_k]]

    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]], doc_id: str, k: int = 3) -> Dict[str, float]:
        """Evaluates retrieval quality against ground truth queries (ACORD benchmark: Recall@K and MRR)."""
        recalls: List[float] = []
        reciprocal_ranks: List[float] = []

        for item in test_queries:
            query = item["query"]
            target_keyword = item.get("target_keyword", "").lower()
            results = self.retrieve(doc_id, query, top_k=k, threshold=0.01)

            hit_rank = -1
            for rank, res in enumerate(results, start=1):
                if target_keyword in res.text.lower() or (res.category and target_keyword in res.category.lower()):
                    hit_rank = rank
                    break

            if hit_rank > 0:
                recalls.append(1.0)
                reciprocal_ranks.append(1.0 / hit_rank)
            else:
                recalls.append(0.0)
                reciprocal_ranks.append(0.0)

        n = len(test_queries) if test_queries else 1
        return {
            f"Recall@{k}": round(sum(recalls) / n, 4),
            "MRR": round(sum(reciprocal_ranks) / n, 4),
            "total_queries_evaluated": len(test_queries),
        }

    STOP_WORDS = {
        "the", "a", "an", "is", "in", "of", "to", "for", "and", "or", "this", "that",
        "it", "with", "as", "by", "on", "at", "from", "does", "do", "did", "are", "be",
        "was", "were", "what", "which", "who", "whom", "will", "would", "shall", "can",
        "could", "should", "contract", "agreement", "provision", "document", "party",
        "parties", "thereof", "hereof", "under", "any", "all", "non", "pre", "post"
    }

    def _tokenize(self, text: str) -> List[str]:
        """Tokenizes, lowercases, and removes common stop words."""
        tokens = re.findall(r"\b[A-Za-z0-9_]{2,}\b", text.lower())
        return [t for t in tokens if t not in self.STOP_WORDS]

    def _build_tfidf_index(self, doc_id: str, chunks: List[RetrievedChunk]):
        """Computes TF-IDF index for the given document chunks."""
        doc_count = len(chunks)
        df: Dict[str, int] = {}
        tf_list: List[Dict[str, float]] = []

        for chunk in chunks:
            tokens = self._tokenize(chunk.text + " " + (chunk.section_title or "") + " " + (chunk.category or ""))
            tf: Dict[str, float] = {}
            seen = set()
            for t in tokens:
                tf[t] = tf.get(t, 0.0) + 1.0
                if t not in seen:
                    df[t] = df.get(t, 0) + 1
                    seen.add(t)
            # Normalize TF
            total_tokens = max(1, len(tokens))
            for t in tf:
                tf[t] = tf[t] / total_tokens
            tf_list.append(tf)

        # Compute IDF
        idf: Dict[str, float] = {}
        for t, freq in df.items():
            idf[t] = math.log((doc_count + 1.0) / (freq + 1.0)) + 1.0

        # Compute TF-IDF
        tfidf_list: List[Dict[str, float]] = []
        for tf in tf_list:
            vec: Dict[str, float] = {}
            for t, val in tf.items():
                vec[t] = val * idf.get(t, 1.0)
            tfidf_list.append(vec)

        self._doc_tfidf[doc_id] = tfidf_list
        self._doc_idf[doc_id] = idf

    def _vectorize_query(self, query: str, idf: Dict[str, float]) -> Dict[str, float]:
        """Converts query to normalized TF-IDF vector."""
        tokens = self._tokenize(query)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0.0) + 1.0
        vec: Dict[str, float] = {}
        for t, count in tf.items():
            vec[t] = (count / max(1, len(tokens))) * idf.get(t, 1.0)
        return vec

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculates cosine similarity between two sparse vectors."""
        dot = sum(val * vec2[k] for k, val in vec1.items() if k in vec2)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
