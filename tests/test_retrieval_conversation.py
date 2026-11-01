"""
Unit tests for Vector Retrieval and Conversational Explainer (RAG-1).
Tests clause indexing, session isolation, citation generation, ACORD metrics, and latency.
"""

import unittest
import time

from src.ingest.parser import ContractParser
from src.ingest.clause_extractor import ClauseExtractor
from src.retrieval.vector_store import ClauseAwareVectorStore
from src.conversation.explainer import ConversationalExplainer


SAMPLE_CONTRACT = """CLOUD SAAS MASTER AGREEMENT

This Agreement is dated March 15, 2024 between CloudNova Ltd ("Vendor") and GlobalFin Services ("Customer").

1. TERMINATION FOR CAUSE AND CONVENIENCE
Either party may terminate this Agreement without cause upon sixty (60) days prior written notice. If either party materially breaches, the non-breaching party may terminate immediately with ten (10) days cure period.

2. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR LOST PROFITS OR INDIRECT DAMAGES. TOTAL LIABILITY IS CAPPED AT $100,000 OR 12 MONTHS FEES PAID.

3. INDEMNITY
Vendor agrees to indemnify and defend Customer against any third-party intellectual property infringement claims.

4. GOVERNING LAW AND ARBITRATION
This Agreement shall be governed by the laws of Singapore. Any disputes shall be settled by arbitration under SIAC Rules.
"""


class TestRetrievalAndConversation(unittest.TestCase):
    def setUp(self):
        self.parser = ContractParser()
        self.extractor = ClauseExtractor()
        self.doc = self.parser.parse_string(SAMPLE_CONTRACT, filename="cloud_saas.txt")
        self.clauses = self.extractor.extract_clauses(self.doc)
        self.vector_store = ClauseAwareVectorStore(session_id="session_test_1")
        self.vector_store.index_document(self.doc, self.clauses)
        self.explainer = ConversationalExplainer(self.vector_store)

    def test_indexing_and_retrieval(self):
        """FR-4, NFR-2: Test fast clause retrieval and ranking."""
        start_time = time.time()
        results = self.vector_store.retrieve(self.doc.doc_id, "Can either party terminate without cause?", top_k=2)
        elapsed = time.time() - start_time

        self.assertLess(elapsed, 0.5, "Retrieval must take under 0.5s")
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertIn("sixty (60) days", top.text)
        self.assertEqual(top.category, "Termination and Renewal")
        self.assertGreater(top.score, 0.1)

    def test_session_and_document_isolation(self):
        """NFR-3: Verify document and tenant isolation in vector index."""
        # Querying an unindexed document ID returns empty list
        empty_res = self.vector_store.retrieve("non_existent_doc", "terminate")
        self.assertEqual(len(empty_res), 0)

        # Another isolated store instance cannot see other sessions
        isolated_store = ClauseAwareVectorStore(session_id="session_test_2")
        self.assertEqual(len(isolated_store.retrieve(self.doc.doc_id, "terminate")), 0)

    def test_acord_evaluation_metrics(self):
        """AC-4: Verify ACORD-style benchmark evaluation (Recall@K, MRR)."""
        benchmark_queries = [
            {"query": "Is there a limitation of liability cap?", "target_keyword": "capped"},
            {"query": "Who resolves disputes and what law governs?", "target_keyword": "singapore"},
            {"query": "How can the agreement be terminated?", "target_keyword": "written notice"},
        ]
        metrics = self.vector_store.evaluate_retrieval(benchmark_queries, self.doc.doc_id, k=3)
        self.assertIn("Recall@3", metrics)
        self.assertIn("MRR", metrics)
        self.assertGreaterEqual(metrics["Recall@3"], 0.6)
        self.assertGreaterEqual(metrics["MRR"], 0.5)

    def test_conversational_explainer_grounded_answer(self):
        """FR-4, NFR-1, AC-4: Verify evidence citations, page numbers, and disclaimers."""
        resp = self.explainer.answer_question(
            doc_id=self.doc.doc_id,
            query="What happens if the other party breaches this agreement?",
        )
        self.assertTrue(resp.is_information_present)
        self.assertGreater(len(resp.citations), 0)
        self.assertIn("cure period", resp.answer.lower())
        self.assertIn("Confrere Global Legale", resp.answer)

        # Citations must reference exact location
        top_cite = resp.citations[0]
        self.assertGreaterEqual(top_cite.page_number, 1)
        self.assertTrue(top_cite.quote_text)

    def test_explainer_missing_information_negative_acknowledgment(self):
        """FR-4, NFR-1: System must state absence clearly when information is missing, not hallucinate."""
        resp = self.explainer.answer_question(
            doc_id=self.doc.doc_id,
            query="Does this contract specify non-solicitation of employees?",
        )
        self.assertFalse(resp.is_information_present)
        self.assertEqual(len(resp.citations), 0)
        self.assertIn("does not contain an express provision", resp.answer)
        self.assertIn("Confrere Global Legale", resp.answer)

    def test_multi_turn_conversation_context(self):
        """FR-4: Verify multi-turn conversational history persistence."""
        sid = self.explainer.get_or_create_session()
        # Turn 1
        resp1 = self.explainer.answer_question(
            doc_id=self.doc.doc_id,
            query="What is the liability cap under this contract?",
            session_id=sid,
        )
        self.assertEqual(resp1.turn_count, 1)

        # Turn 2: Follow-up question referencing earlier context
        resp2 = self.explainer.answer_question(
            doc_id=self.doc.doc_id,
            query="Can you explain more about that provision?",
            session_id=sid,
        )
        self.assertEqual(resp2.turn_count, 2)
        self.assertTrue(len(self.explainer.sessions[sid]) == 4)  # 2 user + 2 assistant turns


if __name__ == "__main__":
    unittest.main()
