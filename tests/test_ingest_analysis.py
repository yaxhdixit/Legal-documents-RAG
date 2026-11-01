"""
Unit tests for Document Ingestion, Clause Extraction, and Risk Analysis (INGEST-1).
Tests parser structure mapping, CUAD clause identification, and grounded risk prioritizations.
"""

import unittest
import os
import tempfile

from src.ingest.parser import ContractParser, ParsedDocument
from src.ingest.clause_extractor import ClauseExtractor
from src.analysis.risk_engine import ContractRiskEngine, RiskFinding


SAMPLE_COMMERCIAL_CONTRACT = """MASTER SERVICES AGREEMENT

This Master Services Agreement is entered into on February 10, 2024 between LexCorp Dynamics ("Client") and TechVanguard Ltd ("Provider").

1. TERM AND TERMINATION
This Agreement shall commence on the Effective Date and remain in effect for two (2) years. Either party may terminate without cause upon thirty (30) days prior written notice. If either party materially breaches, the other party may terminate immediately.

2. LIMITATION OF LIABILITY
IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES. THE TOTAL AGGREGATE LIABILITY OF EITHER PARTY SHALL NOT EXCEED THE TOTAL FEES PAID BY CLIENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.

3. INDEMNIFICATION
Provider shall indemnify, defend, and hold harmless Client against any third-party claims alleging that the software infringes any patent or copyright.

4. CONFIDENTIALITY
Each party shall maintain in strict confidence all proprietary and confidential information disclosed hereunder for a period of three (3) years.

5. INTELLECTUAL PROPERTY OWNERSHIP
Provider retains all right, title, and interest in and to its pre-existing intellectual property and the core platform software.

6. GOVERNING LAW AND DISPUTE RESOLUTION
This Agreement is governed by the laws of England and Wales. Any dispute arising hereunder shall be referred to binding arbitration under the LCIA Rules.

7. PAYMENT OBLIGATIONS
Client shall pay all undisputed invoices within thirty (30) days of receipt (Net 30). Late payments incur interest at 1.5% per month.

8. EXCLUSIVITY AND NON-SOLICITATION
Provider agrees not to supply competing custom software to direct competitors of Client in North America during the term. Neither party shall solicit or hire employees of the other party during the term and for one year thereafter.
"""


class TestIngestAndAnalysis(unittest.TestCase):
    def setUp(self):
        self.parser = ContractParser()
        self.extractor = ClauseExtractor()
        self.risk_engine = ContractRiskEngine()

    def test_parser_string_and_sections(self):
        """FR-1, AC-1: Verify document parsing, page mapping, and section headings."""
        doc = self.parser.parse_string(SAMPLE_COMMERCIAL_CONTRACT, filename="test_contract.txt")
        self.assertEqual(doc.filename, "test_contract.txt")
        self.assertGreater(len(doc.full_text), 500)
        self.assertGreater(len(doc.sections), 4)
        self.assertGreater(len(doc.pages), 0)

        # Check section titles and coordinates
        titles = [s.title for s in doc.sections]
        self.assertTrue(any("TERM AND TERMINATION" in t for t in titles))
        self.assertTrue(any("LIMITATION OF LIABILITY" in t for t in titles))
        for s in doc.sections:
            self.assertGreaterEqual(s.char_end, s.char_start)
            self.assertGreaterEqual(s.page_number, 1)

    def test_clause_extractor_categories(self):
        """FR-2, AC-2: Verify detection of 14 CUAD clause categories."""
        doc = self.parser.parse_string(SAMPLE_COMMERCIAL_CONTRACT)
        clauses = self.extractor.extract_clauses(doc)
        self.assertGreater(len(clauses), 5)

        categories = {c.category for c in clauses}
        self.assertIn("Termination and Renewal", categories)
        self.assertIn("Limitation of Liability", categories)
        self.assertIn("Indemnification", categories)
        self.assertIn("Confidentiality", categories)
        self.assertIn("Governing Law", categories)
        self.assertIn("Payment Obligations", categories)

        for cl in clauses:
            self.assertTrue(cl.text)
            self.assertGreater(cl.confidence, 0.5)
            self.assertGreaterEqual(cl.page_number, 1)
            # Must locate clause within the full contract text
            self.assertIn(cl.text, doc.full_text)

    def test_risk_detection_and_review_priorities(self):
        """FR-3, NFR-1, AC-3: Verify risk priority ratings, grounded quotes, and lawyer questions."""
        doc = self.parser.parse_string(SAMPLE_COMMERCIAL_CONTRACT)
        clauses = self.extractor.extract_clauses(doc)
        report = self.risk_engine.analyze_contract(doc, clauses)

        self.assertGreater(len(report.findings), 0)
        self.assertIn("advisory review priorities", report.overall_risk_summary.lower())

        priorities = {f.risk_priority for f in report.findings}
        self.assertTrue("High" in priorities or "Medium" in priorities)

        for finding in report.findings:
            # Check mandatory attributes
            self.assertTrue(finding.clause_name)
            self.assertIn(finding.risk_priority, ["High", "Medium", "Low", "Requires Review"])
            self.assertIn(finding.risk_type, ["problematic", "ambiguous", "missing", "unfavorable", "contextual"])
            self.assertTrue(finding.concern_explanation)
            self.assertTrue(finding.commercial_implications)
            self.assertTrue(finding.affected_party)
            self.assertTrue(finding.evidence_quote)
            self.assertGreater(len(finding.lawyer_review_questions), 0)

        # Check missing provision detection (Data Protection is absent in the sample)
        self.assertIn("Data Protection and Security", report.missing_provisions)
        missing_finding = next((f for f in report.findings if f.clause_name == "Data Protection and Security"), None)
        self.assertIsNotNone(missing_finding)
        self.assertEqual(missing_finding.risk_type, "missing")
        self.assertEqual(missing_finding.risk_priority, "Requires Review")


if __name__ == "__main__":
    unittest.main()
