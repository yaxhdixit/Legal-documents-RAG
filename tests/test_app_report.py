"""
Unit tests for Web Application API and Report Generator (UI-1).
Tests FastAPI endpoints, CGL layout rendering, and contract review report generation.
"""

import unittest
from fastapi.testclient import TestClient

from src.server.app import app
from src.server.report_generator import ReportGenerator
from src.ingest.parser import ContractParser
from src.ingest.clause_extractor import ClauseExtractor
from src.analysis.risk_engine import ContractRiskEngine


SAMPLE_DOC = """COMMERCIAL LEASE AND LOGISTICS AGREEMENT
Between Global Freight LLC ("Tenant") and Harbour Holdings ("Landlord").
1. TERM: Two years commencing May 1, 2024.
2. INDEMNITY: Tenant shall indemnify Landlord against property damages.
3. GOVERNING LAW: Governed by the laws of England and Wales.
"""


class TestAppAndReport(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.parser = ContractParser()
        self.extractor = ClauseExtractor()
        self.risk_engine = ContractRiskEngine()

    def test_report_generator_html_and_json(self):
        """FR-6, NFR-4, AC-6: Test professional CGL review report generation."""
        doc = self.parser.parse_string(SAMPLE_DOC, filename="lease.txt")
        clauses = self.extractor.extract_clauses(doc)
        risk_rep = self.risk_engine.analyze_contract(doc, clauses)

        html = ReportGenerator.generate_html_report(doc, clauses, risk_rep)
        self.assertIn("Confrere Global Legale (CGL)", html)
        self.assertIn("Executive Risk Summary", html)
        self.assertIn("Detailed Risk Findings", html)
        self.assertIn("Legal Notice & Advisory Nature", html)

        json_rep = ReportGenerator.generate_json_report(doc, clauses, risk_rep)
        self.assertEqual(json_rep["meta"]["organization"], "Confrere Global Legale (CGL)")
        self.assertIn("document", json_rep)
        self.assertIn("risk_analysis", json_rep)

    def test_get_index_page(self):
        """FR-5, AC-5: Test loading CGL web application frontend."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Confrere Global Legale", resp.text)
        self.assertIn("Contract Risk", resp.text)
        self.assertIn("Workspace", resp.text)

    def test_api_contracts_list_and_details(self):
        """AC-5: Test retrieving contracts and clauses."""
        resp = self.client.get("/api/contracts")
        self.assertEqual(resp.status_code, 200)
        contracts = resp.json()
        self.assertGreater(len(contracts), 0)

        first_id = contracts[0]["doc_id"]
        detail_resp = self.client.get(f"/api/contracts/{first_id}")
        self.assertEqual(detail_resp.status_code, 200)
        details = detail_resp.json()
        self.assertIn("document", details)
        self.assertIn("clauses", details)
        self.assertIn("risk_analysis", details)

    def test_api_chat_and_report_endpoints(self):
        """FR-4, FR-6, AC-4, AC-6: Test chat API and report generation endpoint."""
        resp = self.client.get("/api/contracts")
        first_id = resp.json()[0]["doc_id"]

        # Test Chat API
        chat_resp = self.client.post(
            f"/api/contracts/{first_id}/chat",
            json={"query": "What are the termination terms?"},
        )
        self.assertEqual(chat_resp.status_code, 200)
        chat_data = chat_resp.json()
        self.assertIn("answer", chat_data)
        self.assertIn("citations", chat_data)

        # Test Report API (HTML & JSON)
        report_resp = self.client.get(f"/api/contracts/{first_id}/report")
        self.assertEqual(report_resp.status_code, 200)
        self.assertIn("Confrere Global Legale", report_resp.text)

        json_resp = self.client.get(f"/api/contracts/{first_id}/report/json")
        self.assertEqual(json_resp.status_code, 200)
        report_json = json_resp.json()
        self.assertIn("meta", report_json)
        self.assertIn("risk_analysis", report_json)
        self.assertIn("Content-Disposition", json_resp.headers)

    def test_api_benchmark_endpoint(self):
        """AC-8: Test comparative benchmark API."""
        bench_resp = self.client.get("/api/benchmark")
        self.assertEqual(bench_resp.status_code, 200)
        data = bench_resp.json()
        self.assertIn("baseline", data)
        self.assertIn("finetuned_slm", data)
        self.assertIn("relative_improvement", data)

    def test_upload_preserves_original_filename(self):
        """Verify uploaded contract retains exact original filename without temporary renaming."""
        sample_filename = "Strategic_Commercial_Partnership_Agreement.txt"
        file_content = b"""COMMERCIAL AGREEMENT\n1. TERM: 5 years.\n2. TERMINATION: 30 days notice.\n3. GOVERNING LAW: Delaware."""
        
        upload_resp = self.client.post(
            "/api/upload",
            files={"file": (sample_filename, file_content, "text/plain")}
        )
        self.assertEqual(upload_resp.status_code, 200)
        data = upload_resp.json()
        self.assertEqual(data["filename"], sample_filename)
        
        doc_id = data["doc_id"]
        detail_resp = self.client.get(f"/api/contracts/{doc_id}")
        self.assertEqual(detail_resp.status_code, 200)
        details = detail_resp.json()
        self.assertEqual(details["document"]["filename"], sample_filename)
        self.assertIn("overview", details)
        self.assertIn("Delaware", details["overview"]["governing_law"])


if __name__ == "__main__":
    unittest.main()
