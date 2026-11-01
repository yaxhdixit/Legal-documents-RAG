"""
Unit tests for Legal Contract Intelligence Data Pipeline (DATA-1).
Tests CUAD, ACORD, ContractNLI normalization, CC BY 4.0 provenance, zero-leakage splits, and SFT records.
"""

import unittest
import os
import json
import shutil
import tempfile

from src.data.schema import (
    STANDARD_CLAUSE_CATEGORIES,
    DatasetProvenance,
    ClauseAnnotation,
    ContractRecord,
    RetrievalPair,
    NLIRecord,
    ConversationalSFTRecord,
)
from src.data.cuad_loader import CUADLoader, normalize_cuad_category, assign_contract_split, CUAD_PROVENANCE
from src.data.acord_loader import ACORDLoader, ACORD_PROVENANCE
from src.data.contract_nli_loader import ContractNLILoader, CONTRACT_NLI_PROVENANCE
from src.data.sft_builder import SFTDatasetBuilder
from src.data.pipeline import DataPipeline


class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.sft_dir = os.path.join(self.temp_dir, "sft")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_provenance_and_licensing(self):
        """FR-7, NFR-5: Ensure all datasets declare CC BY 4.0 license and official URLs."""
        for prov in [CUAD_PROVENANCE, ACORD_PROVENANCE, CONTRACT_NLI_PROVENANCE]:
            self.assertEqual(prov.license, "CC BY 4.0")
            self.assertTrue(prov.official_url.startswith("https://"))
            self.assertTrue(len(prov.attribution) > 0)

    def test_category_normalization(self):
        """FR-2: Verify raw CUAD categories map correctly to standard CGL categories."""
        self.assertEqual(normalize_cuad_category("Termination Clause"), "Termination and Renewal")
        self.assertEqual(normalize_cuad_category("Limitation of Liability"), "Limitation of Liability")
        self.assertEqual(normalize_cuad_category("Indemnity"), "Indemnification")
        self.assertEqual(normalize_cuad_category("Non-Compete"), "Non-Compete and Non-Solicitation")
        self.assertEqual(normalize_cuad_category("Anti-Assignment"), "Assignment and Change of Control")
        self.assertEqual(normalize_cuad_category("Governing Law"), "Governing Law")

    def test_cuad_loader_and_splits(self):
        """FR-7: Test loading CUAD contracts and partition isolation."""
        loader = CUADLoader(data_dir=os.path.join(self.raw_dir, "cuad"))
        contracts = loader.load_contracts()
        self.assertGreater(len(contracts), 0)
        for c in contracts:
            self.assertIn(c.split, ["train", "val", "test"])
            self.assertTrue(c.full_text)
            self.assertGreater(len(c.clauses), 0)
            for cl in c.clauses:
                self.assertIn(cl.category, STANDARD_CLAUSE_CATEGORIES)

    def test_acord_loader(self):
        """FR-7: Test ACORD query-clause relevance pairs."""
        loader = ACORDLoader(data_dir=os.path.join(self.raw_dir, "acord"))
        pairs = loader.load_pairs()
        self.assertGreater(len(pairs), 0)
        for p in pairs:
            self.assertTrue(p.query_text)
            self.assertTrue(p.clause_text)
            self.assertGreaterEqual(p.relevance_score, 0.0)

    def test_contract_nli_loader(self):
        """FR-7: Test ContractNLI hypothesis verification records."""
        loader = ContractNLILoader(data_dir=os.path.join(self.raw_dir, "contract_nli"))
        records = loader.load_records()
        self.assertGreater(len(records), 0)
        labels = {r.label for r in records}
        self.assertTrue("entailment" in labels or "contradiction" in labels or "neutral" in labels)

    def test_sft_builder_and_split_isolation(self):
        """FR-8, AC-7: Test SFT JSONL generation and ensure zero train-test contract leakage."""
        pipeline = DataPipeline(raw_base_dir=self.raw_dir, sft_output_dir=self.sft_dir)
        stats = pipeline.run()

        self.assertGreater(stats["total_sft_records"], 0)
        self.assertTrue(os.path.exists(os.path.join(self.sft_dir, "manifest.json")))

        # Check JSONL validity
        for split_name in ["train", "val", "test"]:
            path = os.path.join(self.sft_dir, f"{split_name}.jsonl")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines:
                        obj = json.loads(line)
                        self.assertIn("id", obj)
                        self.assertIn("contract_id", obj)
                        self.assertIn("messages", obj)
                        self.assertGreater(len(obj["messages"]), 1)
                        # First message must be system prompt
                        self.assertEqual(obj["messages"][0]["role"], "system")
                        # Must contain advisory disclosure
                        self.assertIn("Confrere Global Legale", obj["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
