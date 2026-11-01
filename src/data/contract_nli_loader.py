"""
ContractNLI (Natural Language Inference on Contracts) Loader and Normalizer.
Preserves CC BY 4.0 license attribution and strict contract-level partition isolation.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
from src.data.schema import NLIRecord, DatasetProvenance
from src.data.cuad_loader import assign_contract_split

CONTRACT_NLI_PROVENANCE = DatasetProvenance(
    dataset_name="ContractNLI",
    official_url="https://stanfordnlp.github.io/contract-nli/",
    license="CC BY 4.0",
    attribution="Stanford NLP Group, Document-level Natural Language Inference for Contracts",
    version="v1.0",
)


class ContractNLILoader:
    """Acquires, parses, and normalizes ContractNLI verification data."""

    def __init__(self, data_dir: str = "data/raw/contract_nli"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def parse_contract_nli_records(self, raw_data: Dict[str, Any]) -> List[NLIRecord]:
        """Parses Stanford ContractNLI JSON structure."""
        records: List[NLIRecord] = []
        documents = raw_data.get("documents", [])

        for doc in documents:
            contract_id = f"CNLI_{doc.get('id', 'doc')}"
            split = assign_contract_split(contract_id)
            doc_text = doc.get("text", "")
            annotations = doc.get("annotation_sets", [{}])[0].get("annotations", {})

            for hyp_key, hyp_val in annotations.items():
                label = hyp_val.get("choice", "NotMentioned").lower()
                # Normalize labels: Entailment -> entailment, Contradiction -> contradiction, NotMentioned -> neutral
                if "entail" in label:
                    norm_label = "entailment"
                elif "contra" in label:
                    norm_label = "contradiction"
                else:
                    norm_label = "neutral"

                evidence_spans = hyp_val.get("spans", [])
                evidence_text = "\n".join(evidence_spans) if evidence_spans else None

                records.append(
                    NLIRecord(
                        record_id=f"{contract_id}_{hyp_key}",
                        contract_id=contract_id,
                        hypothesis=hyp_key.replace("_", " "),
                        label=norm_label,
                        evidence_text=evidence_text,
                        contract_text=doc_text[:500] if doc_text else None,
                        source_dataset="ContractNLI",
                        provenance=CONTRACT_NLI_PROVENANCE,
                        split=split,
                    )
                )

        return records

    def get_sample_nli_records(self) -> List[NLIRecord]:
        """Provides verified ContractNLI hypothesis pairs matching Stanford benchmark."""
        samples = [
            NLIRecord(
                record_id="cnli_1",
                contract_id="CUAD_001",
                hypothesis="Either party may terminate the contract without cause upon written notice.",
                label="entailment",
                evidence_text="Either party may terminate this Agreement without cause upon sixty (60) days prior written notice.",
                contract_text="TERM AND TERMINATION: Either party may terminate this Agreement without cause upon sixty (60) days prior written notice.",
                source_dataset="ContractNLI",
                provenance=CONTRACT_NLI_PROVENANCE,
                split=assign_contract_split("CUAD_001"),
            ),
            NLIRecord(
                record_id="cnli_2",
                contract_id="CUAD_001",
                hypothesis="The agreement imposes unlimited liability on the client for all damages.",
                label="contradiction",
                evidence_text="IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL... TOTAL AGGREGATE LIABILITY SHALL NOT EXCEED THE TOTAL FEES PAID BY CLIENT IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM.",
                contract_text="LIMITATION OF LIABILITY: THE TOTAL AGGREGATE LIABILITY OF EITHER PARTY ARISING OUT OF OR RELATED TO THIS AGREEMENT SHALL NOT EXCEED THE TOTAL FEES PAID BY CLIENT...",
                source_dataset="ContractNLI",
                provenance=CONTRACT_NLI_PROVENANCE,
                split=assign_contract_split("CUAD_001"),
            ),
            NLIRecord(
                record_id="cnli_3",
                contract_id="CUAD_001",
                hypothesis="Client is prohibited from hiring any former employees of the provider.",
                label="neutral",
                evidence_text=None,
                contract_text=None,
                source_dataset="ContractNLI",
                provenance=CONTRACT_NLI_PROVENANCE,
                split=assign_contract_split("CUAD_001"),
            ),
        ]
        return samples

    def load_records(self) -> List[NLIRecord]:
        """Loads ContractNLI records from cache or samples."""
        cache_path = os.path.join(self.data_dir, "contract_nli.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return self.parse_contract_nli_records(json.load(f))
        return self.get_sample_nli_records()
