"""
Orchestration Pipeline for Legal Contract Datasets and Conversational SFT Generation.
Preserves CC BY 4.0 provenance, attribution, and strict contract-level partition isolation.
"""

from __future__ import annotations
import os
import json
from typing import Dict, Any, List

from src.data.cuad_loader import CUADLoader, CUAD_PROVENANCE
from src.data.acord_loader import ACORDLoader, ACORD_PROVENANCE
from src.data.contract_nli_loader import ContractNLILoader, CONTRACT_NLI_PROVENANCE
from src.data.sft_builder import SFTDatasetBuilder
from src.data.schema import ConversationalSFTRecord


class DataPipeline:
    """End-to-end data acquisition, validation, normalization, and SFT pipeline."""

    def __init__(self, raw_base_dir: str = "data/raw", sft_output_dir: str = "data/sft"):
        self.raw_base_dir = raw_base_dir
        self.sft_output_dir = sft_output_dir
        self.cuad_loader = CUADLoader(data_dir=os.path.join(raw_base_dir, "cuad"))
        self.acord_loader = ACORDLoader(data_dir=os.path.join(raw_base_dir, "acord"))
        self.nli_loader = ContractNLILoader(data_dir=os.path.join(raw_base_dir, "contract_nli"))
        self.sft_builder = SFTDatasetBuilder(output_dir=sft_output_dir)

    def run(self) -> Dict[str, Any]:
        """Executes full pipeline and returns summary stats."""
        # 1. Load normalized datasets
        contracts = self.cuad_loader.load_contracts()
        acord_pairs = self.acord_loader.load_pairs()
        nli_records = self.nli_loader.load_records()

        # 2. Check provenance
        provenances = [
            CUAD_PROVENANCE.to_dict(),
            ACORD_PROVENANCE.to_dict(),
            CONTRACT_NLI_PROVENANCE.to_dict(),
        ]
        for prov in provenances:
            assert prov["license"] == "CC BY 4.0", f"License must be CC BY 4.0, got {prov['license']}"
            assert prov["official_url"], "Official URL must be preserved"

        # 3. Verify zero leakage by checking contract ID sets across splits
        contract_splits: Dict[str, str] = {}
        for c in contracts:
            contract_splits[c.contract_id] = c.split

        # 4. Synthesize conversational SFT records
        all_sft_records: List[ConversationalSFTRecord] = []

        for c in contracts:
            all_sft_records.extend(self.sft_builder.build_clause_extraction_dialogue(c))
            all_sft_records.extend(self.sft_builder.build_risk_flagging_dialogue(c))
            multiturn = self.sft_builder.build_multi_turn_dialogue(c)
            if multiturn:
                all_sft_records.append(multiturn)

        for pair in acord_pairs:
            all_sft_records.append(self.sft_builder.build_retrieval_qa_dialogue(pair))

        for nli in nli_records:
            all_sft_records.append(self.sft_builder.build_nli_verification_dialogue(nli))

        # 5. Export split files
        split_paths = self.sft_builder.export_sft_splits(all_sft_records)

        # 6. Verify train/val/test contract isolation
        split_contract_ids: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
        for r in all_sft_records:
            split_contract_ids[r.split].add(r.contract_id)

        # Confirm no overlap between train and test/val contract IDs
        train_val_overlap = split_contract_ids["train"].intersection(split_contract_ids["val"])
        train_test_overlap = split_contract_ids["train"].intersection(split_contract_ids["test"])
        val_test_overlap = split_contract_ids["val"].intersection(split_contract_ids["test"])

        assert len(train_val_overlap) == 0, f"Data leakage detected between train and val: {train_val_overlap}"
        assert len(train_test_overlap) == 0, f"Data leakage detected between train and test: {train_test_overlap}"
        assert len(val_test_overlap) == 0, f"Data leakage detected between val and test: {val_test_overlap}"

        stats = {
            "num_contracts": len(contracts),
            "num_acord_pairs": len(acord_pairs),
            "num_nli_records": len(nli_records),
            "total_sft_records": len(all_sft_records),
            "splits": {k: len(v) for k, v in split_contract_ids.items()},
            "split_file_paths": split_paths,
            "provenances": provenances,
        }

        # Write dataset manifest
        manifest_path = os.path.join(self.sft_output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats


if __name__ == "__main__":
    pipeline = DataPipeline()
    res = pipeline.run()
    print(f"Pipeline executed successfully: {res['total_sft_records']} SFT records generated across {res['num_contracts']} contracts.")
