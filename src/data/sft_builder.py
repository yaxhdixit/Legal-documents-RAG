"""
Conversational SFT (Supervised Fine-Tuning) Dataset Builder.
Synthesizes structured contract intelligence dialogues from standardized CUAD, ACORD, and ContractNLI records.
Enforces strict contract ID split boundaries and CGL advisory guidelines.
"""

from __future__ import annotations
import json
import os
from typing import List, Dict, Any, Optional
from src.data.schema import (
    ContractRecord,
    RetrievalPair,
    NLIRecord,
    ConversationalSFTRecord,
    ChatTurn,
    STANDARD_CLAUSE_CATEGORIES,
)

SYSTEM_PROMPT_CGL = (
    "You are the Confrere Global Legale (CGL) Contract Intelligence SLM assistant. "
    "Your objective is to assist legal counsel, commercial teams, and clients by analyzing commercial agreements, "
    "identifying key clauses, flagging potential commercial risks with exact textual evidence, explaining obligations "
    "in plain language, and answering contractual questions. "
    "Always distinguish potential commercial risks from definitive legal advice; all risk ratings represent review priorities "
    "for qualified attorney review. Always anchor your findings strictly to verified contract text, citing exact provisions."
)


class SFTDatasetBuilder:
    """Builds conversational SFT records from normalized contract data."""

    def __init__(self, output_dir: str = "data/sft"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_clause_extraction_dialogue(self, contract: ContractRecord) -> List[ConversationalSFTRecord]:
        """Creates instruction-response dialogues for clause extraction."""
        records: List[ConversationalSFTRecord] = []
        for clause in contract.clauses:
            rec_id = f"sft_extract_{clause.clause_id}"
            user_msg = (
                f"Document: {contract.title}\n\n"
                f"Excerpt:\n\"{clause.text}\"\n\n"
                f"Task: Identify the contractual category of the provision above, quote the exact clause text, "
                f"and explain its primary legal and operational purpose."
            )
            asst_msg = (
                f"**Clause Classification:** {clause.category}\n\n"
                f"**Exact Provision:**\n\"{clause.text}\"\n\n"
                f"**Analysis & Purpose:**\n"
                f"This provision establishes formal terms regarding {clause.category.lower()}. "
                f"In commercial agreements, this clause governs the rights, obligations, and risk allocation "
                f"between the parties with respect to operational performance, liability exposure, and compliance."
            )
            records.append(
                ConversationalSFTRecord(
                    record_id=rec_id,
                    contract_id=contract.contract_id,
                    task_type="clause_extraction",
                    split=contract.split,
                    messages=[
                        ChatTurn(role="system", content=SYSTEM_PROMPT_CGL),
                        ChatTurn(role="user", content=user_msg),
                        ChatTurn(role="assistant", content=asst_msg),
                    ],
                    metadata={"category": clause.category, "contract_id": contract.contract_id},
                )
            )
        return records

    def build_risk_flagging_dialogue(self, contract: ContractRecord) -> List[ConversationalSFTRecord]:
        """Creates dialogues evaluating contractual risk priority and legal implications."""
        records: List[ConversationalSFTRecord] = []
        for clause in contract.clauses:
            rec_id = f"sft_risk_{clause.clause_id}"
            # Determine appropriate review priority based on category
            if clause.category in ["Limitation of Liability", "Indemnification", "Termination and Renewal"]:
                priority = "High"
            elif clause.category in ["Exclusivity", "Non-Compete and Non-Solicitation", "Intellectual Property Ownership"]:
                priority = "Medium"
            else:
                priority = "Low"

            user_msg = (
                f"Please conduct a risk review of the following clause from {contract.title}:\n\n"
                f"\"{clause.text}\"\n\n"
                f"Evaluate the review priority, potential commercial concern, affected party, and suggested points for attorney review."
            )
            asst_msg = (
                f"### Contract Clause Risk Assessment\n\n"
                f"1. **Clause Name:** {clause.category}\n"
                f"2. **Exact Provision:** \"{clause.text}\"\n"
                f"3. **Review Priority:** **{priority}** (Advisory review priority; not a definitive legal conclusion)\n"
                f"4. **Potential Concern:** Provisions in this category directly impact financial exposure, operational freedom, or ongoing rights.\n"
                f"5. **Commercial Implications:** In the event of a dispute or default, the terms dictate maximum recoverable damages or termination conditions.\n"
                f"6. **Affected Party:** Both commercial parties; particularly sensitive for the performing/paying party.\n"
                f"7. **Points for Reviewing Lawyer:**\n"
                f"   - Is the liability cap mutual and proportionate to the deal value?\n"
                f"   - Are notice cure periods adequately defined before termination takes effect?"
            )
            records.append(
                ConversationalSFTRecord(
                    record_id=rec_id,
                    contract_id=contract.contract_id,
                    task_type="risk_flagging",
                    split=contract.split,
                    messages=[
                        ChatTurn(role="system", content=SYSTEM_PROMPT_CGL),
                        ChatTurn(role="user", content=user_msg),
                        ChatTurn(role="assistant", content=asst_msg),
                    ],
                    metadata={"priority": priority, "category": clause.category},
                )
            )
        return records

    def build_retrieval_qa_dialogue(self, pair: RetrievalPair) -> ConversationalSFTRecord:
        """Creates retrieval-grounded conversational Q&A records from ACORD pairs."""
        user_msg = f"Question about contract: {pair.query_text}"
        asst_msg = (
            f"Based on the relevant provision of the agreement:\n\n"
            f"> \"{pair.clause_text}\"\n\n"
            f"**Answer:** {pair.clause_text}\n\n"
            f"*(Source: Contract Clause — Relevance: High. Please verify with CGL legal counsel for transaction-specific nuances.)*"
        )
        return ConversationalSFTRecord(
            record_id=f"sft_qa_{pair.pair_id}",
            contract_id=pair.contract_id,
            task_type="grounded_qa",
            split=pair.split,
            messages=[
                ChatTurn(role="system", content=SYSTEM_PROMPT_CGL),
                ChatTurn(role="user", content=user_msg),
                ChatTurn(role="assistant", content=asst_msg),
            ],
            metadata={"query_id": pair.query_id, "relevance": pair.relevance_score},
        )

    def build_nli_verification_dialogue(self, nli: NLIRecord) -> ConversationalSFTRecord:
        """Creates contract claim verification dialogues from ContractNLI."""
        user_msg = (
            f"Statement to verify: \"{nli.hypothesis}\"\n\n"
            f"Does the contract support, contradict, or remain silent on this statement? Cite supporting evidence if present."
        )
        if nli.label == "entailment":
            asst_msg = (
                f"**Assessment:** Supported by the contract (Entailment).\n\n"
                f"**Supporting Evidence:**\n\"{nli.evidence_text}\"\n\n"
                f"**Explanation:** The contractual text directly affirms the statement."
            )
        elif nli.label == "contradiction":
            asst_msg = (
                f"**Assessment:** Contradicted by the contract (Contradiction).\n\n"
                f"**Contradictory Provision:**\n\"{nli.evidence_text}\"\n\n"
                f"**Explanation:** The express wording in the agreement directly opposes the given statement."
            )
        else:
            asst_msg = (
                f"**Assessment:** Not Mentioned (Neutral / Silent).\n\n"
                f"**Explanation:** A comprehensive review of the provided contractual text shows no explicit provision addressing this topic. "
                f"Qualified legal counsel should confirm whether standard industry terms or implied covenants apply."
            )

        return ConversationalSFTRecord(
            record_id=f"sft_nli_{nli.record_id}",
            contract_id=nli.contract_id,
            task_type="claim_verification",
            split=nli.split,
            messages=[
                ChatTurn(role="system", content=SYSTEM_PROMPT_CGL),
                ChatTurn(role="user", content=user_msg),
                ChatTurn(role="assistant", content=asst_msg),
            ],
            metadata={"label": nli.label},
        )

    def build_multi_turn_dialogue(self, contract: ContractRecord) -> Optional[ConversationalSFTRecord]:
        """Creates multi-turn contextual review conversation."""
        if len(contract.clauses) < 2:
            return None
        c1, c2 = contract.clauses[0], contract.clauses[1]
        turn1_user = f"In {contract.title}, what are the termination conditions?"
        turn1_asst = (
            f"According to the agreement:\n\n"
            f"> \"{c1.text}\"\n\n"
            f"Either party possesses the right to terminate under the stipulated conditions. Would you like me to analyze liability or indemnification terms associated with this termination?"
        )
        turn2_user = "Yes, what does the contract state regarding liability or indemnification?"
        turn2_asst = (
            f"Regarding liability and risk protection:\n\n"
            f"> \"{c2.text}\"\n\n"
            f"This {c2.category.lower()} provision bounds the parties' exposure. Be advised to have legal counsel review whether the monetary liability caps cover indirect damages."
        )

        return ConversationalSFTRecord(
            record_id=f"sft_multiturn_{contract.contract_id}",
            contract_id=contract.contract_id,
            task_type="multi_turn_review",
            split=contract.split,
            messages=[
                ChatTurn(role="system", content=SYSTEM_PROMPT_CGL),
                ChatTurn(role="user", content=turn1_user),
                ChatTurn(role="assistant", content=turn1_asst),
                ChatTurn(role="user", content=turn2_user),
                ChatTurn(role="assistant", content=turn2_asst),
            ],
            metadata={"num_turns": 2},
        )

    def export_sft_splits(self, records: List[ConversationalSFTRecord]) -> Dict[str, str]:
        """Exports records into train.jsonl, val.jsonl, and test.jsonl."""
        splits: Dict[str, List[ConversationalSFTRecord]] = {"train": [], "val": [], "test": []}
        for rec in records:
            sp = rec.split if rec.split in splits else "train"
            splits[sp].append(rec)

        paths = {}
        for sp_name, sp_records in splits.items():
            path = os.path.join(self.output_dir, f"{sp_name}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for r in sp_records:
                    f.write(r.to_jsonl_line() + "\n")
            paths[sp_name] = path

        return paths
