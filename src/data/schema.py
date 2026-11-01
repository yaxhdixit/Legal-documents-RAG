"""
Standardized Data Schemas for CUAD, ACORD, ContractNLI, and SFT Records.
Preserves CC BY 4.0 license attribution and strict contract-level isolation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import json


@dataclass
class DatasetProvenance:
    dataset_name: str
    official_url: str
    license: str = "CC BY 4.0"
    attribution: str = ""
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# 14 Standardized Core CUAD Categories mapped to CGL Review Scope
STANDARD_CLAUSE_CATEGORIES = [
    "Termination and Renewal",
    "Limitation of Liability",
    "Indemnification",
    "Confidentiality",
    "Intellectual Property Ownership",
    "Governing Law",
    "Dispute Resolution and Arbitration",
    "Payment Obligations",
    "Exclusivity",
    "Assignment and Change of Control",
    "Non-Compete and Non-Solicitation",
    "Representations and Warranties",
    "Force Majeure",
    "Data Protection and Security",
]


@dataclass
class ClauseAnnotation:
    clause_id: str
    category: str
    text: str
    start_char: int = -1
    end_char: int = -1
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContractRecord:
    contract_id: str
    title: str
    full_text: str
    clauses: List[ClauseAnnotation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_dataset: str = "CUAD"
    provenance: DatasetProvenance = field(
        default_factory=lambda: DatasetProvenance(
            dataset_name="CUAD",
            official_url="https://huggingface.co/datasets/theatticusproject/cuad",
            attribution="The Atticus Project, CUAD under CC BY 4.0",
        )
    )
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class RetrievalPair:
    pair_id: str
    query_id: str
    query_text: str
    contract_id: str
    clause_text: str
    relevance_score: float  # e.g., 0 to 3 scale in ACORD
    clause_category: Optional[str] = None
    source_dataset: str = "ACORD"
    provenance: DatasetProvenance = field(
        default_factory=lambda: DatasetProvenance(
            dataset_name="ACORD",
            official_url="https://huggingface.co/datasets/theatticusproject/acord",
            attribution="The Atticus Project, ACORD under CC BY 4.0",
        )
    )
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NLIRecord:
    record_id: str
    contract_id: str
    hypothesis: str
    label: str  # "entailment", "contradiction", "neutral"
    evidence_text: Optional[str] = None
    contract_text: Optional[str] = None
    source_dataset: str = "ContractNLI"
    provenance: DatasetProvenance = field(
        default_factory=lambda: DatasetProvenance(
            dataset_name="ContractNLI",
            official_url="https://stanfordnlp.github.io/contract-nli/",
            attribution="Stanford NLP Group, ContractNLI under CC BY 4.0",
        )
    )
    split: str = "train"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChatTurn:
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConversationalSFTRecord:
    record_id: str
    contract_id: str
    task_type: str  # "clause_extraction", "risk_flagging", "explanation", "grounded_qa", "missing_clause"
    split: str
    messages: List[ChatTurn]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_line(self) -> str:
        return json.dumps({
            "id": self.record_id,
            "contract_id": self.contract_id,
            "task_type": self.task_type,
            "split": self.split,
            "messages": [asdict(m) for m in self.messages],
            "metadata": self.metadata,
        })
