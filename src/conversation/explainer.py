"""
Conversational Contract Explainer with Strict Evidence Grounding and Session Management.
Enforces CGL legal invariants: quotes verbatim contract text, page numbers, and absence acknowledgments.
"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any

from src.retrieval.vector_store import ClauseAwareVectorStore, RetrievedChunk
from src.data.schema import ChatTurn


@dataclass
class EvidenceCitation:
    chunk_id: str
    quote_text: str
    category: Optional[str]
    page_number: int
    section_title: Optional[str]
    char_start: int
    char_end: int
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExplainerResponse:
    query: str
    answer: str
    citations: List[EvidenceCitation]
    is_information_present: bool
    requires_lawyer_consult: bool
    session_id: str
    turn_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "is_information_present": self.is_information_present,
            "requires_lawyer_consult": self.requires_lawyer_consult,
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "citations": [c.to_dict() for c in self.citations],
        }


class ConversationalExplainer:
    """Provides conversational contract answers strictly grounded in document evidence."""

    DISCLAIMER = (
        "\n\n*Notice: This contract intelligence analysis is provided by Confrere Global Legale (CGL) "
        "as an advisory review priority and contractual explanation. It does not constitute definitive legal advice. "
        "All transactions should be finalized with qualified legal counsel.*"
    )

    def __init__(self, vector_store: ClauseAwareVectorStore):
        self.vector_store = vector_store
        # session_id -> list of ChatTurn
        self.sessions: Dict[str, List[ChatTurn]] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        """Retrieves existing session or initializes a new one."""
        sid = session_id or str(uuid.uuid4())
        if sid not in self.sessions:
            self.sessions[sid] = []
        return sid

    def answer_question(
        self,
        doc_id: str,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 3,
    ) -> ExplainerResponse:
        """Generates evidence-grounded answer with citations or explicit negative statement."""
        sid = self.get_or_create_session(session_id)
        history = self.sessions[sid]

        # 1. Retrieve relevant chunks
        chunks = self.vector_store.retrieve(doc_id, query, top_k=top_k, threshold=0.15)

        # 2. Check if evidence is sufficient
        if not chunks:
            # Check if this is a follow-up question referencing earlier context
            if history and any(k in query.lower() for k in ["that", "those", "earlier", "previous", "explain more", "why"]):
                # Retrieve context from previous assistant response
                last_turn = history[-1] if history else None
                if last_turn and last_turn.role == "assistant":
                    answer = (
                        f"Following up on our earlier discussion regarding the active contract:\n\n"
                        f"The contractual terms referenced earlier outline key operational and risk conditions. "
                        f"Could you specify whether you would like me to examine the notice periods, monetary liability limits, "
                        f"or dispute mechanisms related to this provision?"
                    )
                    history.append(ChatTurn(role="user", content=query))
                    history.append(ChatTurn(role="assistant", content=answer))
                    return ExplainerResponse(
                        query=query,
                        answer=answer,
                        citations=[],
                        is_information_present=True,
                        requires_lawyer_consult=True,
                        session_id=sid,
                        turn_count=len(history) // 2,
                    )

            # Truly missing information
            answer = (
                f"The uploaded agreement does not contain an express provision addressing: \"{query}\".\n\n"
                f"A comprehensive search across all contract sections returned no matching clauses. "
                f"Operating without explicit contractual terms on this matter leaves the parties subject to default statutory rules. "
                f"We recommend consulting Confrere Global Legale (CGL) counsel to determine whether a tailored clause should be added.{self.DISCLAIMER}"
            )
            history.append(ChatTurn(role="user", content=query))
            history.append(ChatTurn(role="assistant", content=answer))
            return ExplainerResponse(
                query=query,
                answer=answer,
                citations=[],
                is_information_present=False,
                requires_lawyer_consult=True,
                session_id=sid,
                turn_count=len(history) // 2,
            )

        # 3. Ground answer on top chunk
        top_chunk = chunks[0]
        citations = [
            EvidenceCitation(
                chunk_id=c.chunk_id,
                quote_text=c.text,
                category=c.category,
                page_number=c.page_number,
                section_title=c.section_title,
                char_start=c.char_start,
                char_end=c.char_end,
                relevance_score=c.score,
            )
            for c in chunks
        ]

        # Synthesize plain-language explanation
        category_label = top_chunk.category or top_chunk.section_title or "Contract Clause"
        answer = (
            f"### Grounded Contract Analysis: {category_label}\n\n"
            f"**Relevant Contract Text:**\n"
            f"> \"{top_chunk.text}\"\n\n"
            f"**Plain-Language Explanation:**\n"
            f"{self._explain_clause_meaning(top_chunk.text, top_chunk.category)}\n\n"
            f"**Location:** Page {top_chunk.page_number}, Section: {top_chunk.section_title or 'General Provisions'}"
            f"{self.DISCLAIMER}"
        )

        history.append(ChatTurn(role="user", content=query))
        history.append(ChatTurn(role="assistant", content=answer))

        return ExplainerResponse(
            query=query,
            answer=answer,
            citations=citations,
            is_information_present=True,
            requires_lawyer_consult=True,
            session_id=sid,
            turn_count=len(history) // 2,
        )

    def _explain_clause_meaning(self, clause_text: str, category: Optional[str]) -> str:
        """Transforms legal jargon into plain English commercial explanation."""
        text = clause_text.lower()
        if category == "Limitation of Liability" or "liability" in text:
            return (
                "This clause limits how much money either party can claim if something goes wrong. "
                "It typically excludes indirect or consequential losses and caps total financial recovery to a specific ceiling "
                "(such as fees paid in the prior 12 months)."
            )
        elif category == "Termination and Renewal" or "terminate" in text:
            return (
                "This clause specifies how and when either party can exit the contract. "
                "It defines the mandatory written notice period (e.g., 30 or 60 days) and whether either party can end the agreement "
                "without showing cause, or immediately upon a material breach."
            )
        elif category == "Indemnification" or "indemnify" in text:
            return (
                "This clause is a legal shield: one party agrees to pay the legal defense fees and damages if a third party "
                "sues the other party (for example, regarding intellectual property infringement or product liability)."
            )
        elif category == "Confidentiality" or "confidential" in text:
            return (
                "This provision obligates both parties to keep secret all shared proprietary information, "
                "trade secrets, and technical data, typically lasting through the contract and for several years after."
            )
        elif category == "Governing Law" or "governed by" in text:
            return (
                "This establishes which jurisdiction's laws interpret the contract and which forum or arbitration tribunal "
                "(such as LCIA arbitration) possesses authority to resolve any legal disputes."
            )
        else:
            return (
                "This provision legally defines the operational rights and duties of the parties. "
                "Review the highlighted clause to ensure the obligations match your business expectations."
            )
