"""
Contract Risk Detection and Advisory Review Prioritization Engine.
Enforces CGL legal invariants: review priority framing, exact evidence quotes, and targeted lawyer questions.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import re

from src.data.schema import ClauseAnnotation, STANDARD_CLAUSE_CATEGORIES
from src.ingest.parser import ParsedDocument


@dataclass
class RiskFinding:
    finding_id: str
    clause_id: Optional[str]
    clause_name: str
    exact_text: str
    risk_priority: str  # "High", "Medium", "Low", "Requires Review"
    risk_type: str  # "problematic", "ambiguous", "missing", "unfavorable", "contextual"
    concern_explanation: str
    commercial_implications: str
    affected_party: str
    evidence_quote: str
    page_number: Optional[int]
    section_title: Optional[str]
    lawyer_review_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContractRiskReport:
    doc_id: str
    total_clauses_identified: int
    findings: List[RiskFinding]
    high_priority_count: int
    medium_priority_count: int
    low_priority_count: int
    requires_review_count: int
    missing_provisions: List[str]
    overall_risk_summary: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "total_clauses_identified": self.total_clauses_identified,
            "high_priority_count": self.high_priority_count,
            "medium_priority_count": self.medium_priority_count,
            "low_priority_count": self.low_priority_count,
            "requires_review_count": self.requires_review_count,
            "missing_provisions": self.missing_provisions,
            "overall_risk_summary": self.overall_risk_summary,
            "findings": [f.to_dict() for f in self.findings],
        }


class ContractRiskEngine:
    """Analyzes identified clauses and overall document to detect commercial risk patterns."""

    def analyze_contract(self, parsed_doc: ParsedDocument, clauses: List[ClauseAnnotation]) -> ContractRiskReport:
        """Executes full risk evaluation on extracted clauses and document text."""
        findings: List[RiskFinding] = []
        identified_categories = {c.category for c in clauses}
        finding_counter = 1

        # 1. Analyze each identified clause for specific risk indicators
        for clause in clauses:
            finding = self._evaluate_clause_risk(clause, finding_counter)
            if finding:
                findings.append(finding)
                finding_counter += 1

        # 2. Check for critical missing provisions across standard CUAD categories
        missing_provisions: List[str] = []
        essential_categories = [
            "Limitation of Liability",
            "Termination and Renewal",
            "Governing Law",
            "Dispute Resolution and Arbitration",
            "Confidentiality",
            "Data Protection and Security",
        ]
        for essential in essential_categories:
            if essential not in identified_categories and not self._text_contains_category(parsed_doc.full_text, essential):
                missing_provisions.append(essential)
                findings.append(
                    RiskFinding(
                        finding_id=f"risk_{finding_counter:03d}",
                        clause_id=None,
                        clause_name=essential,
                        exact_text="[Clause completely absent from agreement]",
                        risk_priority="Requires Review",
                        risk_type="missing",
                        concern_explanation=(
                            f"The agreement does not contain an express {essential} provision. "
                            f"Operating without standard {essential.lower()} terms leaves the parties subject to statutory default rules or common law principles."
                        ),
                        commercial_implications=(
                            f"Absence of express {essential.lower()} terms creates unpredictable exposure in event of contract breach, dispute, or regulatory audit."
                        ),
                        affected_party="Both commercial parties",
                        evidence_quote="No clause matching category identified across entire document text.",
                        page_number=None,
                        section_title=None,
                        lawyer_review_questions=[
                            f"Should a standard CGL-approved {essential} clause be inserted?",
                            f"Does the nature of the transaction necessitate specific statutory protections?",
                        ],
                    )
                )
                finding_counter += 1

        # Calculate counts
        high_cnt = sum(1 for f in findings if f.risk_priority == "High")
        med_cnt = sum(1 for f in findings if f.risk_priority == "Medium")
        low_cnt = sum(1 for f in findings if f.risk_priority == "Low")
        review_cnt = sum(1 for f in findings if f.risk_priority == "Requires Review")

        overall_summary = (
            f"Review identified {len(clauses)} contractual clauses and {len(findings)} review priority items "
            f"({high_cnt} High, {med_cnt} Medium, {low_cnt} Low, {review_cnt} Requires Review). "
            f"{f'Attention required on {len(missing_provisions)} missing core provisions.' if missing_provisions else 'Core legal provisions are present.'} "
            f"All findings represent advisory review priorities for Confrere Global Legale (CGL) counsel."
        )

        return ContractRiskReport(
            doc_id=parsed_doc.doc_id,
            total_clauses_identified=len(clauses),
            findings=findings,
            high_priority_count=high_cnt,
            medium_priority_count=med_cnt,
            low_priority_count=low_cnt,
            requires_review_count=review_cnt,
            missing_provisions=missing_provisions,
            overall_risk_summary=overall_summary,
        )

    def _evaluate_clause_risk(self, clause: ClauseAnnotation, index: int) -> Optional[RiskFinding]:
        """Examines clause wording for specific risk types and assigns priority with rich contextual lawyer review guidance."""
        text = clause.text.lower()
        cat = clause.category
        clause_snippet = (clause.text[:95] + "...") if len(clause.text) > 95 else clause.text

        # Limitation of Liability
        if cat == "Limitation of Liability":
            if "solely" in text or "unlimited" in text or "exclusive remedy" in text:
                return RiskFinding(
                    finding_id=f"risk_{index:03d}",
                    clause_id=clause.clause_id,
                    clause_name=cat,
                    exact_text=clause.text,
                    risk_priority="High",
                    risk_type="problematic",
                    concern_explanation="Contains unilateral liability caps or exclusive remedy language that may severely restrict damage recovery upon vendor breach.",
                    commercial_implications="In the event of defective service delivery, gross negligence, or data compromise, recovery may be capped below actual incurred loss.",
                    affected_party="Client / Buyer",
                    evidence_quote=clause.text[:200],
                    page_number=clause.page_number,
                    section_title=clause.section_title,
                    lawyer_review_questions=[
                        f"Review provision snippet: \"{clause_snippet}\" — Verify whether the remedy exclusivity precludes common law statutory damages.",
                        "Action: Negotiate an express super-cap or carve-out excluding gross negligence, willful misconduct, confidentiality breaches, and IP indemnity from this liability limitation.",
                        "Action: Ensure the aggregate liability ceiling is reciprocal and calculated on actual trailing fees rather than an arbitrary nominal limit.",
                    ],
                )
            else:
                cap_match = re.search(r"(?:not exceed|capped at|limited to)\s+([^.;]+)", text, re.IGNORECASE)
                cap_str = cap_match.group(1).strip() if cap_match else "12 months trailing fees"
                return RiskFinding(
                    finding_id=f"risk_{index:03d}",
                    clause_id=clause.clause_id,
                    clause_name=cat,
                    exact_text=clause.text,
                    risk_priority="Medium",
                    risk_type="contextual",
                    concern_explanation=f"Aggregate liability is capped to {cap_str}. Standard commercial structure, but requires validation against transaction risk profile.",
                    commercial_implications="Constrains maximum recoverable financial compensation to the stated fee multiplier regardless of actual business interruption costs.",
                    affected_party="Both Parties",
                    evidence_quote=clause.text[:200],
                    page_number=clause.page_number,
                    section_title=clause.section_title,
                    lawyer_review_questions=[
                        f"Review identified ceiling: \"{clause_snippet}\" — Confirm whether {cap_str} provides proportionate coverage relative to contract deal size.",
                        "Action: Request a multiplier adjustment (e.g. 2x annual contract value) if initial year fees are discounted or phased.",
                        "Action: Confirm that third-party intellectual property indemnification claims remain completely uncapped.",
                    ],
                )

        # Indemnification
        if cat == "Indemnification":
            is_unilateral = ("provider shall indemnify" in text and "client shall indemnify" not in text) or ("client shall indemnify" in text and "provider shall indemnify" not in text)
            indemnitor = "Provider" if "provider shall indemnify" in text else ("Client" if "client shall indemnify" in text else "Specified Party")
            return RiskFinding(
                finding_id=f"risk_{index:03d}",
                clause_id=clause.clause_id,
                clause_name=cat,
                exact_text=clause.text,
                risk_priority="High" if is_unilateral else "Medium",
                risk_type="unfavorable" if is_unilateral else "contextual",
                concern_explanation=f"Imposes direct third-party defense and liability obligations on {indemnitor} without standard reciprocal protections.",
                commercial_implications="Indemnifying party assumes responsibility for external legal defense costs, legal representation choices, and final judgment or settlement sums.",
                affected_party=f"{indemnitor} / Performing Party",
                evidence_quote=clause.text[:200],
                page_number=clause.page_number,
                section_title=clause.section_title,
                lawyer_review_questions=[
                    f"Review indemnification scope: \"{clause_snippet}\" — Confirm whether indemnity triggers only upon final unappealable judicial determination or mere third-party allegation.",
                    "Action: Insert express conditions precedent requiring prompt written notice, right to control defense, and prohibition against non-monetary settlements without prior consent.",
                    f"Action: {'Negotiate mutual reciprocal indemnity for client-provided materials.' if is_unilateral else 'Confirm coverage boundaries for IP infringement and negligence.'}",
                ],
            )

        # Termination and Renewal
        if cat == "Termination and Renewal":
            notice_match = re.search(r"(\d+)\s*(?:day|business day|month)", text)
            notice_val = notice_match.group(0) if notice_match else "immediate / undefined"
            if "without cause" in text:
                return RiskFinding(
                    finding_id=f"risk_{index:03d}",
                    clause_id=clause.clause_id,
                    clause_name=cat,
                    exact_text=clause.text,
                    risk_priority="Medium",
                    risk_type="unfavorable",
                    concern_explanation=f"Permits unilateral termination for convenience upon {notice_val} notice, introducing contractual continuity uncertainty.",
                    commercial_implications="The non-terminating party risks premature contract cancellation without recovery of front-loaded onboarding or infrastructure expenses.",
                    affected_party="Performing party reliant on long-term commitment",
                    evidence_quote=clause.text[:200],
                    page_number=clause.page_number,
                    section_title=clause.section_title,
                    lawyer_review_questions=[
                        f"Review convenience notice: \"{clause_snippet}\" — Verify whether {notice_val} notice provides sufficient runway for operational transition or resource re-allocation.",
                        "Action: Insert an early termination fee or require reimbursement for unamortized setup and licensing costs if terminated before 12 months.",
                        "Action: Ensure accrued payment obligations and ongoing confidentiality commitments expressly survive termination.",
                    ],
                )
            else:
                return RiskFinding(
                    finding_id=f"risk_{index:03d}",
                    clause_id=clause.clause_id,
                    clause_name=cat,
                    exact_text=clause.text,
                    risk_priority="High" if "immediate" in text else "Medium",
                    risk_type="contextual",
                    concern_explanation="Defines default and breach termination procedures, requiring verification of mandatory cure periods.",
                    commercial_implications="Determines the exact standard required to dissolve the agreement and invoke post-termination remedies.",
                    affected_party="Both Parties",
                    evidence_quote=clause.text[:200],
                    page_number=clause.page_number,
                    section_title=clause.section_title,
                    lawyer_review_questions=[
                        f"Review default terms: \"{clause_snippet}\" — Verify that a minimum 30-day written cure period is required prior to exercising termination for cause.",
                        "Action: Restrict 'material breach' definitions to objective non-performance milestones to avoid premature termination threats.",
                    ],
                )

        # Exclusivity or Non-Compete
        if cat in ["Exclusivity", "Non-Compete and Non-Solicitation"]:
            return RiskFinding(
                finding_id=f"risk_{index:03d}",
                clause_id=clause.clause_id,
                clause_name=cat,
                exact_text=clause.text,
                risk_priority="High",
                risk_type="problematic",
                concern_explanation=f"Restricts commercial freedom through express {cat.lower()} covenants in the target territory or industry vertical.",
                commercial_implications="Precludes engagement with alternative vendors or clients and restricts lateral recruitment of specialized personnel.",
                affected_party="Restricted Commercial Party",
                evidence_quote=clause.text[:200],
                page_number=clause.page_number,
                section_title=clause.section_title,
                lawyer_review_questions=[
                    f"Review restrictive covenant: \"{clause_snippet}\" — Validate territorial and temporal scope under applicable trade and competition laws.",
                    "Action: Insert clear exclusions permitting general, non-directed public job recruitment campaigns and legacy preexisting customer accounts.",
                    "Action: Require mutual exclusivity or commensurate minimum revenue commitments to balance the operational lock-in.",
                ],
            )

        # Ambiguous standards like "reasonable efforts" without definition
        if "reasonable efforts" in text or "best efforts" in text:
            return RiskFinding(
                finding_id=f"risk_{index:03d}",
                clause_id=clause.clause_id,
                clause_name=cat,
                exact_text=clause.text,
                risk_priority="Requires Review",
                risk_type="ambiguous",
                concern_explanation="Incorporates subjective performance thresholds ('reasonable efforts' or 'best efforts') without defined objective SLA benchmarks.",
                commercial_implications="Subjective diligence standards increase susceptibility to dispute over whether adequate effort was expended in project delivery.",
                affected_party="Both Parties",
                evidence_quote=clause.text[:200],
                page_number=clause.page_number,
                section_title=clause.section_title,
                lawyer_review_questions=[
                    f"Review subjective standard: \"{clause_snippet}\" — Replace 'reasonable efforts' with concrete deliverable milestones, timetables, and measurable KPIs.",
                    "Action: Specify the financial expenditure ceiling required under any 'efforts' covenant so compliance does not demand commercially unreasonable expense.",
                ],
            )

        # Default low risk for standard operational provisions
        return RiskFinding(
            finding_id=f"risk_{index:03d}",
            clause_id=clause.clause_id,
            clause_name=cat,
            exact_text=clause.text,
            risk_priority="Low",
            risk_type="contextual",
            concern_explanation=f"Standard commercial governance provision regarding {cat.lower()}.",
            commercial_implications="Establishes customary operational guidelines and procedural mechanisms between the contracting parties.",
            affected_party="Both Parties",
            evidence_quote=clause.text[:200],
            page_number=clause.page_number,
            section_title=clause.section_title,
            lawyer_review_questions=[
                f"Review operational clause: \"{clause_snippet}\" — Confirm that notification addresses, banking details, and governance procedures align with internal operations.",
            ],
        )

    def _text_contains_category(self, text: str, category: str) -> bool:
        """Heuristic check if category keywords exist in raw text."""
        keywords = {
            "Limitation of Liability": ["limitation of liability", "liability shall not exceed", "consequential damages"],
            "Termination and Renewal": ["termination", "term and termination", "either party may terminate"],
            "Governing Law": ["governing law", "governed by the laws", "jurisdiction"],
            "Dispute Resolution and Arbitration": ["arbitration", "dispute resolution", "lcia", "arbitrator"],
            "Confidentiality": ["confidentiality", "confidential information", "non-disclosure"],
            "Data Protection and Security": ["data protection", "gdpr", "security measures", "personal data"],
        }
        for kw in keywords.get(category, [category.lower()]):
            if kw in text.lower():
                return True
        return False

    def generate_contract_overview(self, parsed_doc: ParsedDocument, clauses: List[ClauseAnnotation]) -> Dict[str, Any]:
        """Synthesizes high-level executive context, key deal parameters, and tailored questions strictly from document."""
        text = parsed_doc.full_text
        lower_text = text.lower()

        # Map identified clauses by category
        category_clauses: Dict[str, List[ClauseAnnotation]] = {}
        for c in clauses:
            category_clauses.setdefault(c.category, []).append(c)

        # 1. Term Duration
        term_val = "Not specified in agreement"
        term_scope = " ".join([c.text for c in category_clauses.get("Termination and Renewal", [])]) or text
        term_match = re.search(r"(?:initial\s+term\s+of|term\s+of|period\s+of|for\s+a\s+period\s+of)\s+([A-Za-z0-9\(\)\s]{1,40}?(?:years?|months?|weeks?|days?))", term_scope, re.IGNORECASE)
        if term_match:
            term_val = term_match.group(1).strip()
        elif "perpetual" in lower_text or "in perpetuity" in lower_text:
            term_val = "Perpetual"
        elif "at-will" in lower_text or "at will" in lower_text:
            term_val = "At-will / Indefinite"
        elif "Termination and Renewal" in category_clauses:
            term_val = "Ongoing / Subject to Termination"

        # 2. Termination Notice
        notice_val = "Not specified (Immediate / At-will)"
        notice_match = re.search(r"(\d+|[A-Za-z]+)\s*(?:\([^\)]+\)\s*)?(?:business\s+|calendar\s+)?(?:day|month|week)s?\s+(?:prior\s+)?written\s+notice", term_scope, re.IGNORECASE) or \
                       re.search(r"upon\s+(\d+|[A-Za-z]+)\s*(?:\([^\)]+\)\s*)?(?:day|month)s?\s+notice", term_scope, re.IGNORECASE) or \
                       re.search(r"notice\s+of\s+not\s+less\s+than\s+([A-Za-z0-9\(\)\s]{1,30}?(?:days?|months?))", term_scope, re.IGNORECASE)
        if notice_match:
            notice_val = f"{notice_match.group(1).strip()} written notice"
        elif "without notice" in term_scope.lower() or "immediate" in term_scope.lower():
            notice_val = "Immediate (Without Notice)"
        elif "Termination and Renewal" in category_clauses:
            notice_val = "Written notice required (Duration unstated)"

        # 3. Liability Cap
        cap_val = "No liability cap found (Uncapped Risk)"
        cap_scope = " ".join([c.text for c in category_clauses.get("Limitation of Liability", [])]) or text
        money_cap = re.search(r"(?:not\s+exceed|capped\s+at|limited\s+to|maximum\s+liability\s+of)\s*[:\s]*(\$\s*[\d,]+(?:\.\d+)?|\€\s*[\d,]+|£\s*[\d,]+|[\d,]+\s*(?:USD|EUR|GBP|INR))", cap_scope, re.IGNORECASE)
        fees_cap = re.search(r"(?:not\s+exceed|capped\s+at|limited\s+to|maximum\s+of)\s*([A-Za-z0-9\(\)\s]{1,60}?(?:fees|amount|charges)\s+(?:paid|payable)[^.;,\n]*)", cap_scope, re.IGNORECASE)
        gen_cap = re.search(r"(?:total\s+aggregate\s+liability[^.;\n]+?shall\s+not\s+exceed\s+)([^.;\n]+)", cap_scope, re.IGNORECASE)

        if money_cap:
            cap_val = money_cap.group(1).strip()
        elif fees_cap:
            cap_val = fees_cap.group(1).strip()
        elif gen_cap:
            cap_val = gen_cap.group(1).strip()
        elif "Limitation of Liability" in category_clauses:
            cap_val = "Mutual liability capped (See clause text)"

        # 4. Governing Law & Dispute
        gov_val = "Not specified in agreement"
        gov_scope = " ".join([c.text for c in category_clauses.get("Governing Law", [])]) or text
        gov_match = re.search(r"(?:governing\s+law|jurisdiction)\s*[:\-]\s*(?:the\s+laws\s+of\s+)?(?:the\s+State\s+of\s+|State\s+of\s+)?([A-Za-z\s,–—]{2,50}?)(?:\.|\;|\n|$)", gov_scope, re.IGNORECASE) or \
                    re.search(r"governed\s+by\s+(?:and\s+construed\s+in\s+accordance\s+with\s+)?(?:the\s+laws\s+of\s+)?(?:the\s+State\s+of\s+|State\s+of\s+)?([A-Za-z\s,–—]{2,50}?)(?:\.|\;|\n|without\s+regard|and\s+each\s+party)", gov_scope, re.IGNORECASE) or \
                    re.search(r"laws\s+of\s+([A-Za-z\s,–—]{2,50}?)(?:\.|\;|\n|shall)", gov_scope, re.IGNORECASE)
        if gov_match:
            gov_val = gov_match.group(1).strip().rstrip(",").rstrip(" and")
        elif "Governing Law" in category_clauses:
            gov_val = category_clauses["Governing Law"][0].text[:45].strip() + "..."

        arb_val = "Ordinary courts / Unspecified"
        arb_scope = " ".join([c.text for c in category_clauses.get("Dispute Resolution and Arbitration", [])]) or text
        arb_match = re.search(r"(?:arbitration under|resolved by arbitration under|arbitration in accordance with)\s+([^.;\n]+)", arb_scope, re.IGNORECASE)
        if arb_match:
            arb_val = arb_match.group(1).strip()
        elif "arbitration" in arb_scope.lower():
            arb_val = "Binding Arbitration"
        elif "Dispute Resolution and Arbitration" in category_clauses:
            arb_val = category_clauses["Dispute Resolution and Arbitration"][0].text[:45].strip() + "..."

        # 5. Suggested tailored questions based on actual document content
        suggested_questions = [
            "What are the biggest commercial and legal risks in this agreement?",
        ]
        if "Termination and Renewal" in category_clauses:
            suggested_questions.append(f"Can the other party terminate this contract without cause or notice under {notice_val}?")
        else:
            suggested_questions.append("Does this agreement contain an express term duration or termination clause?")

        if "Limitation of Liability" in category_clauses:
            suggested_questions.append(f"How does the liability ceiling ({cap_val[:40]}) protect our business?")
        else:
            suggested_questions.append("Are there any provisions limiting our company's financial liability or exposure?")

        if "Intellectual Property" in category_clauses:
            suggested_questions.append("Who owns intellectual property and software deliverables created under this contract?")
        if "Indemnification" in category_clauses:
            suggested_questions.append("Explain the indemnification clause and third-party infringement defense in simple English.")
        if "Confidentiality" in category_clauses:
            suggested_questions.append("What confidentiality and non-disclosure obligations are enforced?")
        if "exclusive" in lower_text or "exclusivity" in lower_text:
            suggested_questions.append("Are there geographic or product exclusivity restrictions binding the supplier?")
        if "non-solicit" in lower_text or "solicit" in lower_text:
            suggested_questions.append("What restrictions apply to recruiting or soliciting employees?")
        if "data protection" in lower_text or "gdpr" in lower_text:
            suggested_questions.append("What data protection, GDPR, and technical security standards are mandated?")
        if "payment" in lower_text or "net 30" in lower_text:
            suggested_questions.append("What are the payment deadlines and late interest penalties?")

        title_clean = parsed_doc.filename.replace(".txt", "").replace(".pdf", "").replace(".docx", "").replace(".doc", "")
        summary = (
            f"This commercial agreement ({title_clean}) has been analyzed across {len(clauses)} identified clause provisions. "
            f"Term duration is evaluated as {term_val}, with termination notice requirements established as {notice_val}. "
            f"Financial liability posture is identified as {cap_val}, governed under {gov_val} via {arb_val}. "
            f"Legal counsel should review the flagged high and medium risk provisions before execution."
        )

        return {
            "title": title_clean,
            "term_duration": term_val,
            "termination_notice": notice_val,
            "liability_cap": cap_val,
            "governing_law": gov_val,
            "dispute_forum": arb_val,
            "executive_summary": summary,
            "suggested_questions": suggested_questions,
            "total_clauses": len(clauses),
            "total_pages": len(parsed_doc.pages),
        }
