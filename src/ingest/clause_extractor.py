"""
Contract Clause Extractor and Classifier for 14+ CUAD Standardized Categories.
Extracts verbatim clauses with page numbers, section headers, and character span coordinates.
"""

from __future__ import annotations
import re
from typing import List, Dict, Tuple, Optional
from src.data.schema import ClauseAnnotation, STANDARD_CLAUSE_CATEGORIES
from src.ingest.parser import ParsedDocument


class ClauseExtractor:
    """Identifies and classifies commercial clauses using legal syntax rules and pattern matching."""

    # Targeted legal syntax indicators for each standard CUAD category
    CATEGORY_PATTERNS: Dict[str, List[re.Pattern]] = {
        "Termination and Renewal": [
            re.compile(r"\b(?:terminate|termination|renew|renewal|prior written notice|cure period|expire|expiration|without cause)\b", re.IGNORECASE),
            re.compile(r"\b(?:term of this agreement|commence on the effective date|notice period to terminate)\b", re.IGNORECASE),
        ],
        "Limitation of Liability": [
            re.compile(r"\b(?:limitation of liability|in no event shall either party|indirect, incidental|consequential damages|aggregate liability|cap on liability|not exceed the total fees)\b", re.IGNORECASE),
            re.compile(r"\b(?:liquidated damages|punitive damages|liability cap)\b", re.IGNORECASE),
        ],
        "Indemnification": [
            re.compile(r"\b(?:indemnify|indemnification|hold harmless|defend, indemnify|third-party claims|infringement claims)\b", re.IGNORECASE),
            re.compile(r"\b(?:indemnitee|indemnitor|losses, damages, costs and expenses)\b", re.IGNORECASE),
        ],
        "Confidentiality": [
            re.compile(r"\b(?:confidential information|non-disclosure|proprietary information|strict confidence|trade secrets)\b", re.IGNORECASE),
            re.compile(r"\b(?:non-disparagement|obligation of confidentiality)\b", re.IGNORECASE),
        ],
        "Intellectual Property Ownership": [
            re.compile(r"\b(?:intellectual property|ip ownership|work made for hire|all right, title and interest|patent, copyright|license grant|proprietary rights)\b", re.IGNORECASE),
            re.compile(r"\b(?:inventions, discoveries|software license|grant a non-exclusive license)\b", re.IGNORECASE),
        ],
        "Governing Law": [
            re.compile(r"\b(?:governed by|governing law|laws of the state of|laws of england|jurisdiction of the courts of|venue in)\b", re.IGNORECASE),
            re.compile(r"\b(?:construed in accordance with the laws)\b", re.IGNORECASE),
        ],
        "Dispute Resolution and Arbitration": [
            re.compile(r"\b(?:dispute resolution|arbitration|arbitrator|lcia rules|icc rules|american arbitration association|binding arbitration|amicable settlement)\b", re.IGNORECASE),
            re.compile(r"\b(?:mediation|disputes arising out of or in connection)\b", re.IGNORECASE),
        ],
        "Payment Obligations": [
            re.compile(r"\b(?:payment terms|invoiced amounts|net 30|payable within|late payment|interest at|audit rights|royalties|fees)\b", re.IGNORECASE),
            re.compile(r"\b(?:taxes, duties|billing schedule|reimbursement of expenses)\b", re.IGNORECASE),
        ],
        "Exclusivity": [
            re.compile(r"\b(?:exclusive|exclusivity|most favored nation|sole and exclusive|shall not supply competing|right of first refusal|rofr)\b", re.IGNORECASE),
            re.compile(r"\b(?:minimum commitment|exclusively to buyer)\b", re.IGNORECASE),
        ],
        "Assignment and Change of Control": [
            re.compile(r"\b(?:assign this agreement|assignment|change of control|prior written consent|successor and assigns|merger or acquisition)\b", re.IGNORECASE),
            re.compile(r"\b(?:transferable|delegate its duties)\b", re.IGNORECASE),
        ],
        "Non-Compete and Non-Solicitation": [
            re.compile(r"\b(?:non-compete|non-solicitation|solicit or recruit|competing business|covenant not to compete|restrictive covenant)\b", re.IGNORECASE),
            re.compile(r"\b(?:hire any employee|solicit any customer)\b", re.IGNORECASE),
        ],
        "Representations and Warranties": [
            re.compile(r"\b(?:represents and warrants|representation|warranty|warranties|free from defects|as is basis|merchantability|fitness for a particular purpose)\b", re.IGNORECASE),
            re.compile(r"\b(?:covenant not to sue|disclaimer of warranties)\b", re.IGNORECASE),
        ],
        "Force Majeure": [
            re.compile(r"\b(?:force majeure|acts of god|war, terrorism|strikes|beyond reasonable control|epidemic, pandemic|natural disasters)\b", re.IGNORECASE),
            re.compile(r"\b(?:excused from performance|unforeseen events)\b", re.IGNORECASE),
        ],
        "Data Protection and Security": [
            re.compile(r"\b(?:data protection|gdpr|ccpa|personal data|security measures|data breach|confidentiality and security of data|cybersecurity)\b", re.IGNORECASE),
            re.compile(r"\b(?:technical and organizational measures|privacy policy)\b", re.IGNORECASE),
        ],
    }

    def extract_clauses(self, parsed_doc: ParsedDocument) -> List[ClauseAnnotation]:
        """Extracts and classifies clauses across sections of the parsed document."""
        clauses: List[ClauseAnnotation] = []
        clause_counter = 1

        for section in parsed_doc.sections:
            sec_text = section.content
            sec_title = section.title
            
            # Split section into logical paragraphs/sentences
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\.\s+(?=[A-Z])", sec_text) if len(p.strip()) > 30]
            if not paragraphs:
                paragraphs = [sec_text]

            for para in paragraphs:
                detected_cat, confidence = self._classify_text_segment(para, sec_title)
                if detected_cat:
                    # Calculate character offset
                    start_char = parsed_doc.full_text.find(para, section.char_start)
                    if start_char == -1:
                        start_char = section.char_start
                    end_char = start_char + len(para)

                    # Determine page number
                    page_num = section.page_number
                    for p in parsed_doc.pages:
                        if p.char_start <= start_char <= p.char_end:
                            page_num = p.page_number
                            break

                    clauses.append(
                        ClauseAnnotation(
                            clause_id=f"cl_{clause_counter:03d}",
                            category=detected_cat,
                            text=para,
                            start_char=start_char,
                            end_char=end_char,
                            page_number=page_num,
                            section_title=sec_title,
                            confidence=confidence,
                        )
                    )
                    clause_counter += 1

        return clauses

    def _classify_text_segment(self, text: str, section_title: str) -> Tuple[Optional[str], float]:
        """Classifies a text segment by scoring pattern matches against categories."""
        best_category: Optional[str] = None
        best_score = 0.0

        combined = f"{section_title}\n{text}".lower()

        for category, patterns in self.CATEGORY_PATTERNS.items():
            score = 0.0
            # Section title match gives high prior weight
            if any(term in section_title.lower() for term in category.lower().split()):
                score += 2.0

            for pattern in patterns:
                matches = pattern.findall(combined)
                score += len(matches) * 1.5

            if score > best_score and score >= 2.0:
                best_score = score
                best_category = category

        confidence = min(0.99, max(0.60, best_score / 6.0)) if best_category else 0.0
        return best_category, round(confidence, 2)
