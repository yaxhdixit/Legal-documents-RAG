"""
Multi-Format Contract Parser for PDF, DOCX, and TXT files.
Extracts document text preserving page numbers, section hierarchy, and character offsets.
"""

from __future__ import annotations
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


@dataclass
class DocumentPage:
    page_number: int
    text: str
    char_start: int
    char_end: int


@dataclass
class DocumentSection:
    section_id: str
    title: str
    content: str
    page_number: int
    char_start: int
    char_end: int


@dataclass
class ParsedDocument:
    doc_id: str
    filename: str
    file_type: str
    full_text: str
    pages: List[DocumentPage] = field(default_factory=list)
    sections: List[DocumentSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "full_text": self.full_text,
            "char_length": len(self.full_text),
            "num_pages": len(self.pages),
            "num_sections": len(self.sections),
            "metadata": self.metadata,
            "pages": [asdict(p) for p in self.pages],
            "sections": [asdict(s) for s in self.sections],
        }


class ContractParser:
    """Parses PDF, DOCX, and text contract files."""

    SECTION_HEADING_REGEX = re.compile(
        r"^(?:(?:SECTION|ARTICLE|CLAUSE|\d+)\b[.:\s]|(?:[A-Z0-9.\-\s]{3,50}:))(?:\s+[A-Z][A-Za-z0-9\s,\-–—]{2,60})?$",
        re.MULTILINE,
    )

    def parse_file(self, file_path: str, doc_id: Optional[str] = None, original_filename: Optional[str] = None) -> ParsedDocument:
        """Parses document from file path based on extension."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = os.path.basename(original_filename) if original_filename else os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        if not doc_id:
            import hashlib
            doc_id = f"doc_{hashlib.md5(filename.encode('utf-8')).hexdigest()[:8]}"

        if ext == ".pdf":
            return self._parse_pdf(file_path, doc_id, filename)
        elif ext in [".docx", ".doc"]:
            return self._parse_docx(file_path, doc_id, filename)
        elif ext in [".txt", ".md"]:
            return self._parse_text(file_path, doc_id, filename)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats are .pdf, .docx, .txt")

    def _parse_pdf(self, file_path: str, doc_id: str, filename: str) -> ParsedDocument:
        """Extracts text from PDF page by page."""
        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []
        current_offset = 0

        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                p_len = len(page_text)
                pages.append(
                    DocumentPage(
                        page_number=idx + 1,
                        text=page_text,
                        char_start=current_offset,
                        char_end=current_offset + p_len,
                    )
                )
                full_text_parts.append(page_text)
                current_offset += p_len + 2  # plus delimiter
        except Exception as e:
            # Fallback if pypdf encounters an issue
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
            text = raw_bytes.decode("utf-8", errors="ignore")
            pages.append(DocumentPage(page_number=1, text=text, char_start=0, char_end=len(text)))
            full_text_parts.append(text)

        full_text = "\n\n".join(full_text_parts)
        sections = self._extract_sections(full_text, pages)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="pdf",
            full_text=full_text,
            pages=pages,
            sections=sections,
            metadata={"num_pages": len(pages)},
        )

    def _parse_docx(self, file_path: str, doc_id: str, filename: str) -> ParsedDocument:
        """Extracts text from DOCX document paragraphs."""
        pages: List[DocumentPage] = []
        full_text_parts: List[str] = []

        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)
            # Estimate pages ~ 3000 chars per page
            page_size = 3000
            for p_num in range(1, max(2, (len(full_text) // page_size) + 2)):
                start = (p_num - 1) * page_size
                end = min(len(full_text), p_num * page_size)
                if start >= len(full_text) and p_num > 1:
                    break
                pages.append(
                    DocumentPage(
                        page_number=p_num,
                        text=full_text[start:end],
                        char_start=start,
                        char_end=end,
                    )
                )
        except Exception:
            with open(file_path, "rb") as f:
                content = f.read().decode("utf-8", errors="ignore")
            pages.append(DocumentPage(page_number=1, text=content, char_start=0, char_end=len(content)))
            full_text = content

        sections = self._extract_sections(full_text, pages)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="docx",
            full_text=full_text,
            pages=pages,
            sections=sections,
            metadata={"num_pages": len(pages)},
        )

    def _parse_text(self, file_path: str, doc_id: str, filename: str) -> ParsedDocument:
        """Parses plain text document."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            full_text = f.read()

        pages = [DocumentPage(page_number=1, text=full_text, char_start=0, char_end=len(full_text))]
        sections = self._extract_sections(full_text, pages)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="txt",
            full_text=full_text,
            pages=pages,
            sections=sections,
            metadata={"num_pages": 1},
        )

    def parse_string(self, text: str, filename: str = "contract.txt", doc_id: Optional[str] = None) -> ParsedDocument:
        """Directly parses raw text string into a ParsedDocument."""
        if not doc_id:
            import hashlib
            doc_id = f"doc_{hashlib.md5(text.encode('utf-8')).hexdigest()[:8]}"

        # Split into logical pages ~ 2500 characters
        page_size = 2500
        pages: List[DocumentPage] = []
        if len(text) <= page_size:
            pages.append(DocumentPage(page_number=1, text=text, char_start=0, char_end=len(text)))
        else:
            total_pages = (len(text) // page_size) + 1
            for p in range(total_pages):
                c_start = p * page_size
                c_end = min(len(text), (p + 1) * page_size)
                pages.append(DocumentPage(page_number=p + 1, text=text[c_start:c_end], char_start=c_start, char_end=c_end))

        sections = self._extract_sections(text, pages)
        return ParsedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type="txt",
            full_text=text,
            pages=pages,
            sections=sections,
            metadata={"num_pages": len(pages)},
        )

    def _extract_sections(self, text: str, pages: List[DocumentPage]) -> List[DocumentSection]:
        """Detects headings and sections with page mapping using optimized legal heading heuristics."""
        sections: List[DocumentSection] = []
        lines = text.split("\n")
        current_title = "Preamble"
        current_lines: List[str] = []
        section_start = 0
        current_offset = 0

        noise_blacklist = {
            "india sme helpdesk", "india ip sme helpdesk", "confidential", "all rights reserved",
            "table of contents", "page", "model services contract", "core terms", "crown copyright",
            "standard form document", "provision of services", "draft"
        }

        heading_patterns = [
            # 1. "1. Term and Termination" or "2. License Terms:" or "10. Warranties and Disclaimers."
            re.compile(r"^(\d{1,2}\.?\s+[A-Z][A-Za-z0-9\s,\-–—\(\)\/]{2,70})[:\.]?$"),
            # 2. "1.1 Definitions" or "8.6 Surviving Provisions."
            re.compile(r"^(\d{1,2}\.\d{1,2}\.?\s+[A-Z][A-Za-z0-9\s,\-–—\(\)\/]{2,70})[:\.]?$"),
            # 3. "ARTICLE 1: ...", "SECTION 4. ...", "SCHEDULE 1 ...", "PART A: ..."
            re.compile(r"^((?:ARTICLE|SECTION|CLAUSE|SCHEDULE|ANNEX|EXHIBIT|PART)\s+[0-9IVXLCDM\w]+[:\.\s\-–—]+[A-Za-z0-9\s,\-–—\(\)\/]{2,70})[:\.]?$", re.IGNORECASE),
            # 4. Standalone legal division keywords
            re.compile(r"^(RECITALS|DEFINITIONS|OPERATIVE PROVISIONS|GENERAL PROVISIONS|CORE TERMS|SPECIAL TERMS)$", re.IGNORECASE),
        ]

        for line in lines:
            trimmed = line.strip()
            line_len = len(line)
            is_heading = False
            heading_title = trimmed

            if 3 <= len(trimmed) <= 80:
                lower_t = trimmed.lower()
                # Skip noise and running headers/footers
                if lower_t not in noise_blacklist and not re.match(r"^page\s+\d+", lower_t) and not trimmed.endswith(",") and not trimmed.endswith(";"):
                    for pat in heading_patterns:
                        m = pat.match(trimmed)
                        if m:
                            clean_t = m.group(1).strip().rstrip(".:")
                            if len(clean_t) > 3 and clean_t.lower() not in noise_blacklist:
                                is_heading = True
                                heading_title = clean_t
                                break

            if is_heading and current_lines:
                sec_text = "\n".join(current_lines).strip()
                if sec_text:
                    p_num = self._find_page_number(section_start, pages)
                    sections.append(
                        DocumentSection(
                            section_id=f"sec_{len(sections)+1}",
                            title=current_title,
                            content=sec_text,
                            page_number=p_num,
                            char_start=section_start,
                            char_end=current_offset,
                        )
                    )
                current_title = heading_title
                current_lines = []
                section_start = current_offset

            current_lines.append(line)
            current_offset += line_len + 1

        # Final section
        if current_lines:
            sec_text = "\n".join(current_lines).strip()
            if sec_text:
                p_num = self._find_page_number(section_start, pages)
                sections.append(
                    DocumentSection(
                        section_id=f"sec_{len(sections)+1}",
                        title=current_title,
                        content=sec_text,
                        page_number=p_num,
                        char_start=section_start,
                        char_end=current_offset,
                    )
                )

        return sections

    def _find_page_number(self, offset: int, pages: List[DocumentPage]) -> int:
        """Finds page number corresponding to character offset."""
        for page in pages:
            if page.char_start <= offset <= page.char_end:
                return page.page_number
        return 1 if not pages else pages[-1].page_number
