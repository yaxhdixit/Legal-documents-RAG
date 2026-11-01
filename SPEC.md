# Product specification — Legal RAG

> Status: draft. A coding agent must not implement product code until this specification is approved through Genesis.

## Problem

Commercial and business agreements are lengthy, dense, and difficult to review quickly. Legal professionals, commercial managers, and clients spend excessive time locating clauses, identifying hidden liabilities, verifying party obligations, assessing unfavorable terms, and comparing provisions. Existing generic chatbots produce ungrounded hallucinations, lack legal document structure awareness, fail to cite precise evidence spans, and cannot provide review-ready structured audit reports. Confrere Global Legale (CGL) requires a specialized Small Language Model (SLM) powered contract intelligence platform that reliably extracts clauses, flags review priorities with evidence, explains obligations in plain English, powers conversational inquiry, and exports professional review reports.

## Users

- Legal Counsel and Attorneys at Confrere Global Legale (CGL) handling commercial contract review, risk audits, and due diligence.
- Commercial, Procurement, and Business Operations Teams negotiating vendor and client contracts, seeking rapid clause interpretation and risk highlights.
- Legal Clients of CGL interacting with reviewed agreements to understand their rights, liabilities, and obligations in plain language.

## Functional requirements

- FR-1: Multi-format Document Ingestion & Structure Preservation — Parse PDF and DOCX contracts, preserving page numbers, document structure, sections, headings, clauses, and subclause location coordinates.
- FR-2: Expert Clause Extraction & Classification — Identify and classify commercial contract provisions across 14+ standardized categories based on CUAD (including Termination, Limitation of Liability, Indemnification, Confidentiality, IP Ownership, Governing Law, Dispute Resolution, Payment Obligations, Exclusivity, Assignment, Non-compete, Reps & Warranties, Force Majeure, and Data Protection).
- FR-3: Grounded Contract Risk Detection & Priority Scoring — Analyze identified provisions and assign review priorities (High, Medium, Low, Requires Review), providing exact clause quotes, commercial implications, affected party, and targeted attorney review questions, distinguishing problematic, ambiguous, missing, and unfavorable provisions.
- FR-4: Conversational Contract Explainer with Retrieval Grounding — Provide a conversational assistant that answers user queries strictly grounded in retrieved document clauses, citing page numbers and section references, and explicitly stating when information is absent.
- FR-5: Split-Screen Interactive Contract Workspace — Deliver a professional web UI featuring an original document viewer with clause highlight navigation on the left, and structured findings, clause metadata, risk badges, and conversational assistant on the right.
- FR-6: Downloadable Professional Contract Review Report — Generate exportable structured audit reports containing contract overview, parties, key dates, detected clauses, risk breakdown, and actionable attorney recommendations.
- FR-7: Official Legal Dataset Acquisition & Standardization Pipeline — Ingest and validate official CUAD, ACORD, and ContractNLI datasets under CC BY 4.0, transforming them into a unified intermediate schema partitioned strictly by contract ID to prevent data leakage.
- FR-8: Curated Conversational SFT Dataset Pipeline — Transform standardized contract annotations into conversational instruction-response records in JSONL covering clause extraction, risk assessment, evidence reasoning, and multi-turn Q&A.
- FR-9: Specialized 1B-3B SLM Fine-Tuning & Evaluation Suite — Provide parameter-efficient fine-tuning (PEFT/LoRA) workflows and a rigorous evaluation harness benchmarking clause extraction F1, retrieval Recall@K/MRR, NLI entailment accuracy, and hallucination rate against the pretrained baseline.

## Non-functional requirements

- NFR-1: Grounded Attribution & Zero Unreferenced Legal Claims — 100% of risk explanations and conversational answers must cite exact contract clause text or explicit document absence; never emit fabricated citations.
- NFR-2: Retrieval & Response Latency — Achieve sub-3-second retrieval and generation latency for conversational turns on standard CPU/GPU inference hardware for commercial agreements up to 100 pages.
- NFR-3: Document & Tenant Isolation — Uploaded agreements and conversational sessions are strictly isolated in memory and disk storage, preventing cross-session leakage.
- NFR-4: CGL Brand Consistency & Professional Design — Web UI conforms to Confrere Global Legale brand guidelines from cgl.world and cgladr.world, using deep navy (#0A192F / #1E3A8A), warm gold (#D97706 / #C59B27), and clean typography.
- NFR-5: Data Provenance & CC BY 4.0 Compliance — All dataset pipelines preserve licensing attribution and provenance metadata across training and evaluation sets.

## Constraints

- Architecture must support local and containerized execution on developer hardware (macOS / Linux) using Python 3 and modern web standards.
- Compute constraints for local training/evaluation require parameter-efficient fine-tuning (LoRA/QLoRA) for 1B-3B parameter models rather than full pretraining from scratch.
- Untrusted Document Input constraint: Uploaded files must be treated as untrusted text, preventing prompt-injection attacks from overriding system instructions.

## Non-goals

- Providing binding or definitive legal advice or substituting for a qualified attorney.
- Automating autonomous contract signing, modification execution, or external litigation filing.
- Supporting non-contract legal domains such as criminal law, immigration, or patent filing.

## Acceptance criteria

- AC-1: System successfully ingests sample commercial PDF and DOCX contracts and extracts structured sections with page mapping.
- AC-2: Clause classifier identifies at least 14 standardized CUAD clause categories with precision/recall benchmarking.
- AC-3: Risk detection engine flags risky provisions and outputs structured risk priority, rationale, affected party, and attorney questions.
- AC-4: Conversational assistant answers contract questions using retrieved context and provides clickable citations linking to the exact document view.
- AC-5: Interactive web dashboard presents CGL-branded split-screen interface with document viewer, clause explorer, risk badges, and chat panel.
- AC-6: Downloadable contract review report exports complete executive summary, clause inventory, and risk matrix.
- AC-7: Preprocessing scripts download and validate CUAD, ACORD, and ContractNLI, outputting leakage-free train/val/test splits.
- AC-8: Evaluation harness produces quantitative comparison metrics between pretrained baseline model and fine-tuned SLM.

## Risks

- Model hallucination or inaccurate risk interpretation. Mitigation: Retrieval-grounded generation with strict prompt constraints, quote-anchoring, and explicit disclaimer that risk scores represent review priorities.
- Compute limitations on local machine for large SLM training. Mitigation: Parameter-efficient LoRA on lightweight 1B-3B model with cached embeddings and verifiable pipeline checkpoints.
- Complex multi-column or scanned PDF formatting. Mitigation: Robust text extraction utilizing pdfplumber/PyPDF with fallback layout cleaning.

## Open questions

- None.
