"""
FastAPI Server for Confrere Global Legale (CGL) Contract Intelligence SLM Platform.
Mounts multi-format contract ingestion, clause extraction, risk assessment, RAG chat, and report generation.
"""

from __future__ import annotations
import os
import shutil
import tempfile
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.ingest.parser import ContractParser, ParsedDocument
from src.ingest.clause_extractor import ClauseExtractor
from src.analysis.risk_engine import ContractRiskEngine, ContractRiskReport
from src.retrieval.vector_store import ClauseAwareVectorStore
from src.conversation.explainer import ConversationalExplainer
from src.server.report_generator import ReportGenerator
from src.models.benchmark import LegalModelBenchmark
from src.data.cuad_loader import CUADLoader
from src.data.schema import ClauseAnnotation


app = FastAPI(
    title="Confrere Global Legale (CGL) — Contract Risk Intelligence SLM",
    description="Specialized Small Language Model Contract Intelligence API for Confrere Global Legale.",
    version="1.0.0",
)

# Global in-memory storage for active session contracts
PARSER = ContractParser()
EXTRACTOR = ClauseExtractor()
RISK_ENGINE = ContractRiskEngine()
VECTOR_STORE = ClauseAwareVectorStore(session_id="cgl_global_session")
EXPLAINER = ConversationalExplainer(VECTOR_STORE)
BENCHMARK_SUITE = LegalModelBenchmark()

DOCUMENTS: Dict[str, ParsedDocument] = {}
CLAUSES: Dict[str, List[ClauseAnnotation]] = {}
RISK_REPORTS: Dict[str, ContractRiskReport] = {}


def _index_contract(parsed_doc: ParsedDocument) -> str:
    """Extracts clauses, computes risk findings, and indexes for retrieval."""
    clauses = EXTRACTOR.extract_clauses(parsed_doc)
    risk_report = RISK_ENGINE.analyze_contract(parsed_doc, clauses)
    VECTOR_STORE.index_document(parsed_doc, clauses)

    DOCUMENTS[parsed_doc.doc_id] = parsed_doc
    CLAUSES[parsed_doc.doc_id] = clauses
    RISK_REPORTS[parsed_doc.doc_id] = risk_report
    return parsed_doc.doc_id


def init_sample_contracts():
    """Initializes with verified commercial contracts on server startup."""
    loader = CUADLoader()
    sample_records = loader.get_sample_cuad_contracts()
    for rec in sample_records:
        parsed = PARSER.parse_string(rec.full_text, filename=f"{rec.title}.txt", doc_id=rec.contract_id)
        _index_contract(parsed)


# Initialize sample contracts immediately
init_sample_contracts()


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def get_index_page():
    """Serves primary CGL contract intelligence workspace."""
    web_index = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "web", "index.html")
    if os.path.exists(web_index):
        with open(web_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h3>CGL Web Application loaded.</h3>")


@app.get("/api/contracts")
def list_contracts():
    """Lists all active reviewed contracts."""
    return [
        {
            "doc_id": doc.doc_id,
            "filename": doc.filename,
            "file_type": doc.file_type,
            "char_length": len(doc.full_text),
            "num_clauses": len(CLAUSES.get(doc.doc_id, [])),
            "high_risks": RISK_REPORTS[doc.doc_id].high_priority_count if doc.doc_id in RISK_REPORTS else 0,
        }
        for doc in DOCUMENTS.values()
    ]


@app.get("/api/contracts/{doc_id}")
def get_contract_details(doc_id: str):
    """Retrieves full contract text, extracted clauses, risk assessment, and executive overview."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Contract not found")
    doc_dict = DOCUMENTS[doc_id].to_dict()
    if "full_text" not in doc_dict or not doc_dict["full_text"]:
        doc_dict["full_text"] = DOCUMENTS[doc_id].full_text
    
    clauses = CLAUSES.get(doc_id, [])
    overview = RISK_ENGINE.generate_contract_overview(DOCUMENTS[doc_id], clauses)
    
    return {
        "document": doc_dict,
        "clauses": [c.to_dict() for c in clauses],
        "risk_analysis": RISK_REPORTS[doc_id].to_dict() if doc_id in RISK_REPORTS else None,
        "overview": overview,
    }


@app.post("/api/upload")
async def upload_contract_file(file: UploadFile = File(...)):
    """Uploads and parses a contract in PDF, DOCX, or TXT format."""
    original_filename = os.path.basename(file.filename) if file.filename else "Uploaded_Contract.pdf"
    suffix = os.path.splitext(original_filename)[1].lower()
    if suffix not in [".pdf", ".docx", ".doc", ".txt", ".md"]:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        parsed_doc = PARSER.parse_file(tmp_path, original_filename=original_filename)
        parsed_doc.filename = original_filename
        doc_id = _index_contract(parsed_doc)
        return {
            "status": "success",
            "doc_id": doc_id,
            "filename": original_filename,
            "clauses_identified": len(CLAUSES[doc_id]),
            "risk_findings": len(RISK_REPORTS[doc_id].findings),
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/api/contracts/{doc_id}/chat")
def chat_contract(doc_id: str, req: ChatRequest):
    """Answers contractual questions strictly grounded in retrieved evidence."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Contract not found")

    resp = EXPLAINER.answer_question(
        doc_id=doc_id,
        query=req.query,
        session_id=req.session_id,
    )
    return resp.to_dict()


@app.get("/api/contracts/{doc_id}/report", response_class=HTMLResponse)
def get_contract_report(doc_id: str):
    """Renders professional CGL contract review report."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Contract not found")

    html = ReportGenerator.generate_html_report(
        parsed_doc=DOCUMENTS[doc_id],
        clauses=CLAUSES.get(doc_id, []),
        risk_report=RISK_REPORTS[doc_id],
    )
    return HTMLResponse(content=html)


@app.get("/api/contracts/{doc_id}/report/json")
def get_contract_report_json(doc_id: str):
    """Downloads structured CGL contract review report as machine-readable JSON."""
    if doc_id not in DOCUMENTS:
        raise HTTPException(status_code=404, detail="Contract not found")

    report_dict = ReportGenerator.generate_json_report(
        parsed_doc=DOCUMENTS[doc_id],
        clauses=CLAUSES.get(doc_id, []),
        risk_report=RISK_REPORTS[doc_id],
    )
    safe_name = (DOCUMENTS[doc_id].filename or doc_id).replace(" ", "_")
    return JSONResponse(
        content=report_dict,
        headers={"Content-Disposition": f'attachment; filename="CGL_Audit_Report_{safe_name}.json"'}
    )


@app.get("/api/benchmark")
def get_benchmark_report():
    """Generates comparative evaluation between baseline and fine-tuned SLM."""
    first_doc = next(iter(DOCUMENTS.values()), None)
    text = first_doc.full_text if first_doc else "Standard contract text for evaluation."
    return BENCHMARK_SUITE.run_benchmark(text)
