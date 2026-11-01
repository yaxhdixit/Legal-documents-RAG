"""
Professional Contract Review Report Generator for Confrere Global Legale (CGL).
Exports structured audit reports in HTML and JSON formats.
"""

from __future__ import annotations
import datetime
from typing import Dict, Any, List

from src.ingest.parser import ParsedDocument
from src.data.schema import ClauseAnnotation
from src.analysis.risk_engine import ContractRiskReport


class ReportGenerator:
    """Generates CGL-branded legal review reports."""

    @staticmethod
    def generate_html_report(
        parsed_doc: ParsedDocument,
        clauses: List[ClauseAnnotation],
        risk_report: ContractRiskReport,
        reviewer_name: str = "Confrere Global Legale Contract Intelligence SLM",
    ) -> str:
        """Renders an executive-ready, printable HTML contract review report."""
        now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")

        # Build risk findings HTML
        findings_html = []
        for f in risk_report.findings:
            badge_color = {
                "High": "#DC2626",
                "Medium": "#D97706",
                "Low": "#2563EB",
                "Requires Review": "#7C3AED",
            }.get(f.risk_priority, "#4B5563")

            questions_html = "".join(f"<li>{q}</li>" for q in f.lawyer_review_questions)

            findings_html.append(f"""
            <div style="border: 1px solid #E2E8F0; border-radius: 8px; margin-bottom: 16px; padding: 16px; background-color: #FFFFFF;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; padding-bottom: 8px; margin-bottom: 12px;">
                    <h3 style="margin: 0; color: #0F172A; font-size: 16px;">{f.clause_name}</h3>
                    <span style="background-color: {badge_color}; color: #FFFFFF; font-size: 12px; font-weight: bold; padding: 4px 10px; border-radius: 12px;">
                        {f.risk_priority} Priority
                    </span>
                </div>
                <div style="margin-bottom: 8px; font-size: 14px; color: #334155;">
                    <strong>Risk Classification:</strong> <span style="text-transform: capitalize;">{f.risk_type}</span> | 
                    <strong>Affected Party:</strong> {f.affected_party} |
                    <strong>Location:</strong> Page {f.page_number or 'N/A'}, Section {f.section_title or 'General'}
                </div>
                <div style="background-color: #F8FAFC; border-left: 4px solid #0284C7; padding: 10px 14px; margin-bottom: 12px; font-family: monospace; font-size: 13px; color: #1E293B;">
                    "{f.exact_text}"
                </div>
                <div style="margin-bottom: 8px; font-size: 14px; color: #1E293B;">
                    <strong>Concern:</strong> {f.concern_explanation}
                </div>
                <div style="margin-bottom: 12px; font-size: 14px; color: #1E293B;">
                    <strong>Commercial Implications:</strong> {f.commercial_implications}
                </div>
                <div style="background-color: #FFFBEB; border: 1px solid #FDE68A; border-radius: 6px; padding: 10px 14px;">
                    <strong style="color: #92400E; font-size: 13px;">Key Points for Reviewing Lawyer:</strong>
                    <ul style="margin: 6px 0 0 18px; padding: 0; font-size: 13px; color: #78350F;">
                        {questions_html}
                    </ul>
                </div>
            </div>
            """)

        # Missing provisions HTML
        missing_html = ""
        if risk_report.missing_provisions:
            items = "".join(f"<li style='margin-bottom: 4px;'><strong>{m}:</strong> Provision not detected in document.</li>" for m in risk_report.missing_provisions)
            missing_html = f"""
            <div style="background-color: #FEF2F2; border: 1px solid #FECACA; border-radius: 8px; padding: 16px; margin-bottom: 24px;">
                <h3 style="margin-top: 0; color: #991B1B; font-size: 16px;">Critical Missing Contractual Provisions</h3>
                <p style="font-size: 14px; color: #7F1D1D; margin-bottom: 8px;">The following standard commercial protections were not identified in the uploaded contract:</p>
                <ul style="margin: 0 0 0 20px; color: #991B1B; font-size: 14px;">
                    {items}
                </ul>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CGL Contract Intelligence Review Report — {parsed_doc.filename}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; color: #0F172A; line-height: 1.5; margin: 0; padding: 40px; background-color: #F8FAFC; }}
        .no-print-toolbar {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: #0A192F;
            color: #FFFFFF;
            padding: 12px 24px;
            margin: -40px -40px 30px -40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .toolbar-brand {{ font-size: 14px; font-weight: 600; letter-spacing: 0.5px; }}
        .toolbar-brand strong {{ color: #F59E0B; }}
        .toolbar-actions {{ display: flex; gap: 10px; align-items: center; }}
        .t-btn {{
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: all 0.2s;
            text-decoration: none;
        }}
        .btn-print {{ background: #D97706; color: #FFFFFF; }}
        .btn-print:hover {{ background: #B45309; }}
        .btn-json {{ background: #1E293B; color: #E2E8F0; border: 1px solid #334155; }}
        .btn-json:hover {{ background: #334155; color: #FFFFFF; }}
        .btn-close {{ background: transparent; color: #94A3B8; border: 1px solid #334155; }}
        .btn-close:hover {{ color: #FFFFFF; border-color: #64748B; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #FFFFFF; padding: 40px; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); border: 1px solid #E2E8F0; }}
        .header {{ border-bottom: 3px solid #0A192F; padding-bottom: 20px; margin-bottom: 30px; display: flex; justify-content: space-between; align-items: flex-start; }}
        .brand-title {{ font-size: 24px; font-weight: 800; color: #0A192F; margin: 0; text-transform: uppercase; letter-spacing: 0.5px; }}
        .brand-sub {{ font-size: 13px; color: #D97706; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
        .meta-box {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; background-color: #F1F5F9; padding: 16px; border-radius: 8px; margin-bottom: 24px; font-size: 13px; }}
        .stat-card {{ background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; text-align: center; }}
        .stat-val {{ font-size: 28px; font-weight: 700; }}
        .disclaimer {{ background-color: #F8FAFC; border: 1px solid #CBD5E1; border-radius: 6px; padding: 14px; font-size: 12px; color: #475569; margin-top: 40px; }}
        @media print {{
            .no-print-toolbar {{ display: none !important; }}
            body {{ background: #FFFFFF; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 15px; width: 100% !important; max-width: 100% !important; margin: 0 !important; }}
            @page {{ margin: 12mm; size: auto; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Floating Download / Print Toolbar -->
        <div class="no-print-toolbar">
            <div class="toolbar-brand">
                <strong>Confrere Global Legale</strong> • Executive Contract Audit
            </div>
            <div class="toolbar-actions">
                <button class="t-btn btn-print" onclick="window.print()">
                    <span>📥</span> Save / Print as PDF
                </button>
                <button class="t-btn btn-json" onclick="downloadJsonReport()">
                    <span>📊</span> Download JSON
                </button>
                <button class="t-btn btn-close" onclick="window.close()">
                    ✕ Close
                </button>
            </div>
        </div>

        <div class="header">
            <div>
                <div class="brand-title">Confrere Global Legale (CGL)</div>
                <div class="brand-sub">Contract Intelligence & Risk Assessment Report</div>
            </div>
            <div style="text-align: right; font-size: 12px; color: #64748B;">
                <div><strong>Date:</strong> {now_str}</div>
                <div><strong>Reviewer:</strong> {reviewer_name}</div>
            </div>
        </div>

        <div class="meta-box">
            <div><strong>Document:</strong><br>{parsed_doc.filename}</div>
            <div><strong>Format / Pages:</strong><br>{parsed_doc.file_type.upper()} ({len(parsed_doc.pages)} pages)</div>
            <div><strong>Clauses Extracted:</strong><br>{len(clauses)} provisions</div>
            <div><strong>Document ID:</strong><br><code style="font-size: 11px;">{parsed_doc.doc_id}</code></div>
        </div>

        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 30px;">
            <div class="stat-card">
                <div class="stat-val" style="color: #DC2626;">{risk_report.high_priority_count}</div>
                <div style="font-size: 12px; color: #64748B; font-weight: 600;">High Priority</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #D97706;">{risk_report.medium_priority_count}</div>
                <div style="font-size: 12px; color: #64748B; font-weight: 600;">Medium Priority</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #2563EB;">{risk_report.low_priority_count}</div>
                <div style="font-size: 12px; color: #64748B; font-weight: 600;">Low Priority</div>
            </div>
            <div class="stat-card">
                <div class="stat-val" style="color: #7C3AED;">{risk_report.requires_review_count}</div>
                <div style="font-size: 12px; color: #64748B; font-weight: 600;">Requires Review</div>
            </div>
        </div>

        <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 18px; margin-bottom: 30px;">
            <h3 style="margin-top: 0; color: #0A192F; font-size: 16px;">Executive Risk Summary</h3>
            <p style="margin-bottom: 0; font-size: 14px; color: #334155;">{risk_report.overall_risk_summary}</p>
        </div>

        {missing_html}

        <h2 style="color: #0A192F; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; margin-bottom: 20px; font-size: 20px;">
            Detailed Risk Findings & Contractual Evidence
        </h2>
        {"".join(findings_html)}

        <div class="disclaimer">
            <strong>Legal Notice & Advisory Nature:</strong> This document audit was prepared by the Confrere Global Legale (CGL) Contract Intelligence system.
            Risk priority classifications (High, Medium, Low, Requires Review) represent automated legal review recommendations rather than definitive legal conclusions.
            The findings are intended to assist qualified attorneys in commercial negotiations and due diligence. Independent legal validation is required prior to contract execution.
        </div>
    </div>

    <script>
        function downloadJsonReport() {{
            var path = window.location.pathname;
            var jsonUrl = path.endsWith('/json') ? path : path + '/json';
            window.location.href = jsonUrl;
        }}

        // Auto-launch PDF print dialog if opened with ?auto_print=1 or ?download=pdf
        if (window.location.search.includes('auto_print=1') || window.location.search.includes('download=pdf')) {{
            window.addEventListener('load', function() {{
                setTimeout(function() {{
                    window.print();
                }}, 400);
            }});
        }}
    </script>
</body>
</html>
"""
        return html

    @staticmethod
    def generate_json_report(
        parsed_doc: ParsedDocument,
        clauses: List[ClauseAnnotation],
        risk_report: ContractRiskReport,
    ) -> Dict[str, Any]:
        """Generates machine-readable structured JSON report."""
        return {
            "meta": {
                "generated_at": datetime.datetime.now().isoformat(),
                "organization": "Confrere Global Legale (CGL)",
                "associated": "Confrere Global Legale Legal Operations",
                "system": "CGL Contract Risk Intelligence SLM",
            },
            "document": parsed_doc.to_dict(),
            "clauses": [c.to_dict() for c in clauses],
            "risk_analysis": risk_report.to_dict(),
        }
