"""
IBM DeliveryIQ — PDF Report Generator
======================================
Generates IBM-format project status reports as PDFs.
Uses reportlab if available, falls back to plain-text export otherwise.
"""

# ── Graceful import: won't crash the whole app if reportlab is missing ──
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

import io
import datetime


# ─────────────────────────────────────────────────────────────────
# IBM COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────
IBM_BLUE   = colors.HexColor("#0062FF") if REPORTLAB_AVAILABLE else None
IBM_DARK   = colors.HexColor("#161616") if REPORTLAB_AVAILABLE else None
IBM_GRAY   = colors.HexColor("#F4F4F4") if REPORTLAB_AVAILABLE else None
IBM_GREEN  = colors.HexColor("#24A148") if REPORTLAB_AVAILABLE else None
IBM_AMBER  = colors.HexColor("#F1C21B") if REPORTLAB_AVAILABLE else None
IBM_RED    = colors.HexColor("#DA1E28") if REPORTLAB_AVAILABLE else None


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def generate_pdf_report(project_data: dict, risk_result: dict, health_data: dict) -> bytes:
    """
    Generate an IBM-format PDF status report.

    Returns bytes of the PDF (or a plain-text fallback if reportlab is missing).
    """
    if REPORTLAB_AVAILABLE:
        return _generate_with_reportlab(project_data, risk_result, health_data)
    else:
        return _generate_text_fallback(project_data, risk_result, health_data)


# ─────────────────────────────────────────────────────────────────
# REPORTLAB PDF GENERATION
# ─────────────────────────────────────────────────────────────────

def _generate_with_reportlab(project_data: dict, risk_result: dict, health_data: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=1 * inch,
        bottomMargin=1 * inch,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "IBMTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=IBM_DARK,
        spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    subtitle_style = ParagraphStyle(
        "IBMSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=IBM_BLUE,
        spaceAfter=2,
        fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "IBMSection",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=IBM_BLUE,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "IBMBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=IBM_DARK,
        spaceAfter=4,
        fontName="Helvetica",
        leading=14,
    )

    story = []
    now = datetime.datetime.now().strftime("%B %d, %Y")

    # ── Header ──
    story.append(Paragraph("IBM DeliveryIQ", title_style))
    story.append(Paragraph("Project Status Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=IBM_BLUE))
    story.append(Spacer(1, 10))
    story.append(Paragraph(f"Generated: {now}", body_style))
    story.append(Spacer(1, 12))

    # ── Project Details ──
    story.append(Paragraph("PROJECT OVERVIEW", section_style))
    project_name = project_data.get("project_name", "IBM Project")
    team_size    = project_data.get("team_size", "N/A")
    duration     = project_data.get("duration_weeks", "N/A")
    budget       = project_data.get("budget_usd", "N/A")

    overview_data = [
        ["Project Name",  project_name],
        ["Team Size",     str(team_size)],
        ["Duration",      f"{duration} weeks"],
        ["Budget",        f"${budget:,}" if isinstance(budget, (int, float)) else str(budget)],
        ["Report Date",   now],
    ]
    overview_table = Table(overview_data, colWidths=[2 * inch, 4.5 * inch])
    overview_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), IBM_GRAY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",   (0, 0), (-1, -1), IBM_DARK),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, IBM_GRAY]),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 12))

    # ── Risk Assessment ──
    story.append(Paragraph("RISK ASSESSMENT", section_style))
    risk_level   = risk_result.get("risk_level", "Unknown")
    confidence   = risk_result.get("confidence", 0)
    recommendation = risk_result.get("recommendation", "No recommendation available.")

    rag_color = {"Low": IBM_GREEN, "Medium": IBM_AMBER,
                 "High": IBM_RED, "Critical": IBM_RED}.get(risk_level, IBM_GRAY)

    risk_data = [
        ["Risk Level",    risk_level],
        ["Confidence",    f"{confidence:.1%}" if isinstance(confidence, float) else str(confidence)],
        ["Recommendation", recommendation],
    ]
    risk_table = Table(risk_data, colWidths=[2 * inch, 4.5 * inch])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (0, -1), IBM_GRAY),
        ("FONTNAME",    (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME",    (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",   (0, 0), (-1, -1), IBM_DARK),
        ("BACKGROUND",  (1, 0), (1, 0), rag_color),
        ("TEXTCOLOR",   (1, 0), (1, 0), colors.white),
        ("FONTNAME",    (1, 0), (1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IBM_GRAY]),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
        ("INNERGRID",   (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("WORDWRAP",    (1, 2), (1, 2), True),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 12))

    # ── Health Dimensions ──
    story.append(Paragraph("PROJECT HEALTH DIMENSIONS", section_style))
    dimensions = health_data.get("dimensions", {})
    if dimensions:
        health_rows = [["Dimension", "Score", "Status"]]
        for dim_name, dim_info in dimensions.items():
            score  = dim_info.get("score", 0) if isinstance(dim_info, dict) else dim_info
            status = ("Healthy" if score >= 70 else "Warning" if score >= 40 else "Critical")
            health_rows.append([dim_name, f"{score}%", status])

        health_table = Table(health_rows, colWidths=[3 * inch, 1.5 * inch, 2 * inch])
        status_colors = [colors.white]  # header row
        for row in health_rows[1:]:
            st = row[2]
            status_colors.append(
                IBM_GREEN if st == "Healthy" else IBM_AMBER if st == "Warning" else IBM_RED
            )

        ts = TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0), IBM_BLUE),
            ("TEXTCOLOR",   (0, 0), (-1, 0), colors.white),
            ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",   (0, 1), (-1, -1), IBM_DARK),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, IBM_GRAY]),
            ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ("INNERGRID",   (0, 0), (-1, -1), 0.25, colors.HexColor("#E0E0E0")),
            ("TOPPADDING",  (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ])
        for i, sc in enumerate(status_colors[1:], start=1):
            ts.add("BACKGROUND", (2, i), (2, i), sc)
            ts.add("TEXTCOLOR",  (2, i), (2, i), colors.white)
            ts.add("FONTNAME",   (2, i), (2, i), "Helvetica-Bold")

        health_table.setStyle(ts)
        story.append(health_table)
    else:
        health_score = health_data.get("score", "N/A")
        story.append(Paragraph(f"Overall Health Score: {health_score}/100", body_style))

    story.append(Spacer(1, 20))

    # ── Footer ──
    story.append(HRFlowable(width="100%", thickness=1, color=IBM_BLUE))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "IBM DeliveryIQ — AI-Powered Delivery Intelligence | Confidential",
        ParagraphStyle("Footer", parent=styles["Normal"],
                       fontSize=8, textColor=colors.HexColor("#6F6F6F"),
                       alignment=1)
    ))

    doc.build(story)
    return buffer.getvalue()


# ─────────────────────────────────────────────────────────────────
# PLAIN-TEXT FALLBACK (when reportlab not installed)
# ─────────────────────────────────────────────────────────────────

def _generate_text_fallback(project_data: dict, risk_result: dict, health_data: dict) -> bytes:
    """Returns a UTF-8 encoded plain-text report when reportlab is unavailable."""
    now = datetime.datetime.now().strftime("%B %d, %Y")
    lines = [
        "IBM DeliveryIQ — Project Status Report",
        "=" * 50,
        f"Generated: {now}",
        "",
        "PROJECT OVERVIEW",
        "-" * 30,
        f"Project:    {project_data.get('project_name', 'IBM Project')}",
        f"Team Size:  {project_data.get('team_size', 'N/A')}",
        f"Duration:   {project_data.get('duration_weeks', 'N/A')} weeks",
        f"Budget:     ${project_data.get('budget_usd', 'N/A'):,}" if isinstance(project_data.get('budget_usd'), (int, float)) else f"Budget: N/A",
        "",
        "RISK ASSESSMENT",
        "-" * 30,
        f"Risk Level:     {risk_result.get('risk_level', 'N/A')}",
        f"Confidence:     {risk_result.get('confidence', 0):.1%}" if isinstance(risk_result.get('confidence'), float) else "Confidence: N/A",
        f"Recommendation: {risk_result.get('recommendation', 'N/A')}",
        "",
        "PROJECT HEALTH",
        "-" * 30,
        f"Health Score: {health_data.get('score', 'N/A')}/100",
        "",
        "=" * 50,
        "IBM DeliveryIQ | Confidential",
    ]
    return "\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────────────────────────
# UTILITY: check if PDF generation is available
# ─────────────────────────────────────────────────────────────────

def is_pdf_available() -> bool:
    """Returns True if reportlab is installed and PDF generation works."""
    return REPORTLAB_AVAILABLE
