"""AQZ Exam Paper Studio - Streamlit front end for the existing AQZ backend.

Run: streamlit run app.py
Implementation modules are organized under the `aqz/` package.

Requires: streamlit, reportlab, pymupdf and the existing AQZ dependencies.
"""

import hashlib
import io
import json
import os
import re
import tempfile
from collections import Counter
from datetime import date
from html import escape as html_escape
from pathlib import Path

import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
)

from aqz.blueprint.exam_blueprint import validate_blueprint


TYPE_NAMES = {
    "WHQ": "WH-Questions",
}
DIFFICULTIES = ("Easy", "Medium", "Hard")
BLOOMS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_PROCESSED_CANDIDATES = 80
MAX_T5_PASSAGES = 24
PIPELINE_VERSION = "2026-11-21-clean-source-v1"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    color-scheme: light;
}
[data-testid="stAppViewContainer"] {
    background: #f5f7fc !important;
    color: #15213c !important;
    --text-color: #15213c;
    --background-color: #f5f7fc;
    --secondary-background-color: #ffffff;
    --primary-color: #3157c8;
}
.block-container {padding-top: 1.75rem; padding-bottom: 3.5rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: #101e3a;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 {color: #e4eaff !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #b6c5e0;}
h1, h2, h3 {font-family: 'Outfit', sans-serif !important; letter-spacing:-.02em;}
h1 {color:#15213c !important;} h2, h3 {color:#203054 !important;}
/* Streamlit inherits its widget palette from the browser theme. Pin the
   content area to a readable light palette so dark browser themes cannot
   produce pale labels on a white page or dark controls. */
[data-testid="stMain"] p,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] {color:#263653 !important;}
.aqz-hero p {color:#dbe6ff !important;}
[data-testid="stMain"] input,
[data-testid="stMain"] textarea,
[data-testid="stMain"] [data-baseweb="select"] > div,
[data-testid="stMain"] [data-baseweb="input"] > div {
    background-color:#fff !important;
    color:#192945 !important;
    border-color:#cbd5e5 !important;
}
[data-testid="stMain"] input::placeholder,
[data-testid="stMain"] textarea::placeholder {color:#73819b !important;opacity:1;}
[data-testid="stMain"] [role="radiogroup"] label,
[data-testid="stMain"] [data-testid="stCheckbox"] label {color:#263653 !important;}
[data-testid="stMain"] [data-testid="stFileUploader"] section {
    background:#fff !important;color:#263653 !important;
}
[data-testid="stMain"] [data-testid="stFileUploader"] button {
    background:#eef2fa !important;color:#203054 !important;
    border:1px solid #cbd5e5 !important;
}
[data-testid="stMain"] [data-testid="stFileUploader"] * {color:#263653 !important;}
[data-testid="stMain"] [data-testid="stExpander"] {background:#fff !important;}
[data-testid="stMain"] [data-testid="stExpander"] summary,
[data-testid="stMain"] [data-testid="stExpander"] summary * {
    background:#fff !important;color:#203054 !important;
}
[data-testid="stMain"] button {
    background:#fff !important;color:#203054 !important;
    border:1px solid #cbd5e5 !important;
}
[data-testid="stMain"] button[kind="primary"],
[data-testid="stMain"] [data-testid="stDownloadButton"] button[kind="primary"] {
    background:linear-gradient(100deg,#244cb4,#5169d7) !important;
    border:0 !important;color:#fff !important;
}
[data-testid="stMain"] button[kind="primary"] *,
[data-testid="stMain"] [data-testid="stDownloadButton"] button[kind="primary"] * {
    color:#fff !important;
}
[data-testid="stMain"] [data-testid="stAlert"] {
    background:#eaf0fc !important;border:1px solid #cbd8f2 !important;
    border-radius:12px !important;
}
[data-testid="stMain"] [data-testid="stAlert"] * {color:#203054 !important;}
/* Streamlit versions use different wrappers for status messages. */
[role="alert"], [role="status"] {
    background:#eaf0fc !important;border:1px solid #cbd8f2 !important;
    border-radius:12px !important;color:#203054 !important;
}
[role="alert"] *, [role="status"] * {color:#203054 !important;}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background-color:#fff !important;color:#192945 !important;
    border-color:#cbd5e5 !important;
}
.aqz-hero {background:linear-gradient(112deg,#12254c,#234c9e 68%,#526ce5);
    padding:31px 36px; border-radius:20px; color:#fff; margin-bottom:24px;
    box-shadow:0 14px 30px rgba(22,43,90,.12);}
.aqz-kicker {letter-spacing: .14em; font-size:11px; font-weight:800; color:#b7ceff;}
.aqz-hero h1 {color:white !important;font-size:37px;margin:7px 0 8px;line-height:1.1;}
.aqz-hero p {font-size:14px;color:#dbe6ff;max-width:670px;margin:0;}
.aqz-card {background:#fff;border:1px solid #e6ebf5;border-radius:15px;
    padding:17px 19px;margin:3px 0 14px;box-shadow:0 3px 10px rgba(35,57,104,.035);}
.aqz-metric .number {font:700 28px 'Outfit',sans-serif;color:#1b3263;line-height:1.05;}
.aqz-metric .label {font-size:12px;color:#687894;margin-top:8px;}
.aqz-tag {display:inline-block;font-size:11px;font-weight:700;letter-spacing:.05em;
    background:#e8eefc;border-radius:40px;padding:5px 10px;color:#274f98;}
.aqz-footnote {font-size:12px;color:#73819b;}
[data-testid="stFileUploader"] {border:1px dashed #9aadd1;border-radius:12px;padding:8px;}
div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
    background:linear-gradient(100deg,#244cb4,#5169d7); border:0; color:white;
    border-radius:10px; font-weight:700; min-height:45px;}
div.stButton > button, div.stDownloadButton > button {border-radius:10px;}
[data-testid="stExpander"] {border:1px solid #e4e9f4;border-radius:12px;background:#fff;}
hr {border-color:#e5ebf4;}
</style>
"""


def new_section(uid, kind="WHQ", offered=5, answer=3, marks=5):
    return {
        "uid": uid,
        "type": kind,
        "title": TYPE_NAMES[kind],
        "offered_count": offered,
        "answer_count": answer,
        "marks_each": marks,
        "difficulty_enabled": False,
        "bloom_enabled": False,
        "difficulty_counts": {x: 0 for x in DIFFICULTIES},
        "bloom_counts": {x: 0 for x in BLOOMS},
    }


def init_state():
    if "aqz_sections" not in st.session_state:
        st.session_state.aqz_sections = [
            new_section(1, "WHQ", 12, 10, 1),
            new_section(2, "WHQ", 12, 6, 5),
        ]
        st.session_state.aqz_next_uid = 4
    if "aqz_result" not in st.session_state:
        st.session_state.aqz_result = None
    if "aqz_pdfs" not in st.session_state:
        st.session_state.aqz_pdfs = None


def section_id(index):
    return chr(65 + index) if index < 26 else f"S{index + 1}"


def build_blueprint(metadata, sections, expected_marks=0):
    paper = dict(metadata)
    paper["sections"] = []

    for i, item in enumerate(sections):
        sec = {
            "id": section_id(i),
            "title": item["title"].strip() or TYPE_NAMES[item["type"]],
            "type": item["type"],
            "offered_count": int(item["offered_count"]),
            "answer_count": int(item["answer_count"]),
            "marks_each": int(item["marks_each"]),
        }
        if item["difficulty_enabled"]:
            sec["difficulty_targets"] = {
                k: int(v) for k, v in item["difficulty_counts"].items() if v > 0
            }
            if sum(sec["difficulty_targets"].values()) != sec["offered_count"]:
                raise ValueError(
                    f"Section {sec['id']}: difficulty counts must total "
                    f"{sec['offered_count']} offered questions."
                )
        if item["bloom_enabled"]:
            sec["bloom_targets"] = {
                k: int(v) for k, v in item["bloom_counts"].items() if v > 0
            }
            if sum(sec["bloom_targets"].values()) != sec["offered_count"]:
                raise ValueError(
                    f"Section {sec['id']}: Bloom counts must total "
                    f"{sec['offered_count']} offered questions."
                )
        paper["sections"].append(sec)

    if expected_marks:
        paper["full_marks"] = int(expected_marks)

    return validate_blueprint(paper)


def _xml(text):
    return html_escape(str(text), quote=False).replace("\n", "<br/>")


def _subparts_text(item):
    return "\n".join(
        f"{part['question']} | {part['marks']}"
        for part in item.get("subparts", [])
    )


def _parse_subparts(value, total_marks):
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    parts = []
    for index, line in enumerate(lines):
        question, separator, marks_text = line.rpartition("|")
        if not separator or not question.strip():
            raise ValueError(
                f"LAQ subpart {index + 1} must use 'question text | marks'."
            )
        try:
            marks = int(marks_text.strip())
        except ValueError as error:
            raise ValueError(
                f"LAQ subpart {index + 1} marks must be a positive whole number."
            ) from error
        if marks <= 0:
            raise ValueError(
                f"LAQ subpart {index + 1} marks must be a positive whole number."
            )
        parts.append({"question": question.strip(), "marks": marks})
    if parts and sum(part["marks"] for part in parts) != int(total_marks):
        raise ValueError(
            f"LAQ subpart marks must total {total_marks} for this question."
        )
    return parts


def _roman(number):
    numerals = (
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    )
    result = []
    for value, numeral in numerals:
        count, number = divmod(number, value)
        result.extend([numeral] * count)
    return "".join(result)


def _display_question(item):
    question = str(item.get("question", ""))
    if item.get("question_type") == "TRUE_FALSE":
        question = re.sub(
            r"^\s*true\s*(?:/|or)\s*false\s*:\s*",
            "",
            question,
            count=1,
            flags=re.I,
        )
    elif item.get("question_type") == "FILL_BLANK":
        question = re.sub(
            r"^\s*fill\s+in\s+the\s+blank\s*:\s*",
            "",
            question,
            count=1,
            flags=re.I,
        )
    return question


def make_pdf(paper, questions, logo_bytes=None, answer_key=False):
    """Create a fully separate PDF in memory; no public output files required."""
    out = io.BytesIO()
    doc = SimpleDocTemplate(
        out, pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=34, bottomMargin=48,
        title=("AQZ Answer Key" if answer_key else "AQZ Question Paper"),
        author=paper["college_name"],
    )
    styles = getSampleStyleSheet()
    navy, muted = colors.HexColor("#203559"), colors.HexColor("#64748b")
    body = ParagraphStyle(
        "AQZBody", parent=styles["Normal"], fontName="Helvetica",
        fontSize=10, leading=16, textColor=navy, spaceAfter=6
    )
    headline = ParagraphStyle(
        "AQZHeadline", parent=body, fontName="Helvetica-Bold",
        fontSize=15, leading=19, alignment=TA_CENTER, spaceAfter=5
    )
    center = ParagraphStyle(
        "AQZCenter", parent=body, alignment=TA_CENTER,
        fontSize=9, leading=13
    )
    section_style = ParagraphStyle(
        "AQZSection", parent=body, fontName="Helvetica-Bold",
        fontSize=11, leading=14, alignment=TA_CENTER,
        textColor=colors.black, spaceBefore=12, spaceAfter=3
    )
    small = ParagraphStyle(
        "AQZSmall", parent=body, fontSize=8.5, leading=12,
        textColor=colors.black, spaceAfter=5
    )
    answer_style = ParagraphStyle(
        "AQZAnswer", parent=body, leftIndent=21, spaceAfter=11,
        textColor=colors.HexColor("#334155")
    )

    story = []
    logo = None
    if logo_bytes:
        try:
            from reportlab.lib.utils import ImageReader
            reader = ImageReader(io.BytesIO(logo_bytes))
            w, h = reader.getSize()
            scale = min(57 / w, 57 / h)
            logo = Image(
                io.BytesIO(logo_bytes), width=w * scale, height=h * scale,
                hAlign="CENTER"
            )
        except Exception:
            pass  # Invalid logo should not prevent PDF export.

    institution = [
        Paragraph(_xml(paper["college_name"]).upper(), headline),
        Paragraph(_xml(paper["college_address"]), center),
        Paragraph(
            f"Paper Code: {_xml(paper['paper_code'])} &nbsp; | &nbsp; "
            f"{_xml(paper['course_name'])} / {_xml(paper.get('branch', ''))}",
            center,
        ),
        Paragraph(
            f"{_xml(paper['subject_name'])} &nbsp; | &nbsp; "
            f"Semester {_xml(paper['semester'])} &nbsp; | &nbsp; "
            f"{_xml(paper['academic_year'])}",
            center,
        ),
    ]
    if paper.get("university_name"):
        institution.insert(1, Paragraph(_xml(paper["university_name"]), center))
    header = Table(
        [[logo, institution]] if logo else [[institution]],
        colWidths=[58, doc.width - 58] if logo else [doc.width],
        hAlign="CENTER",
    )
    header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(header)
    story.append(Spacer(1, 8))

    duration = int(paper["duration_minutes"])
    duration_label = (
        f"{duration // 60} Hours" if duration % 60 == 0
        else f"{duration} Minutes"
    )
    exam_line = Table(
        [[
            Paragraph(f"<i>Time Allotted: {_xml(duration_label)}</i>", body),
            Paragraph(f"<b>Full Marks: {_xml(paper['full_marks'])}</b>", body),
        ]],
        colWidths=[doc.width / 2, doc.width / 2],
    )
    exam_line.setStyle(TableStyle([
        ("ALIGN", (0, 0), (0, 0), "LEFT"),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(exam_line)
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        f"<b>{_xml(paper.get('exam_name') or 'Examination')} - "
        f"{'ANSWER KEY &amp; MARKING GUIDE' if answer_key else 'QUESTION PAPER'}</b>",
        center,
    ))
    story.append(Paragraph(
        f"Exam Date: {_xml(paper['exam_date'])}", center
    ))
    has_figures_instruction = any(
        "figures in the margin" in item.casefold()
        for item in paper.get("instructions", [])
    )

    if not answer_key:
        for instruction in paper.get("instructions", []):
            story.append(Paragraph(f"<i>{_xml(instruction)}</i>", center))
    if not has_figures_instruction:
        story.append(Paragraph(
            "<i>The figures in the margin indicate full marks.</i>", center
        ))
    story.append(Spacer(1, 7))

    n = 1
    for section in paper["sections"]:
        kind = section["type"]
        story.append(Paragraph(
            f"GROUP-{_xml(section['id'])} ({_xml(section['title'])})",
            section_style
        ))
        directive = f"Answer any {section['answer_count']} of the following:"
        marks_summary = (
            f"[{section['marks_each']} x {section['answer_count']} = "
            f"{section['section_marks']}]"
        )
        directive_text = (
            f"<b>{_xml(directive)}</b>"
            if kind in {"FILL_BLANK", "TRUE_FALSE"} else _xml(directive)
        )
        directive_table = Table(
            [[Paragraph(directive_text, body),
              Paragraph(_xml(marks_summary), body)]],
            colWidths=[doc.width * .72, doc.width * .28],
        )
        directive_table.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story.append(directive_table)

        items = [item for item in questions if item.get("section_id") == section["id"]]
        if kind in {"FILL_BLANK", "TRUE_FALSE"}:
            for index, item in enumerate(items, 1):
                row = Table(
                    [[Paragraph(f"({_roman(index)})", body),
                      Paragraph(_xml(_display_question(item)), body),
                      Paragraph(
                          "" if answer_key else
                          f"[{item.get('marks', section['marks_each'])}]",
                          body,
                      )]],
                    colWidths=[35, doc.width - 80, 45],
                )
                row.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]))
                story.append(row)
                if answer_key:
                    story.append(Paragraph(
                        "<b>Answer:</b> "
                        + _xml(item.get("answer") or "Answer not provided."),
                        answer_style,
                    ))
            n += 1
        else:
            for item in items:
                number = n
                subparts = item.get("subparts") or []
                mark_value = int(item.get("marks", section["marks_each"]))
                question_marks = "" if answer_key or subparts else f"[{mark_value}]"
                row = Table(
                    [[Paragraph(f"<b>{number}.</b>", body),
                      Paragraph(_xml(item["question"]), body),
                      Paragraph(question_marks, body)]],
                    colWidths=[30, doc.width - 75, 45],
                )
                row.setStyle(TableStyle([
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(row)
                for part_index, part in enumerate(subparts):
                    part_row = Table(
                        [[Paragraph(f"({chr(97 + part_index)})", body),
                          Paragraph(_xml(part["question"]), body),
                          Paragraph("" if answer_key else f"[{part['marks']}]", body)]],
                        colWidths=[43, doc.width - 88, 45],
                    )
                    part_row.setStyle(TableStyle([
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ALIGN", (2, 0), (2, 0), "RIGHT"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                        ("TOPPADDING", (0, 0), (-1, -1), 1),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                    ]))
                    story.append(part_row)
                if answer_key:
                    answer = item.get("answer") or "Answer not provided."
                    story.append(Paragraph(
                        "<b>Answer / marking points:</b> " + _xml(answer),
                        answer_style,
                    ))
                    story.append(Paragraph(
                        f"Bloom: {_xml(item.get('bloom', 'N/A'))} | "
                        f"Difficulty: {_xml(item.get('difficulty', 'N/A'))} | "
                        f"Marks: {mark_value}",
                        small,
                    ))
                n += 1
        story.append(Spacer(1, 5))

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFillColor(colors.HexColor("#F3F4F6"))
        canvas.setFont("Helvetica-Bold", 17)
        watermark = str(paper.get("college_name", "AQZ"))[:34]
        for y in range(50, int(A4[1]), 105):
            for x in range(-80, int(A4[0]) + 100, 190):
                canvas.saveState()
                canvas.translate(x, y)
                canvas.rotate(35)
                canvas.drawCentredString(0, 0, watermark)
                canvas.restoreState()
        canvas.setStrokeColor(colors.HexColor("#dce4f0"))
        canvas.line(42, 36, A4[0] - 42, 36)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#718096"))
        canvas.drawString(42, 24, str(paper["paper_code"])[:36])
        canvas.drawRightString(A4[0] - 42, 24, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return out.getvalue()


def inspect_pdf(pdf_bytes):
    """Read page count and detect image-only/scanned PDFs early."""
    import fitz
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        pages = len(doc)
        sample = sum(len(p.get_text().strip()) for p in doc)
    return pages, sample


def read_pdf_chunks(pdf_bytes):
    """Reuse original path-based PDF extractor safely via temporary PDF."""
    from aqz.ingestion.pdf_service import extract_pdf_text
    from aqz.preprocessing.text_processor import preprocess_text, chunk_text
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(pdf_bytes)
            temp_path = temp.name
        text = preprocess_text(extract_pdf_text(temp_path))
        return chunk_text(text)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def run_pipeline(pdf_bytes, paper, use_t5=True, quick_test=False, log=None, source_text=None):
    """Returns report and, if feasible, an exact blueprint-matched selection."""
    from aqz.generation.question_generator import generate_questions
    from aqz.processing.question_processor import process_questions, select_questions, QuestionSelectionError

    if log:
        log("Cleaning and chunking learning material...", .18)
    if source_text is not None:
        from aqz.preprocessing.text_processor import preprocess_text, chunk_text
        chunks = chunk_text(preprocess_text(source_text))
    else:
        chunks = read_pdf_chunks(pdf_bytes)
    if quick_test:
        chunks = chunks[:2]
    if not chunks:
        raise ValueError("No usable text found. Enter or upload explanatory learning material and try again.")

    if log:
        log(f"Generating candidates from {len(chunks)} chunks...", .38)
    candidates = generate_questions(
        chunks,
        use_t5=use_t5,
        max_t5_passages=MAX_T5_PASSAGES,
    )
    if not candidates:
        raise ValueError("No questions generated from the document.")

    if log:
        log("Extracting answers and classifying questions...", .63)
    processed = process_questions(
        candidates,
        max_questions=MAX_PROCESSED_CANDIDATES,
    )

    usable = [
        q for q in processed
        if q.get("answer_valid") is not False
        and str(q.get("answer") or "").strip()
        and "could not be confidently extracted" not in str(q.get("answer", "")).lower()
        and q.get("question_type") in TYPE_NAMES
    ]
    counts = Counter(q.get("question_type") for q in usable)
    report = {
        "chunks": len(chunks), "generated": len(candidates),
        "processed": len(processed), "counts": dict(counts),
        "candidates": usable, "selected": [], "shortage": None,
    }

    if log:
        log("Matching section counts, Bloom and difficulty quotas...", .85)
    try:
        report["selected"] = select_questions(processed, paper)
    except QuestionSelectionError as exc:
        report["shortage"] = str(exc)
    if log:
        log("Finished.", 1.0)
    return report


def _metric(title, value):
    st.markdown(
        '<div class="aqz-card aqz-metric"><div class="number">'
        + html_escape(str(value))
        + '</div><div class="label">'
        + html_escape(title)
        + '</div></div>',
        unsafe_allow_html=True,
    )


def show_sections():
    st.markdown("### 02  ·  Question pattern")
    st.caption("Add or remove sections. These values are fully dynamic—not fixed to 12 / 6 / 6.")
    sections = st.session_state.aqz_sections
    for i, item in enumerate(list(sections)):
        uid = item["uid"]
        sec_label = section_id(i)
        with st.expander(f"Section {sec_label}  ·  {TYPE_NAMES[item['type']]}", expanded=(i == 0)):
            left, right = st.columns([3, 1])
            with left:
                kinds = list(TYPE_NAMES)
                item["type"] = st.selectbox(
                    "Question type", kinds, index=kinds.index(item["type"]),
                    format_func=lambda x: TYPE_NAMES[x], key=f"type_{uid}"
                )
            with right:
                if st.button("Remove", key=f"remove_{uid}", disabled=len(sections) == 1):
                    st.session_state.aqz_sections = [s for s in sections if s["uid"] != uid]
                    st.rerun()
            item["title"] = st.text_input(
                "Section title", value=item["title"], key=f"title_{uid}"
            )
            c1, c2, c3 = st.columns(3)
            item["offered_count"] = c1.number_input(
                "Questions offered", min_value=1, max_value=250,
                value=item["offered_count"], key=f"offered_{uid}"
            )
            item["answer_count"] = c2.number_input(
                "Answer any", min_value=1, max_value=250,
                value=item["answer_count"], key=f"answer_{uid}"
            )
            item["marks_each"] = c3.number_input(
                "Marks each", min_value=1, max_value=100,
                value=item["marks_each"], key=f"marks_{uid}"
            )
            st.caption(
                f"Section {sec_label} full marks: "
                f"{item['answer_count']} × {item['marks_each']} = "
                f"{item['answer_count'] * item['marks_each']}"
            )
            with st.expander("Advanced: exact difficulty & Bloom quotas", expanded=False):
                item["difficulty_enabled"] = st.checkbox(
                    "Enforce difficulty counts", value=item["difficulty_enabled"],
                    key=f"difficulty_on_{uid}"
                )
                if item["difficulty_enabled"]:
                    cols = st.columns(3)
                    for j, level in enumerate(DIFFICULTIES):
                        item["difficulty_counts"][level] = cols[j].number_input(
                            level, min_value=0, max_value=250,
                            value=item["difficulty_counts"][level],
                            key=f"diff_{uid}_{level}"
                        )
                    st.caption("Difficulty counts must add up to questions offered.")
                item["bloom_enabled"] = st.checkbox(
                    "Enforce Bloom-level counts", value=item["bloom_enabled"],
                    key=f"bloom_on_{uid}"
                )
                if item["bloom_enabled"]:
                    cols = st.columns(3)
                    for j, level in enumerate(BLOOMS):
                        item["bloom_counts"][level] = cols[j % 3].number_input(
                            level, min_value=0, max_value=250,
                            value=item["bloom_counts"][level],
                            key=f"bloom_{uid}_{level}"
                        )
                    st.caption("Bloom counts must add up to questions offered.")
    if st.button("＋ Add another section", use_container_width=True):
        uid = st.session_state.aqz_next_uid
        st.session_state.aqz_next_uid += 1
        sections.append(new_section(uid))
        st.rerun()


def app_main():
    st.set_page_config(page_title="AQZ | Exam Paper Studio", page_icon="📘", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    with st.sidebar:
        st.markdown("### ◈ AQZ  /  Exam Studio")
        st.caption("Academic assessment · Teacher workspace")
        st.divider()
        st.markdown("### 01  ·  Institution")
        college = st.text_input("College name", value="Example Institute of Technology")
        address = st.text_input("College address", value="Kolkata, West Bengal, India")
        university = st.text_input("University (optional)", value="")
        exam_name = st.text_input("Examination", value="Semester Examination")
        course = st.text_input("Course", value="B.Tech")
        branch = st.text_input("Branch", value="Information Technology")
        semester = st.text_input("Semester", value="6")
        year = st.text_input("Academic year", value="2026-2027")
        subject = st.text_input("Subject / paper", value="Artificial Intelligence")
        code = st.text_input("Paper code", value="AI-601")
        exam_day = st.date_input("Exam date", value=date.today())
        duration = st.number_input("Duration (minutes)", min_value=15, max_value=600, value=180, step=15)
        logo_file = st.file_uploader("Institution logo (optional)", type=["png", "jpg", "jpeg"], key="logo_upload")
        st.divider()
        notes = st.text_area(
            "Teacher review notes (optional)",
            value="Cover the uploaded material and avoid repeated concepts.", height=90,
            help="Stored for review; free-form instructions are not automatically enforced by the current generator."
        )
        instructions_text = st.text_area(
            "Extra printed paper instructions", value="Figures in the margin indicate full marks.", height=90
        )
        st.caption("Source PDFs are processed temporarily and deleted after extraction.")

    st.markdown(
        '<div class="aqz-hero"><div class="aqz-kicker">AQZ  /  ASSESSMENT INTELLIGENCE</div>'
        '<h1>Exam Paper Studio</h1>'
        '<p>Turn course PDFs into structured examination papers with dynamic marks, '
        'answer-any rules, question review, and separate PDF exports.</p></div>',
        unsafe_allow_html=True,
    )

    metadata = {
        "college_name": college, "college_address": address,
        "university_name": university, "exam_name": exam_name,
        "course_name": course, "branch": branch,
        "semester": semester, "academic_year": year,
        "subject_name": subject, "paper_code": code,
        "exam_date": exam_day.isoformat(), "duration_minutes": int(duration),
        "teacher_prompt": notes, "logo_path": "",
        "instructions": [x.strip() for x in instructions_text.splitlines() if x.strip()],
    }

    left, right = st.columns([1.02, .98], gap="large")
    with left:
        st.markdown("### 01  ·  Upload learning material")
        source_mode = st.radio("Source type", ["PDF", "Paste text"], horizontal=True, key="source_mode")
        page_count = None
        source_bytes = None
        source_text = None
        uploaded = None
        if source_mode == "PDF":
            uploaded = st.file_uploader(
                "Upload a text-based PDF (7–8 pages or more)", type=["pdf"],
                help="Use course notes, learning material or a syllabus with explanations. Scanned PDFs need OCR.",
                key="source_pdf"
            )
        else:
            source_text = st.text_area(
                "Paste your learning material", height=260,
                placeholder="Paste course notes, textbook excerpts, or other explanatory material here...",
                help="Paste enough explanatory material to generate the question pattern you configured.",
                key="source_text"
            )
            if source_text and not source_text.strip():
                source_text = None
            if source_text:
                st.success(f"Text entered · {len(source_text.split())} words")
        if uploaded:
            source_bytes = uploaded.getvalue()
            try:
                if len(source_bytes) > MAX_UPLOAD_BYTES:
                    raise ValueError("File exceeds 25 MB. Upload a smaller PDF.")
                if not source_bytes.startswith(b"%PDF-"):
                    raise ValueError("This file is not a valid PDF.")
                page_count, chars = inspect_pdf(source_bytes)
                st.success(f"{uploaded.name} · {page_count} pages · {len(source_bytes)/1024/1024:.2f} MB")
                if chars < 100:
                    st.warning("Little selectable text detected. Scanned/image PDFs need OCR before processing.")
            except Exception as exc:
                st.error(f"PDF validation failed: {exc}")
                source_bytes = None

        engine = st.radio(
            "Generation mode", ["T5 + source-based rules", "Source-based rules only (fast test)"],
            index=1, horizontal=False,
            help="Source-based rules are recommended on CPU machines. T5 also loads a large local model and is substantially slower."
        )
        full_scope = st.checkbox("Use all PDF content", value=True)
        if not full_scope:
            st.info("Quick test uses only the first two chunks. Do not treat it as a full-paper assessment.")
    with right:
        show_sections()

    expected_marks = st.number_input(
        "Optional target full marks (0 = calculate automatically)",
        min_value=0, max_value=1000, value=0, step=5,
        help="If set, AQZ will reject a pattern whose answer-any totals do not match."
    )
    sections = st.session_state.aqz_sections
    total_offered = sum(int(s["offered_count"]) for s in sections)
    total_answer = sum(int(s["answer_count"]) for s in sections)
    total_marks = sum(int(s["answer_count"]) * int(s["marks_each"]) for s in sections)
    a, b, c, d = st.columns(4)
    with a: _metric("Offered questions", total_offered)
    with b: _metric("Questions to answer", total_answer)
    with c: _metric("Calculated full marks", total_marks)
    with d: _metric("Uploaded pages", page_count or "—")

    paper, validation_error = None, None
    try:
        paper = build_blueprint(metadata, sections, int(expected_marks))
    except Exception as exc:
        validation_error = str(exc)
        st.error(f"Paper configuration: {validation_error}")

    if paper:
        st.success(
            f"Valid pattern: {paper['total_offered_questions']} offered · "
            f"{paper['total_questions_to_answer']} to answer · {paper['full_marks']} full marks."
        )

    st.markdown("### 03  ·  Generate question paper")
    st.caption("The generator produces candidates, then AQZ selects exactly the required counts. If the content is insufficient, it reports the shortage instead of inventing questions.")
    source_signature = source_bytes if source_bytes is not None else (source_text or "").encode("utf-8")
    can_generate = bool(source_signature.strip() and paper)
    signature = None
    if can_generate:
        logo_for_signature = logo_file.getvalue() if logo_file else b""
        signature = hashlib.sha256(
            source_signature + logo_for_signature
            + json.dumps(paper, sort_keys=True).encode()
            + engine.encode() + str(full_scope).encode()
            + PIPELINE_VERSION.encode()
        ).hexdigest()

    if st.button(
        "✦ Generate & match exam pattern", type="primary",
        use_container_width=True, disabled=not can_generate
    ):
        st.session_state.aqz_result = None
        st.session_state.aqz_pdfs = None
        if "aqz_review_editor" in st.session_state:
            del st.session_state["aqz_review_editor"]
        bar = st.progress(0, text="Starting AQZ...")
        def update(message, fraction):
            bar.progress(int(fraction * 100), text=message)
        try:
            report = run_pipeline(
                source_bytes, paper,
                use_t5=(engine == "T5 + source-based rules"),
                quick_test=not full_scope, log=update, source_text=source_text,
            )
            st.session_state.aqz_result = {
                "signature": signature, "paper": paper,
                "report": report, "logo": logo_file.getvalue() if logo_file else None,
                "full_scope": full_scope,
            }
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            with st.expander("Technical traceback"):
                import traceback
                st.code(traceback.format_exc())
        finally:
            bar.empty()

    result = st.session_state.aqz_result
    if not result:
        st.info("Upload a PDF or paste learning material, adjust your sections and click Generate. No model is loaded until you start generation.")
        return
    if result["signature"] != signature:
        st.warning("Your inputs have changed since the last generation. Click Generate again to refresh the paper.")
        return

    report = result["report"]
    st.markdown("### 04  ·  Generation results")
    a, b, c = st.columns(3)
    a.metric("Chunks processed", report["chunks"])
    b.metric("Candidates generated", report["generated"])
    c.metric("Valid answer candidates", len(report["candidates"]))
    st.markdown("**Available question types**")
    cols = st.columns(4)
    for col, (kind, label) in zip(cols, TYPE_NAMES.items()):
        col.metric(label, report["counts"].get(kind, 0))

    if report["shortage"]:
        st.error("Cannot build a complete paper from this material: " + report["shortage"])
        st.info("The requested exam pattern was not fully matched. You can still review and export the available questions as a DRAFT, or reduce the section counts and generate again.")

        candidates = report["candidates"]
        st.markdown("### 05 · Review available questions (draft)")
        if not candidates:
            st.warning("No valid questions with answers are available to export.")
            return

        import pandas as pd
        draft_rows = []
        for i, q in enumerate(candidates, 1):
            draft_rows.append({
                "Q#": i,
                "Type": q.get("question_type", "WHQ"),
                "Question": _display_question(q),
                "Answer": str(q.get("answer", "")),
                "Bloom": str(q.get("bloom", "")),
                "Difficulty": str(q.get("difficulty", "")),
            })
        draft_edit = st.data_editor(
            pd.DataFrame(draft_rows), hide_index=True,
            use_container_width=True,
            disabled=["Q#", "Type", "Bloom", "Difficulty"],
            column_config={
                "Question": st.column_config.TextColumn("Question", width="large"),
                "Answer": st.column_config.TextColumn("Answer", width="large"),
            },
            key="aqz_draft_editor",
        )

        draft_subparts = {}
        for i, q in enumerate(candidates):
            if q.get("question_type") == "WHQ" and q.get("suggested_marks", 5) > 5:
                enabled = st.checkbox(
                    f"Add subparts to draft LAQ {i + 1}",
                    value=bool(q.get("subparts")),
                    key=f"draft_subparts_on_{signature[:12]}_{i}",
                )
                draft_subparts[i] = None
                if enabled:
                    draft_subparts[i] = st.text_area(
                        "One subpart per line: question text | marks",
                        value=_subparts_text(q),
                        key=f"draft_subparts_{signature[:12]}_{i}",
                        help="If provided, subpart marks must add up to the question's section marks.",
                    )
        draft_signature = hashlib.sha256(
            json.dumps({
                "rows": draft_edit.to_dict("records"),
                "subparts": draft_subparts,
            }, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        draft_text = "\n\n".join(
            f"Q{row['Q#']} [{row['Type']}]\n{str(row['Question']).strip()}\n"
            f"Answer: {str(row['Answer']).strip()}"
            for _, row in draft_edit.iterrows()
        )
        st.markdown("#### Select or copy all draft questions and answers")
        st.caption("Click the text box, press Ctrl+A, then Ctrl+C to copy the full draft.")
        st.text_area(
            "Draft questions and answers", value=draft_text, height=280,
            key=f"aqz_all_draft_text_{signature[:8]}_{draft_signature[:8]}",
        )
        st.download_button(
            "Download all draft questions and answers (.txt)",
            data=draft_text, file_name="AQZ_Draft_Questions_and_Answers.txt",
            mime="text/plain", use_container_width=True,
        )
        st.caption("Draft export includes only the available valid questions. It does not claim to satisfy the requested exam blueprint.")
        if st.button("Prepare draft question & answer PDFs", type="primary", use_container_width=True):
            try:
                draft_paper = dict(paper)
                draft_paper["exam_name"] = (paper.get("exam_name") or "Examination") + " - DRAFT (INCOMPLETE PATTERN)"
                draft_paper["sections"] = []
                final_draft = []
                for kind in TYPE_NAMES:
                    group = draft_edit[draft_edit["Type"] == kind]
                    if group.empty:
                        continue
                    original = next((sec for sec in paper["sections"] if sec["type"] == kind), None)
                    marks = int(original["marks_each"]) if original else 1
                    if original:
                        group = group.head(int(original["offered_count"]))
                    sec_id = section_id(len(draft_paper["sections"]))
                    count = len(group)
                    answer_count = min(
                        int(original["answer_count"]) if original else count,
                        count,
                    )
                    draft_paper["sections"].append({
                        "id": sec_id,
                        "title": TYPE_NAMES[kind] + " (draft)",
                        "type": kind,
                        "offered_count": count,
                        "answer_count": answer_count,
                        "marks_each": marks,
                        "section_marks": answer_count * marks,
                    })
                    for _, row in group.iterrows():
                        question = str(row["Question"]).strip()
                        answer = str(row["Answer"]).strip()
                        if not question or not answer:
                            raise ValueError(f"Question {row['Q#']} has an empty question or answer.")
                        final_draft.append({
                            "section_id": sec_id, "question_type": kind,
                            "question": question, "answer": answer,
                            "marks": marks, "bloom": row["Bloom"],
                            "difficulty": row["Difficulty"],
                            "subparts": (
                                _parse_subparts(
                                    draft_subparts.get(int(row["Q#"]) - 1) or "",
                                    marks,
                                )
                                if kind == "WHQ" and int(row["Marks"]) > 5
                                and draft_subparts.get(int(row["Q#"]) - 1) is not None
                                else []
                            ),
                        })
                draft_paper["full_marks"] = sum(sec["section_marks"] for sec in draft_paper["sections"])
                st.session_state.aqz_draft_pdfs = {
                    "signature": signature,
                    "review_signature": draft_signature,
                    "questions": make_pdf(draft_paper, final_draft, result["logo"], answer_key=False),
                    "answers": make_pdf(draft_paper, final_draft, result["logo"], answer_key=True),
                }
                st.success("Draft PDFs are ready.")
            except Exception as exc:
                st.session_state.aqz_draft_pdfs = None
                st.error(f"Draft PDF preparation failed: {exc}")

        draft_pdfs = st.session_state.get("aqz_draft_pdfs")
        if draft_pdfs and draft_pdfs["signature"] == signature:
            if draft_pdfs["review_signature"] != draft_signature:
                st.warning("Questions changed since PDF preparation. Prepare draft PDFs again.")
            else:
                st.markdown("### 06 · Export draft")
                d1, d2 = st.columns(2)
                d1.download_button("↓ Draft Question Paper (PDF)", draft_pdfs["questions"],
                                   file_name="AQZ_Draft_Questions.pdf", mime="application/pdf",
                                   use_container_width=True)
                d2.download_button("↓ Draft Answer Key (PDF)", draft_pdfs["answers"],
                                   file_name="AQZ_Draft_Answers.pdf", mime="application/pdf",
                                   use_container_width=True)
        st.download_button("Download available candidates (JSON)",
                           data=json.dumps(candidates, indent=2, ensure_ascii=False),
                           file_name="aqz_candidates.json", mime="application/json")
        return

    selected = report["selected"]
    st.success(f"Exact blueprint matched: {len(selected)} questions across {len(paper['sections'])} sections.")
    if not result["full_scope"]:
        st.warning("Quick-test mode was used. Run again with all PDF content before issuing a paper.")

    st.markdown("### 05  ·  Review and edit questions")
    st.caption("Double-click Question or Answer cells to edit. For LAQs, add optional subparts below using one line per part: question text | marks. Subpart marks must add up to the question's marks.")
    import pandas as pd
    rows = []
    for i, item in enumerate(selected, 1):
        rows.append({
            "Q#": i,
            "Section": item["section_id"],
            "Type": item["question_type"],
            "Question": _display_question(item),
            "Answer": str(item.get("answer") or ""),
            "Marks": item["marks"],
            "Bloom": item.get("bloom", ""),
            "Difficulty": item.get("difficulty", ""),
        })
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        disabled=["Q#", "Section", "Type", "Marks", "Bloom", "Difficulty"],
        column_config={
            "Question": st.column_config.TextColumn("Question", width="large"),
            "Answer": st.column_config.TextColumn("Answer / marking points", width="large"),
        },
        key="aqz_review_editor",
    )
    reviewed_subparts = {}
    for i, item in enumerate(selected):
        if item.get("question_type") == "WHQ" and item.get("suggested_marks", item.get("marks", 5)) > 5:
            enabled = st.checkbox(
                f"Add subparts to LAQ {i + 1}",
                value=bool(item.get("subparts")),
                key=f"review_subparts_on_{signature[:12]}_{i}",
            )
            reviewed_subparts[i] = None
            if enabled:
                reviewed_subparts[i] = st.text_area(
                    "One subpart per line: question text | marks",
                    value=_subparts_text(item),
                    key=f"review_subparts_{signature[:12]}_{i}",
                    help="If provided, subpart marks must add up to the question's marks.",
                )
    review_signature = hashlib.sha256(
        json.dumps({
            "rows": edited[["Question", "Answer"]].to_dict("records"),
            "subparts": reviewed_subparts,
        },
                   sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    reviewed = st.checkbox("I have reviewed question wording and marking points for academic accuracy.")
    if st.button(
        "Prepare question paper and answer-key PDFs", type="primary",
        disabled=(not reviewed or not result["full_scope"]),
        use_container_width=True
    ):
        try:
            final = []
            for i, item in enumerate(selected):
                current = dict(item)
                current["question"] = str(edited.iloc[i]["Question"]).strip()
                current["answer"] = str(edited.iloc[i]["Answer"]).strip()
                if not current["question"] or not current["answer"]:
                    raise ValueError(
                        f"Q{i + 1} has an empty question or answer."
                    )
                current["subparts"] = (
                    _parse_subparts(
                        reviewed_subparts.get(i) or "", current["marks"]
                    )
                    if current.get("question_type") == "WHQ" and current.get("suggested_marks", current.get("marks", 5)) > 5
                    and reviewed_subparts.get(i) is not None
                    else []
                )
                final.append(current)
            st.session_state.aqz_pdfs = {
                "signature": signature,
                "questions": make_pdf(paper, final, result["logo"], answer_key=False),
                "answers": make_pdf(paper, final, result["logo"], answer_key=True),
                "final": final,
                "review_signature": review_signature,
            }
            st.success("Both PDF documents are ready for download.")
        except Exception as exc:
            st.error(f"PDF export failed: {exc}")

    pdfs = st.session_state.aqz_pdfs
    if pdfs and pdfs["signature"] == signature and pdfs["review_signature"] != review_signature:
        st.warning("Your question/answer edits changed. Click Prepare PDFs again before downloading.")
    if (pdfs and pdfs["signature"] == signature
            and pdfs["review_signature"] == review_signature):
        st.markdown("### 06  ·  Export")
        d1, d2, d3 = st.columns(3)
        slug = "".join(c if c.isalnum() or c in "-_" else "_" for c in paper["paper_code"])
        d1.download_button(
            "↓ Question Paper (PDF)", pdfs["questions"],
            file_name=f"AQZ_{slug}_Question_Paper.pdf",
            mime="application/pdf", use_container_width=True
        )
        d2.download_button(
            "↓ Answer Key (PDF)", pdfs["answers"],
            file_name=f"AQZ_{slug}_Answer_Key.pdf",
            mime="application/pdf", use_container_width=True
        )
        d3.download_button(
            "↓ Exam Blueprint (JSON)",
            json.dumps(paper, ensure_ascii=False, indent=2),
            file_name=f"AQZ_{slug}_Blueprint.json",
            mime="application/json", use_container_width=True
        )
        st.caption("Exports are based on the last saved review. If you edit questions again, click Prepare PDFs again.")


if __name__ == "__main__":
    app_main()
