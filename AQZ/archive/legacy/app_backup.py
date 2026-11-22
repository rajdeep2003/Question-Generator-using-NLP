
# # from transformers import pipeline

# # print("Loading model...")

# # make_questions = pipeline(
# #     "text2text-generation",
# #     model="valhalla/t5-base-e2e-qg"
# # )

# # print("Model loaded!")

# # # A relatively detailed educational paragraph
# # paragraph = """
# # Artificial Intelligence (AI) is a branch of computer science
# # that focuses on developing machines capable of performing
# # tasks that normally require human intelligence. These tasks
# # include learning, reasoning, problem-solving, decision-making,
# # and understanding natural language.

# # Machine Learning is a subset of Artificial Intelligence that
# # enables computers to learn patterns from data without being
# # explicitly programmed for every task. Deep Learning is a
# # specialized area of Machine Learning that uses artificial
# # neural networks inspired by the structure of the human brain.

# # In 1956, the term Artificial Intelligence was introduced
# # during the Dartmouth Conference, which is considered an
# # important milestone in the development of AI. John McCarthy
# # is widely known as the father of Artificial Intelligence.

# # AI is widely used in healthcare, education, banking,
# # transportation, and agriculture. In healthcare, AI helps
# # doctors analyze medical images and predict diseases.
# # In education, intelligent systems provide personalized
# # learning experiences to students.

# # Natural Language Processing (NLP) is another important
# # field of AI that enables computers to understand, interpret,
# # and generate human language. Applications of NLP include
# # chatbots, language translation, sentiment analysis,
# # and speech recognition.

# # Despite its advantages, Artificial Intelligence also
# # creates challenges related to data privacy, security,
# # job displacement, and ethical decision-making.
# # """

# # print("\nGenerating questions...\n")

# # result = make_questions(
# #     paragraph,
# #     max_new_tokens=256,
# #     num_beams=5,
# #     do_sample=False
# # )

# # # Extract individual questions
# # raw_output = result[0]["generated_text"]

# # questions = [
# #     q.strip()
# #     for q in raw_output.split("<sep>")
# #     if q.strip()
# # ]

# # print("Generated Questions:\n")

# # for i, question in enumerate(questions[:10], 1):
# #     print(f"Q{i}. {question}")

# # print("\nTotal questions generated:", min(len(questions), 10))

# # if len(questions) < 10:
# #     print("Note: Model generated fewer than 10 questions.")




# import os
# import tkinter as tk

# from tkinter import filedialog, simpledialog, messagebox

# from aqz.ingestion.pdf_service import extract_pdf_text

# from aqz.preprocessing.text_processor import (
#     preprocess_text,
#     chunk_text
# )

# from aqz.generation.question_generator import generate_questions

# from aqz.processing.question_processor import (
#     remove_duplicates,
#     process_questions,
#     select_questions
# )

# from pdf_exporter import export_question_paper


# def main():

#     root = tk.Tk()

#     root.withdraw()

#     messagebox.showinfo(
#         "Question Paper Generator",
#         "Select an educational PDF."
#     )

#     pdf_path = input(
#     "\nEnter your PDF file location: "
# ).strip().strip('"')

#     if not os.path.isfile(pdf_path):
#         print("PDF file not found!")
#         return

#     if not pdf_path.lower().endswith(".pdf"):
#         print("Please provide a PDF file!")
#         return

#     if not pdf_path:
#         return

#     required_count = simpledialog.askinteger(
#         "Question Count",
#         "Enter number of questions:",
#         minvalue=1,
#         maxvalue=100
#     )

#     if not required_count:
#         return

#     try:

#         print("\n1. Extracting PDF text...")

#         text = extract_pdf_text(pdf_path)

#         text = preprocess_text(text)

#         if not text:
#             raise ValueError("No text found in PDF.")

#         print("\n2. Splitting text into chunks...")

#         chunks = chunk_text(text)

#         print("Total chunks:", len(chunks))

#         print("\n3. Generating questions...")

#         questions = generate_questions(chunks)

#         questions = remove_duplicates(questions)

#         print("Unique questions:", len(questions))

#         if not questions:
#             raise ValueError("No questions generated.")

#         print("\n4. Classifying and generating answers...")

#         results = process_questions(questions)

#         print("\n5. Selecting final questions...")

#         final_results = select_questions(
#             results,
#             required_count
#         )

#         print("\n6. Exporting question paper...")

#         output_folder = os.path.join(
#             os.path.dirname(pdf_path),
#             "outputs"
#         )

#         os.makedirs(output_folder, exist_ok=True)

#         output_path = os.path.join(
#             output_folder,
#             "generated_question_paper.pdf"
#         )

#         export_question_paper(
#             final_results,
#             output_path
#         )

#         print("\nPROCESS COMPLETED!")

#         print("Generated:", len(final_results))

#         messagebox.showinfo(
#             "Success",
#             f"Question paper created!\n\n{output_path}"
#         )

#     except Exception as error:

#         print("ERROR:", error)

#         messagebox.showerror(
#             "Application Error",
#             str(error)
#         )

#     finally:

#         root.destroy()


# if __name__ == "__main__":

#     main()



"""AQZ Exam Paper Studio - Streamlit front end for the existing AQZ backend.

Run: streamlit run app.py
Implementation modules are organized under the `aqz/` package.

Requires: streamlit, reportlab, pymupdf and the existing AQZ dependencies.
"""

import hashlib
import io
import json
import os
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
    "VSAQ": "Very Short Answer",
    "WHQ": "WH-Questions",
    
    "TRUE_FALSE": "True / False",
}
DIFFICULTIES = ("Easy", "Medium", "Hard")
BLOOMS = ("Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] {background: #f5f7fc; color: #15213c;}
.block-container {padding-top: 1.75rem; padding-bottom: 3.5rem; max-width: 1280px;}
[data-testid="stSidebar"] {background: #101e3a;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] h3 {color: #e4eaff !important;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #b6c5e0;}
h1, h2, h3 {font-family: 'Outfit', sans-serif !important; letter-spacing:-.02em;}
h1 {color:#15213c !important;} h2, h3 {color:#203054 !important;}
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
            new_section(1, "VSAQ", 12, 10, 1),
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


def make_pdf(paper, questions, logo_bytes=None, answer_key=False):
    """Create a fully separate PDF in memory; no public output files required."""
    out = io.BytesIO()
    doc = SimpleDocTemplate(
        out, pagesize=A4, rightMargin=42, leftMargin=42,
        topMargin=43, bottomMargin=48,
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
        fontSize=11.5, leading=16, textColor=navy,
        spaceBefore=14, spaceAfter=4
    )
    small = ParagraphStyle(
        "AQZSmall", parent=body, fontSize=8.5, leading=12,
        textColor=muted, spaceAfter=8
    )
    answer_style = ParagraphStyle(
        "AQZAnswer", parent=body, leftIndent=21, spaceAfter=11,
        textColor=colors.HexColor("#334155")
    )

    story = []
    if logo_bytes:
        try:
            from reportlab.lib.utils import ImageReader
            reader = ImageReader(io.BytesIO(logo_bytes))
            w, h = reader.getSize()
            scale = min(57 / w, 57 / h)
            story.append(Image(io.BytesIO(logo_bytes), width=w * scale, height=h * scale, hAlign="CENTER"))
            story.append(Spacer(1, 6))
        except Exception:
            pass  # Invalid logo should not prevent PDF export.

    story.append(Paragraph(_xml(paper["college_name"]), headline))
    story.append(Paragraph(_xml(paper["college_address"]), center))
    if paper.get("university_name"):
        story.append(Paragraph(_xml(paper["university_name"]), center))
    story.append(Spacer(1, 8))
    story.append(Paragraph(_xml(paper.get("exam_name") or "Examination"), center))
    story.append(Paragraph(
        "<b>" + ("ANSWER KEY &amp; MARKING GUIDE" if answer_key else "QUESTION PAPER") + "</b>",
        center
    ))
    story.append(Spacer(1, 14))

    table_rows = [
        ["Course / Branch", f"{paper['course_name']} / {paper.get('branch', '')}"],
        ["Paper", f"{paper['subject_name']} ({paper['paper_code']})"],
        ["Semester / Academic Year", f"{paper['semester']} / {paper['academic_year']}"],
        ["Exam Date", paper["exam_date"]],
        ["Duration", f"{paper['duration_minutes']} minutes"],
        ["Full Marks", str(paper["full_marks"])],
    ]
    pdf_table = Table(
        [[Paragraph(_xml(a), small), Paragraph(_xml(b), body)] for a, b in table_rows],
        colWidths=[145, doc.width - 145], hAlign="CENTER"
    )
    pdf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4fb")),
        ("LINEBELOW", (0, 0), (-1, -2), .35, colors.HexColor("#dde5f1")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(pdf_table)
    story.append(Spacer(1, 10))

    if not answer_key:
        for instruction in paper.get("instructions", []):
            story.append(Paragraph("&#8226; " + _xml(instruction), small))
        story.append(Spacer(1, 6))

    n = 1
    for section in paper["sections"]:
        kind = section["type"]
        story.append(Paragraph(
            f"SECTION {_xml(section['id'])} - {_xml(section['title'])}",
            section_style
        ))
        directive = (
            f"Answer any {section['answer_count']} of {section['offered_count']} "
            f"questions. {section['marks_each']} mark(s) each. "
            f"Section marks: {section['section_marks']}."
        )
        story.append(Paragraph(_xml(directive), small))
        for item in questions:
            if item.get("section_id") != section["id"]:
                continue
            q = _xml(item["question"])
            heading = f"<b>Q{n}.</b> {q}"
            if not answer_key:
                heading += f" <font color='#66758d'>[{section['marks_each']}]</font>"
                story.append(KeepTogether([
                    Paragraph(heading, body), Spacer(1, 9)
                ]))
            else:
                story.append(Paragraph(heading, body))
                answer = item.get("answer") or "Answer not provided."
                story.append(Paragraph("<b>Answer / marking points:</b> " + _xml(answer), answer_style))
                story.append(Paragraph(
                    f"Bloom: {_xml(item.get('bloom', 'N/A'))} | "
                    f"Difficulty: {_xml(item.get('difficulty', 'N/A'))} | "
                    f"Marks: {section['marks_each']}",
                    small
                ))
            n += 1

    def footer(canvas, document):
        canvas.saveState()
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


def run_pipeline(pdf_bytes, paper, use_t5=True, quick_test=False, log=None):
    """Returns report and, if feasible, an exact blueprint-matched selection."""
    from aqz.generation.question_generator import generate_questions
    from aqz.processing.question_processor import process_questions, select_questions, QuestionSelectionError

    if log:
        log("Extracting and cleaning uploaded PDF...", .18)
    chunks = read_pdf_chunks(pdf_bytes)
    if quick_test:
        chunks = chunks[:2]
    if not chunks:
        raise ValueError("No usable text found. Please upload a text-based PDF (not scanned images).")

    if log:
        log(f"Generating candidates from {len(chunks)} chunks...", .38)
    candidates = generate_questions(chunks, use_t5=use_t5)
    if not candidates:
        raise ValueError("No questions generated from the document.")

    if log:
        log("Extracting answers and classifying questions...", .63)
    processed = process_questions(candidates)

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
        uploaded = st.file_uploader(
            "Upload a text-based PDF (7–8 pages or more)", type=["pdf"],
            help="Use course notes, learning material or a syllabus with explanations. Scanned PDFs need OCR.",
            key="source_pdf"
        )
        page_count = None
        source_bytes = None
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
            index=0, horizontal=False,
            help="The first option uses your existing local T5 and RoBERTa models."
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
    can_generate = bool(source_bytes and paper)
    signature = None
    if can_generate:
        logo_for_signature = logo_file.getvalue() if logo_file else b""
        signature = hashlib.sha256(
            source_bytes + logo_for_signature
            + json.dumps(paper, sort_keys=True).encode()
            + engine.encode() + str(full_scope).encode()
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
                quick_test=not full_scope, log=update,
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
        st.info("Upload a PDF, adjust your sections and click Generate. No model is loaded until you start generation.")
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
        st.info("Try adding richer course content, reducing requested counts, or relaxing optional difficulty/Bloom quotas. AQZ will not silently repeat questions to fill a section.")
        st.download_button(
            "Download available candidates (JSON)",
            data=json.dumps(report["candidates"], indent=2, ensure_ascii=False),
            file_name="aqz_candidates.json", mime="application/json"
        )
        return

    selected = report["selected"]
    st.success(f"Exact blueprint matched: {len(selected)} questions across {len(paper['sections'])} sections.")
    if not result["full_scope"]:
        st.warning("Quick-test mode was used. Run again with all PDF content before issuing a paper.")

    st.markdown("### 05  ·  Review and edit questions")
    st.caption("Double-click Question or Answer cells to edit. The same edits will appear in the exported papers. Bloom and marks metadata cannot be changed here.")
    import pandas as pd
    rows = []
    for i, item in enumerate(selected, 1):
        rows.append({
            "Q#": i,
            "Section": item["section_id"],
            "Type": item["question_type"],
            "Question": str(item["question"]),
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
    review_signature = hashlib.sha256(
        json.dumps(edited[["Question", "Answer"]].to_dict("records"),
                   sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    reviewed = st.checkbox("I have reviewed question wording and marking points for academic accuracy.")
    if st.button(
        "Prepare question paper and answer-key PDFs", type="primary",
        disabled=(not reviewed or not result["full_scope"]),
        use_container_width=True
    ):
        final = []
        for i, item in enumerate(selected):
            current = dict(item)
            current["question"] = str(edited.iloc[i]["Question"]).strip()
            current["answer"] = str(edited.iloc[i]["Answer"]).strip()
            if not current["question"] or not current["answer"]:
                st.error(f"Q{i + 1} has an empty question or answer. Please fix it.")
                st.stop()
            final.append(current)
        try:
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
