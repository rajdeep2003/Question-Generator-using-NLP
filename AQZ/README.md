# AQZ Exam Paper Studio

AQZ is a local Streamlit application that turns a text-based course PDF into
candidate examination questions, checks them against an exam blueprint, lets a
teacher review the results, and exports a question paper and answer key as PDFs.

> **Important:** AQZ is an exam-paper drafting tool, not a fully autonomous
> assessment authority. Question generation, answer extraction, and Bloom
> classification use a mixture of rules and pretrained models. Review academic
> accuracy, answer keys, marks, and taxonomy labels before using a paper in an
> actual examination.

## Contents

- [Requirements](#requirements)
- [Install and run on Windows](#install-and-run-on-windows)
- [Run the application](#run-the-application)
- [Teacher workflow](#teacher-workflow)
- [Processing pipeline](#processing-pipeline)
- [Files and responsibilities](#files-and-responsibilities)
- [Models and NLP behavior](#models-and-nlp-behavior)
- [Command-line tools and tests](#command-line-tools-and-tests)
- [Known limitations](#known-limitations)
- [Sharing the project](#sharing-the-project)

## Requirements

- Windows 10/11 (or another OS with Python and the listed packages)
- Python 3.10+ recommended
- Internet access the first time a Hugging Face model is used
- RAM and disk space for PyTorch and transformer models
- A GPU is optional; the application falls back to CPU

The package list is in [`requirements.txt`](requirements.txt). It includes
Streamlit, pandas, PyTorch, Transformers, PyMuPDF, ReportLab, and some legacy
packages. The current question-paper workflow does not use Flask, spaCy, NLTK,
or python-docx directly.

MCQ fallback ranking automatically downloads the Sense2Vec Reddit 2015 vectors
the first time the fallback needs them. The archive is about 573 MB, so the
computer needs internet access and enough temporary disk space. The files are
stored under `AQZ/models/` and are excluded from Git. If the download fails,
generation continues with lexical ranking. Sense2Vec only ranks answers
already extracted from the course source; it does not add outside facts as
answer choices. Set `SENSE2VEC_MODEL_PATH` to use an already downloaded model
from another location.

## Install and run on Windows

Open PowerShell in the extracted project folder:

```powershell
cd "C:\path\to\AQZ"
```

Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in this window, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies and prefetch the Sense2Vec 2015 fallback
vectors in one step:

```powershell
python -m pip install --upgrade pip
python install.py
```

The vector archive is about 573 MB. Other Transformer model weights download
automatically the first time their feature is used. If you install dependencies
with `pip install -r requirements.txt` instead, Sense2Vec downloads automatically
the first time MCQ fallback ranking needs it.

For PyTorch, choose a CPU or CUDA build that matches the computer from the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if the default `torch` install is not suitable.

## Run the application

```powershell
python -m streamlit run app.py
```

Open the local address printed by Streamlit, normally:

```text
http://localhost:8501
```

If port 8501 is already in use:

```powershell
python -m streamlit run app.py --server.port 8502
```

### Run from the terminal without opening the website

`aqz_cli.py` accepts plain text and exam settings, then prints the selected
questions, answers/marking points, Bloom levels, and difficulty. It uses the
same generation, answer processing, classification, and selection modules as
the Streamlit app.

For longer source material, save the text as `source.txt` and run:

```powershell
python aqz_cli.py --text-file .\source.txt --type WHQ --offered 12 --answer 10 --marks 1 --subject "Artificial Intelligence"
```

Or paste text directly into PowerShell:

```powershell
@'
Paste the course text here. Include a few explanatory paragraphs for useful
question candidates.
'@ | python aqz_cli.py --type WHQ --offered 4 --answer 2 --marks 5
```

Supported `--type` values are `WHQ`, `FILL_BLANK`, `TRUE_FALSE`, and `MCQ`. Use `--blueprint .\examples\exam_blueprint.json` with `--text-file` to run the
sections and quotas from a blueprint. Add `--use-t5` to include T5 question
generation; otherwise AQZ uses its rules-based generator. The answer and
classification steps may still load their configured Transformer models when
needed, including downloading them on first use. Add `--json .\result.json`
to save a machine-readable result. Run `python aqz_cli.py --help` for options.

### Generation modes

- **Source-based rules only**: faster and does not load the T5 question model.
  Uses source patterns and templates; candidate coverage and wording depend on
  the uploaded material.
- **T5 + source-based rules**: additionally runs the configured T5 question
  generation model. It may be substantially slower on CPU and downloads model
  files the first time.

For a complete paper, enable **Use all PDF content**. Quick-test mode uses only
the first two chunks and is not intended for final-paper export.

## Teacher workflow

1. Enter institution, course, examination, subject, date, and duration details.
2. Upload a text-based PDF (maximum 25 MB).
3. Configure each section's type, offered count, answer-any count, and marks.
4. Optionally enforce Bloom or difficulty distributions. Each target
   distribution must add up to the number of questions offered.
5. Generate candidates and review the reported counts/shortages.
6. If exact blueprint matching succeeds, review the selected questions and
   answers. Extended WH-question subparts are optional.
7. Prepare and download the question-paper PDF, answer-key PDF, and blueprint
   JSON.

Supported section types:

- `FILL_BLANK` — Fill in the Blanks
- `TRUE_FALSE` — True / False
- `MCQ` — Four-option multiple choice
- `WHQ` — WH-Questions (very short, short, and long answers, distinguished by marks)

If AQZ cannot satisfy the blueprint, it displays available candidates as a
**draft** instead of silently repeating questions or claiming an exact match.
A draft is explicitly marked incomplete. It should not be treated as a
blueprint-compliant final paper.

PDFs are built in memory. Uploaded PDFs are written to a temporary file only
for extraction and that temporary file is deleted afterward. Scanned/image-only
PDFs need OCR first; OCR is not included.

## Processing pipeline

```text
Streamlit form + exam section settings
                 |
                 v
PDF validation (size, signature, password, selectable text)
                 |
                 v
PyMuPDF text extraction
                 |
                 v
Text cleanup, headings, sentence splitting, 180-word chunks
                 |
                 v
Candidate generation
  - T5-generated questions (optional)
  - source-grounded templates/rules
  - fill-in-the-blank source phrases
  - true/false source assertions
                 |
                 v
Deduplication, question-type inference, answer processing
                 |
                 v
Bloom classification and difficulty estimation
                 |
                 v
Blueprint validation and exact quota selection
                 |
                 v
Teacher review / optional edits
                 |
                 v
ReportLab question-paper PDF + answer-key PDF + blueprint JSON
```

### Pipeline details

1. **PDF validation:** `app.py` checks the upload size, PDF signature, password
   protection, page count, and whether selectable text is present.
2. **Extraction:** `pdf_service.py` extracts selectable page text with PyMuPDF.
3. **Cleaning and chunking:** `text_processor.py` removes common PDF artifacts,
   preserves section boundaries, splits by sentence, and chunks using defaults
   of 180 words with 30 words of overlap (`config.py`).
4. **Question generation:** `question_generator.py` combines optional T5
   output with source-grounded templates. It handles WHQ across answer lengths, fill-ups,
   true/false candidates, and MCQs. MCQs use `valhalla/t5-small-qa-qg-hl` via
   the supplied MultiTaskQAQGPipeline format to generate answer/question pairs,
   then `fares7elsadek/t5-base-distractor-generation` to generate three choices.
   The rules-only mode can instead use distinct source definitions as choices.
   Some WHQ patterns are heuristic.
5. **Processing:** `question_processor.py` removes duplicates and malformed
   items, infers question type where needed, obtains missing answers,
   and attaches Bloom/difficulty metadata.
6. **Answer handling:** Answers already derived from source sentences or
   true/false rules are retained. For eligible missing factual answers,
   `answer_generator.py` uses extractive question answering and rejects answers
   that are unsupported by the supplied context.
7. **Classification:** `classifier.py` first checks cognitive-action rules.
   It uses BART zero-shot classification only when no rule matches. Difficulty
   is primarily estimated from the Bloom level and complexity keywords.
8. **Blueprint matching:** `exam_blueprint.py` validates metadata, sections,
   counts, marks, and optional Bloom/difficulty quotas. The selector in
   `question_processor.py` uses a flow algorithm to select candidates meeting
   both quota types without reusing questions.
9. **Export:** `app.py` builds print-layout PDFs in memory with ReportLab. It
   exports a question paper and a separate answer key/marking guide.

## Files and responsibilities

| File | Purpose |
|---|---|
| `app.py` | Streamlit launcher |
| `aqz/application/streamlit_app.py` | Main UI, orchestration, review, PDF export |
| `aqz/ingestion/pdf_service.py` | PDF text extraction |
| `aqz/preprocessing/text_processor.py` | Text cleanup, sentence splitting, chunking |
| `aqz/generation/question_generator.py` | T5 integration and source-based generation |
| `aqz/processing/question_processor.py` | Deduplication, answer handling, metadata, selection |
| `aqz/analysis/answer_generator.py` | Source-checked extractive answers |
| `aqz/analysis/classifier.py` | Bloom classification and difficulty estimation |
| `aqz/blueprint/exam_blueprint.py` | Blueprint validation and CLI |
| `aqz/export/pdf_exporter.py` | Legacy standalone PDF exporter |
| `aqz/config.py` | Model, device, chunk, and taxonomy settings |
| `aqz_cli.py` | Plain-text terminal entry point |
| `examples/exam_blueprint.json` | Sample paper configuration |
| `outputs/aqz_result.json` | Example output payload |
| `tests/test_text_processor.py` | Text processor demonstration script |
| `archive/legacy/` | Prior app backup and T5 demo |
| `requirements.txt` | Python dependencies |

Use `app.py` as the application entry point. The implementation is grouped by pipeline stage under `aqz/`. Root-level `text_processor.py`, `question_generator.py`, and `question_processor.py` remain thin compatibility launchers. The old demo and backup app are under `archive/legacy/`; `pdf_exporter.py` is retained under `aqz/export/` as a legacy path-based exporter, while the Streamlit UI owns the current in-memory exports.

## Repository layout

```text
app.py                         Streamlit launcher
 aqz/
   application/streamlit_app.py UI, orchestration, review and in-memory PDF exports
   ingestion/pdf_service.py     PDF text extraction
   preprocessing/text_processor.py cleanup and chunking
   generation/question_generator.py rule and optional T5 question generation
   processing/question_processor.py deduplication, answers, metadata and selection
   analysis/answer_generator.py extractive answers
   analysis/classifier.py       Bloom and difficulty classification
   blueprint/exam_blueprint.py  paper schema and validation
   export/pdf_exporter.py       legacy standalone exporter
   config.py                    model and chunk configuration
 examples/                      sample blueprint
 outputs/                       checked-in example result
 tests/                         text processor tests
 archive/legacy/                prior app and T5 demo
```

Run the app from the repository root with `python -m streamlit run app.py`. Stage modules can be launched with `python -m aqz.preprocessing.text_processor --self-test`, `python -m aqz.generation.question_generator ...`, `python -m aqz.processing.question_processor ...`, and `python -m aqz.blueprint.exam_blueprint ...`.

## Models and NLP behavior

Model names are configured in `config.py`:

| Purpose | Model |
|---|---|
| Question generation | `valhalla/t5-base-e2e-qg` |
| MCQ question and answer generation | `valhalla/t5-small-qa-qg-hl` |
| MCQ distractors | `fares7elsadek/t5-base-distractor-generation` |
| Bloom zero-shot fallback | `facebook/bart-large-mnli` |
| Extractive answer generation | `deepset/roberta-base-squad2` |

The T5 model is lazy-loaded only when T5 mode is selected. BART and RoBERTa are
also loaded only when a processed question needs them. PyTorch selects CUDA
when available and CPU otherwise.

This project uses a **hybrid NLP approach**, not a general-purpose chat model:

- regular expressions and templates handle many question patterns;
- Transformer models assist with question generation, Bloom fallback, and
  short factual answer extraction;
- the answer extractor does not write essays;
- spaCy and NLTK are listed as dependencies but are not currently called by
  the main pipeline.

Bloom and difficulty values are estimates. Rule-based classification looks
mainly at question wording, so a correct label is not guaranteed from a verb
alone. Teacher review is required, especially for higher-order taxonomy and
long-answer questions.

## Command-line tools and tests

Run offline self-tests (no model download should be needed for these):

```powershell
python -m aqz.preprocessing.text_processor --self-test
python -m aqz.generation.question_generator --self-test
python -m aqz.processing.question_processor --self-test
python -m aqz.analysis.classifier --self-test
python -m aqz.analysis.answer_generator --self-test
python -m aqz.blueprint.exam_blueprint --self-test
```

Compile the main Python files:

```powershell
python -m compileall -q app.py aqz_cli.py text_processor.py question_generator.py question_processor.py aqz
```

Blueprint examples:

```powershell
python -m aqz.blueprint.exam_blueprint --demo
python -m aqz.blueprint.exam_blueprint --validate examples/exam_blueprint.json
python -m aqz.blueprint.exam_blueprint --write-sample my_blueprint.json
```

Rules-only question generation from a PDF:

```powershell
python -m aqz.generation.question_generator --pdf "C:\path\to\course.pdf" --rules-only --max-chunks 0
```

T5 plus rule-based generation from a PDF:

```powershell
python -m aqz.generation.question_generator --pdf "C:\path\to\course.pdf" --max-chunks 0
```

Process a PDF and optionally try selecting against the example blueprint:

```powershell
python -m aqz.processing.question_processor --pdf "C:\path\to\course.pdf" --max-chunks 0 --blueprint examples/exam_blueprint.json
```

The question-processor CLI uses transformer models for some questions and may
download model files. `--max-chunks 0` means process all chunks; the CLI's
default is one chunk.

Test an extractive answer:

```powershell
python -m aqz.analysis.answer_generator --question "What does the source define?" --context "Paste a short source passage here."
```

Test classification:

```powershell
python -m aqz.analysis.classifier --question "Compare two approaches."
```

## Known limitations

- Scanned PDFs require OCR, which is not included.
- Source-based generation is rule/template driven and may produce too few
  candidates or repetitive/weak wording for some subjects.
- T5 is primarily used for short factual question generation; it does not
  produce a reliable essay answer.
- Long-answer marking points are source excerpts, not generated model answers.
- Bloom/difficulty labels are heuristic estimates and need teacher validation.
- The free-form teacher prompt is stored but is not enforced by the generator.
- The app caps processed candidates at 80 and T5 passages at 24 per run for
  performance.
- The default UI generation mode is rules-only for speed. T5 mode is slower,
  especially on CPU.
- A blueprint shortage is intentional behavior: AQZ should not invent
  candidates or duplicate questions merely to fill the requested counts.
- The sample `examples/exam_blueprint.json` is an example, not a required configuration.

## Sharing the project

The companion archive `AQZ_Team_Package.zip` contains the README, Python source
files, requirements, and example blueprint. It excludes Python cache files.
Extract the ZIP, follow [Install and run on Windows](#install-and-run-on-windows),
then launch with `python -m streamlit run app.py`.

Do not include real student data, credentials, or confidential course materials
when sharing the project archive.


