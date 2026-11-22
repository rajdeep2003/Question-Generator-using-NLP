"""Run AQZ's text-to-question workflow without Streamlit."""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

TYPE_NAMES = {
    "FILL_BLANK": "Fill in the Blanks",
    "TRUE_FALSE": "True / False", "WHQ": "WH-Questions",
    "MCQ": "Multiple choice",
}

def _parser():
    parser = argparse.ArgumentParser(description="Generate and select AQZ exam questions from plain text.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", help="Source text.")
    source.add_argument("--text-file", type=Path, help="UTF-8 text file.")
    parser.add_argument("--blueprint", type=Path)
    parser.add_argument("--type", choices=tuple(TYPE_NAMES), default="WHQ")
    parser.add_argument("--offered", type=int, default=5)
    parser.add_argument("--answer", type=int, default=3)
    parser.add_argument("--marks", type=int, default=5)
    parser.add_argument("--title")
    parser.add_argument("--subject", default="Terminal test")
    parser.add_argument("--paper-code", default="AQZ-CLI")
    parser.add_argument("--course", default="Course")
    parser.add_argument("--duration", type=int, default=180)
    parser.add_argument("--college", default="AQZ CLI")
    parser.add_argument("--use-t5", action="store_true")
    parser.add_argument("--json", dest="json_path", type=Path)
    return parser

def _read_text(args, parser):
    if args.text is not None: text = args.text
    elif args.text_file is not None:
        try: text = args.text_file.read_text(encoding="utf-8")
        except OSError as exc: parser.error(f"Cannot read text file: {exc}")
    else:
        if sys.stdin.isatty(): print("Paste source text, then press Ctrl+Z and Enter (Windows) or Ctrl+D (macOS/Linux):", file=sys.stderr)
        text = sys.stdin.read()
    if not text.strip(): parser.error("No source text supplied.")
    return text

def _make_blueprint(args):
    if args.blueprint:
        from aqz.blueprint.exam_blueprint import load_blueprint
        return load_blueprint(args.blueprint)
    if min(args.offered, args.answer, args.marks, args.duration) < 1 or args.answer > args.offered:
        raise ValueError("Counts, marks and duration must be positive; answer count cannot exceed offered count.")
    return {
        "college_name": args.college, "college_address": "Not specified", "course_name": args.course,
        "semester": "Not specified", "academic_year": str(date.today().year),
        "subject_name": args.subject, "paper_code": args.paper_code, "exam_date": date.today().isoformat(),
        "duration_minutes": args.duration,
        "sections": [{"id": "A", "title": args.title or TYPE_NAMES[args.type], "type": args.type,
                      "offered_count": args.offered, "answer_count": args.answer, "marks_each": args.marks}],
    }

def main():
    parser = _parser(); args = parser.parse_args(); text = _read_text(args, parser)
    try: blueprint = _make_blueprint(args)
    except (ValueError, OSError) as exc: parser.error(str(exc))
    from aqz.blueprint.exam_blueprint import validate_blueprint
    from aqz.generation.question_generator import generate_questions
    from aqz.processing.question_processor import QuestionSelectionError, process_questions, select_questions
    from aqz.preprocessing.text_processor import chunk_text, preprocess_text
    paper = validate_blueprint(blueprint)
    chunks = chunk_text(preprocess_text(text))
    if not chunks: parser.error("Supplied text did not contain enough usable material.")
    print(f"Text prepared as {len(chunks)} chunk(s). Generating candidates...")
    candidates = generate_questions(chunks, use_t5=args.use_t5)
    print("Answering and classifying candidates...")
    processed = process_questions(candidates)
    try: selected = select_questions(processed, paper); shortage = None
    except QuestionSelectionError as exc: selected = []; shortage = str(exc)
    print(f"\nAQZ TERMINAL RESULT\nSubject: {paper['subject_name']} | Paper: {paper['paper_code']}\nGenerated candidates: {len(processed)}")
    if shortage:
        print(f"Pattern not filled: {shortage}")
        for i, item in enumerate(processed, 1): print(f"{i}. [{item.get('question_type')}] {item['question']}\n   Answer: {item.get('answer','')}")
    else:
        for i, item in enumerate(selected, 1): print(f"\n{i}. [{item['question_type']}] {item['question']} [{item['marks']} marks]\n   Answer: {item.get('answer') or 'Review manually'}\n   Bloom: {item.get('bloom')} | Difficulty: {item.get('difficulty')}")
    payload = {"paper": paper, "generated_candidate_count": len(processed), "shortage": shortage,
               "selected_questions": selected, "candidates": processed if shortage else []}
    if args.json_path:
        args.json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON result saved to: {args.json_path}")
    return 2 if shortage else 0

if __name__ == "__main__": raise SystemExit(main())
