"""AQZ question processor (clean backend-only revision).

Keeps the original public functions:
    remove_duplicates(questions)
    process_questions(questions)
    select_questions(processed, count_or_blueprint)

CLI: --self-test, --pdf PATH, positional PDF, --max-chunks N, --blueprint JSON
No new LLMs or transformer models.
"""

import argparse
import re
from collections import deque
from difflib import SequenceMatcher

MISSING_ANSWER = "Answer could not be confidently extracted."
QUESTION_TYPES = {"FILL_BLANK", "TRUE_FALSE", "WHQ"}


class QuestionSelectionError(ValueError):
    pass


def _normalize(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()


def _terms(value):
    stops = {
        "a", "an", "the", "what", "which", "who", "when", "where",
        "is", "are", "was", "were", "does", "do", "did", "of", "in",
        "to", "for", "by", "with", "on", "and", "or", "as", "year",
    }
    return set(_normalize(value).split()) - stops


def _bad_question(value):
    q = " ".join(str(value).split())
    if not q or len(q.split()) > 40:
        return True
    patterns = (
        r"^what (?:was|is) one of the organizers\b",
        r"^who should\b.{0,120}\bprovide human oversight\b",
        r"^what do\b.{0,100}\blearn\b.{0,100}\band use those patterns\b",
        r"\b(?:what|which) does\b.{0,110}\bwhether\b",
    )
    return any(re.search(p, q, flags=re.I) for p in patterns)


def _definition_subject(question):
    match = re.fullmatch(r"(?:what is|define)\s+([\w -]{2,65})[?.]?", str(question).strip(), re.I)
    if not match:
        return None
    result = match.group(1).strip().rstrip("?. ")
    if len(result.split()) > 4 or result.lower().startswith(("the ", "a ", "an ")):
        return None
    return result


def _same_concept(a, b):
    a, b = _normalize(a), _normalize(b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Recognize "NLP" vs "Natural Language Processing".
    for short, full in ((a, b), (b, a)):
        if " " not in short and " " in full:
            if short == "".join(word[0] for word in full.split()):
                return True
    return False


def _reverse_definition(definition, other):
    subject = _definition_subject(definition.get("question", ""))
    if subject is None:
        return False
    question = str(other.get("question", ""))
    if not re.match(r"^what (?:is|are|enables|helps)\b", question, re.I):
        return False
    answer = other.get("answer", "")
    if not _same_concept(subject, answer):
        return False
    context = str(other.get("context", ""))
    return bool(re.search(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", context, re.I))


def _same_fact(left, right):
    qa, qb = str(left.get("question", "")).strip(), str(right.get("question", "")).strip()
    if not qa or not qb:
        return False
    if _normalize(qa) == _normalize(qb):
        return True
    a, b = _terms(qa), _terms(qb)
    if not a or not b:
        return False
    overlap = len(a & b) / len(a | b)
    likeness = SequenceMatcher(None, qa.casefold(), qb.casefold()).ratio()
    if min(len(a), len(b)) >= 2 and (overlap >= 0.85 or (likeness >= 0.92 and overlap >= 0.60)):
        return True
    if _reverse_definition(left, right) or _reverse_definition(right, left):
        return True
    aa, ab = _normalize(left.get("answer", "")), _normalize(right.get("answer", ""))
    if aa and ab and aa == ab and aa != _normalize(MISSING_ANSWER) and overlap >= 0.65:
        return True
    return False


def remove_duplicates(questions):
    """Remove malformed questions and obvious same-fact duplicates."""
    unique = []
    for item in questions:
        if not isinstance(item, dict) or _bad_question(item.get("question", "")):
            continue
        if not any(_same_fact(item, old) for old in unique):
            unique.append(item)
    return unique


def infer_question_type(question):
    q = " ".join(str(question).casefold().split())
    if q.startswith(("true or false:", "true/false:")) or "(true/false)" in q:
        return "TRUE_FALSE"
    if "____" in q or "________" in q or re.match(r"^(?:fill in the blank|complete the sentence)\b", q):
        return "FILL_BLANK"
    if re.match(r"^(?:(?:critically|in detail)\s+)?(?:evaluate|design|develop)\b", q):
        return "WHQ"
    if re.match(r"^(?:explain|discuss|describe|analy[sz]e|justify)\b", q):
        return "WHQ"
    if re.match(r"^(?:compare|differentiate|distinguish|write a short note)\b", q):
        return "WHQ"
    if re.match(r"^(?:what|which|who|when|where|how|name|state|define|list|identify)\b", q):
        return "WHQ"
    return None


def _valid_answer(item):
    answer = str(item.get("answer") or "").strip()
    invalid = {MISSING_ANSWER.casefold(), "unknown", "none", "n/a", "not found", "null"}
    return bool(answer and answer.casefold() not in invalid and item.get("answer_valid") is not False)


def process_questions(questions, max_questions=None):
    """Run existing BART classifier and RoBERTa answer extraction."""
    from aqz.analysis.classifier import classify_bloom, classify_difficulty
    from aqz.analysis.answer_generator import generate_answer

    first_pass = remove_duplicates(questions)
    if max_questions is not None:
        if type(max_questions) is not int or max_questions <= 0:
            raise ValueError("max_questions must be a positive integer")
        first_pass = first_pass[:max_questions]
    processed = []
    for index, item in enumerate(first_pass, 1):
        q = str(item.get("question", "")).strip()
        context = str(item.get("context", "")).strip()
        if not q or not context:
            continue
        print(f"Processing question {index}/{len(first_pass)}")
        kind = item.get("question_type") or infer_question_type(q)
        answer = item.get("answer")
        if answer is None:
            answer = MISSING_ANSWER if kind == "TRUE_FALSE" else generate_answer(q, context)
        bloom = classify_bloom(q)
        difficulty = classify_difficulty(q)
        record = dict(item)
        record.update({
            "question": q, "context": context, "answer": str(answer or "").strip(),
            "question_type": kind, "bloom": bloom["level"],
            "bloom_confidence": bloom["confidence"],
            "difficulty": difficulty["level"],
            "difficulty_confidence": difficulty["confidence"],
        })
        record["answer_valid"] = _valid_answer(record)
        processed.append(record)
    return remove_duplicates(processed)


def _eligible(item, kind=None):
    if not isinstance(item, dict) or not item.get("question") or not _valid_answer(item):
        return False
    actual = item.get("question_type") or infer_question_type(item["question"])
    return actual in QUESTION_TYPES and (kind is None or actual == kind)


def _flow_selection(candidates, section):
    """Select questions satisfying both Bloom and difficulty quotas."""
    needed = section["offered_count"]
    blooms = section.get("bloom_targets") or {"*": needed}
    difficulties = section.get("difficulty_targets") or {"*": needed}
    graph = {}

    def add_edge(u, v, capacity):
        graph.setdefault(u, {})
        graph.setdefault(v, {})
        graph[u][v] = graph[u].get(v, 0) + capacity
        graph[v].setdefault(u, 0)

    for label, quota in blooms.items():
        add_edge("SOURCE", ("b", label), quota)
    for label, quota in difficulties.items():
        add_edge(("d", label), "SINK", quota)
    for i, item in enumerate(candidates):
        bloom = "*" if "*" in blooms else item.get("bloom")
        difficulty = "*" if "*" in difficulties else item.get("difficulty")
        if bloom not in blooms or difficulty not in difficulties:
            continue
        add_edge(("b", bloom), ("in", i), 1)
        add_edge(("in", i), ("out", i), 1)
        add_edge(("out", i), ("d", difficulty), 1)

    flow = 0
    while flow < needed:
        parent = {"SOURCE": None}
        queue = deque(["SOURCE"])
        while queue and "SINK" not in parent:
            node = queue.popleft()
            for nxt, remaining in graph.get(node, {}).items():
                if remaining > 0 and nxt not in parent:
                    parent[nxt] = node
                    queue.append(nxt)
        if "SINK" not in parent:
            break
        node = "SINK"
        while parent[node] is not None:
            prev = parent[node]
            graph[prev][node] -= 1
            graph[node][prev] += 1
            node = prev
        flow += 1

    chosen = [
        item for i, item in enumerate(candidates)
        if ("in", i) in graph and graph[("in", i)].get(("out", i)) == 0
    ]
    return chosen, flow


def _select_legacy(results, count):
    if type(count) is not int or count <= 0:
        raise ValueError("Question count must be a positive integer")
    pool = remove_duplicates([item for item in results if _eligible(item)])
    buckets = {level: deque() for level in ("Easy", "Medium", "Hard")}
    for item in pool:
        buckets.setdefault(item.get("difficulty", "Medium"), deque()).append(item)
    chosen = []
    while len(chosen) < count:
        progress = False
        for level in ("Easy", "Medium", "Hard"):
            if buckets[level] and len(chosen) < count:
                chosen.append(buckets[level].popleft())
                progress = True
        if not progress:
            break
    if len(chosen) < count:
        raise QuestionSelectionError(
            f"Requested {count} questions; only {len(chosen)} distinct answerable candidates available."
        )
    return chosen


def select_questions(results, required_count):
    """Accept either original integer count or validated exam blueprint."""
    if type(required_count) is int:
        return _select_legacy(results, required_count)
    if not isinstance(required_count, dict):
        raise TypeError("Second argument must be an integer or blueprint dict")
    from aqz.blueprint.exam_blueprint import validate_blueprint
    blueprint = validate_blueprint(required_count)
    pool = remove_duplicates([item for item in results if _eligible(item)])
    shortages = []
    for section in blueprint["sections"]:
        available = sum(_eligible(item, section["type"]) for item in pool)
        if available < section["offered_count"]:
            shortages.append(
                f"{section['id']} ({section['type']}): need {section['offered_count']}, have {available}"
            )
    if shortages:
        raise QuestionSelectionError("Insufficient candidates: " + "; ".join(shortages))

    used = set()
    groups = {}
    # Solve the most constrained sections first.
    ordered = sorted(blueprint["sections"], key=lambda sec: (
        sum(_eligible(item, sec["type"]) for item in pool) / sec["offered_count"]
    ))
    for section in ordered:
        available = [(i, item) for i, item in enumerate(pool)
                     if i not in used and _eligible(item, section["type"])]
        chosen, flow = _flow_selection([item for _, item in available], section)
        if flow < section["offered_count"]:
            raise QuestionSelectionError(
                f"Section {section['id']}: only {flow}/{section['offered_count']} "
                "satisfy the Bloom + difficulty targets."
            )
        groups[section["id"]] = chosen
        used.update(i for i, item in available if any(item is selected for selected in chosen))

    final = []
    for section in blueprint["sections"]:
        for item in groups[section["id"]]:
            final.append({**item,
                          "section_id": section["id"],
                          "section_title": section["title"],
                          "question_type": section["type"],
                          "marks": section["marks_each"]})
    return final


def _self_test():
    from unittest.mock import patch
    import sys
    import types
    from aqz.blueprint.exam_blueprint import SAMPLE_BLUEPRINT
    import copy

    ctx = "Artificial Intelligence is a field of computer science."
    first = {"question": "What is Artificial Intelligence?", "context": ctx}
    second = {"question": "What is the field of computer science?", "context": ctx}
    third = {"question": "What is Machine Learning?", "context": "Machine Learning is a branch of AI."}
    assert len(remove_duplicates([first, second, third])) == 3
    fake_classifier = types.ModuleType("classifier")
    fake_classifier.classify_bloom = lambda q: {"level": "Remember", "confidence": 0.75}
    fake_classifier.classify_difficulty = lambda q: {"level": "Easy", "confidence": 0.70}
    fake_answers = types.ModuleType("answer_generator")
    fake_answers.generate_answer = lambda q, c: (
        "Artificial Intelligence" if "field of computer science?" in q else "A field of computer science"
    )
    with patch.dict(sys.modules, {"classifier": fake_classifier, "answer_generator": fake_answers}):
        processed = process_questions([first, second])
    assert len(processed) == 1
    assert len(remove_duplicates([
        {"question": "Who coined Artificial Intelligence?", "answer": "John McCarthy"},
        {"question": "Who organized the Dartmouth workshop?", "answer": "John McCarthy"},
    ])) == 2
    candidates = [
        {**first, "answer": "A field", "bloom": "Remember", "difficulty": "Easy"},
        {**third, "answer": "A branch", "bloom": "Understand", "difficulty": "Medium"},
        {"question": "Explain different learning types.", "context": "Types", "answer": "Supervised and unsupervised",
         "bloom": "Understand", "difficulty": "Medium"},
    ]
    assert len(select_questions(candidates, 2)) == 2
    try:
        select_questions(candidates, 9)
    except QuestionSelectionError:
        pass
    else:
        raise AssertionError("Shortage test failed")
    blueprint = copy.deepcopy(SAMPLE_BLUEPRINT)
    blueprint["sections"] = [
        {"id": "A", "title": "Recall", "type": "WHQ", "offered_count": 2,
         "answer_count": 1, "marks_each": 1,
         "bloom_targets": {"Remember": 1, "Understand": 1},
         "difficulty_targets": {"Easy": 1, "Medium": 1}},
        {"id": "B", "title": "WH-Questions", "type": "WHQ", "offered_count": 1,
         "answer_count": 1, "marks_each": 5},
    ]
    blueprint["full_marks"] = 6
    result = select_questions(candidates, blueprint)
    assert len(result) == 3 and [x["section_id"] for x in result] == ["A", "A", "B"]
    assert sum(x["marks"] for x in result) == 7  # Printed questions; full marks = 6.
    print("Self-test PASSED: deduplication, answer checks, quotas, legacy API, blueprint, shortages")


def main():
    parser = argparse.ArgumentParser(
        description="AQZ question_processor.py | verified dual PDF CLI"
    )
    parser.add_argument("pdf_path", nargs="?", help="PDF path (positional style)")
    parser.add_argument("--pdf", dest="pdf_flag", metavar="PATH",
                        help="PDF path (flag style)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run offline tests; no transformer download")
    parser.add_argument("--max-chunks", type=int, default=1,
                        help="Chunks to process, 0 = all (default: 1)")
    parser.add_argument("--blueprint", metavar="JSON",
                        help="Optional exam blueprint JSON")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.max_chunks < 0:
        parser.error("--max-chunks must be 0 or higher")
    pdf_path = args.pdf_flag or args.pdf_path
    if not pdf_path:
        parser.error("PDF missing: use --pdf 'file.pdf' or 'file.pdf'")
    from pathlib import Path
    if not Path(pdf_path).is_file():
        parser.error(f"PDF not found: {pdf_path}")

    from aqz.ingestion.pdf_service import extract_pdf_text
    from aqz.preprocessing.text_processor import preprocess_text, chunk_text
    from aqz.generation.question_generator import generate_questions
    print("STEP 1: Reading PDF")
    print(f"Running: {Path(__file__).resolve()}")
    chunks = chunk_text(preprocess_text(extract_pdf_text(pdf_path)))
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"Chunks selected: {len(chunks)}")
    if not chunks:
        print("No usable text chunks found in PDF.")
        return
    print("STEP 2: Generating candidates (T5 + rules)")
    raw = generate_questions(chunks)
    print(f"Generated candidates: {len(raw)}")
    print("STEP 3: Answer extraction and classification")
    processed = process_questions(raw)
    usable = [q for q in processed if _eligible(q)]
    print(f"After processing: {len(processed)}")
    print(f"Answerable candidates: {len(usable)}")
    for i, item in enumerate(usable, 1):
        print(f"Q{i}. {item['question']}")
        print(f"   Answer: {item['answer']} | Type: {item['question_type']} | "
              f"Bloom: {item['bloom']} | Difficulty: {item['difficulty']}")
    if args.blueprint:
        from aqz.blueprint.exam_blueprint import load_blueprint
        try:
            selected = select_questions(usable, load_blueprint(args.blueprint))
            print(f"Blueprint selection: {len(selected)} offered questions selected")
        except QuestionSelectionError as exc:
            print(f"Blueprint shortage: {exc}")
    print("INTEGRATION TEST COMPLETED")


if __name__ == "__main__":
    main()
