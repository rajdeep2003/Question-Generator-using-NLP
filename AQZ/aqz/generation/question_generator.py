"""AQZ Question Generator V4: unified WHQ, fill-blank and true/false.

Drop-in replacement for question_generator.py. Original public API is unchanged:
    generate_questions(chunks)
Optional offline mode: generate_questions(chunks, use_t5=False)

No extra dependencies, no chat LLM, no external API. Source-based WHQ
answers are marking-point excerpts, NOT generated essays; T/F assertions are
literal source statements or explicit copula negations.
"""

import argparse
import re
from collections import Counter
from functools import lru_cache

from aqz.config import DEVICE, QUESTION_MODEL
from aqz.generation.types.common import (
    _clean_question, _key, _passages, _sentences,
)

@lru_cache(maxsize=1)
def _get_question_pipeline():
    from transformers import pipeline
    print(f"Loading AQZ question model: {QUESTION_MODEL}")
    return pipeline("text2text-generation", model=QUESTION_MODEL, device=DEVICE)






























def generate_questions(chunks, use_t5=True, max_t5_passages=24, question_types=None):
    """Return candidates for supported question types, optionally restricted.

    Existing callers may continue using generate_questions(chunks).
    use_t5=False is a FAST, offline rules-only test mode.
    """
    if not chunks:
        return []

    allowed_types = set(question_types) if question_types is not None else {
        "WHQ", "FILL_BLANK", "TRUE_FALSE", "MCQ"
    }

    # Keep generation orchestration here while each question format exposes
    # its own implementation entry point under generation/types/.
    from aqz.generation.types.wh_questions.generator import generate as generate_whq
    from aqz.generation.types.wh_questions.generator import generate_with_model as generate_whq_with_model
    from aqz.generation.types.fill_in_the_blanks.generator import generate as generate_fill_blank
    from aqz.generation.types.true_false.generator import generate as generate_true_false
    from aqz.generation.types.mcq.generator import generate as generate_mcq

    qg_model = _get_question_pipeline() if use_t5 else None
    t5_passages_used = 0
    generated, seen = [], set()
    tf_index = 0

    def add(item):
        question = item.get("question", "")
        kind = item.get("question_type")
        if kind == "FILL_BLANK":
            if "________" not in question or not item.get("answer"):
                return
        elif kind != "TRUE_FALSE":
            cleaned = _clean_question(question)
            if not cleaned:
                return
            item = {**item, "question": cleaned}
        elif not question.startswith("True or False: "):
            return
        key = _key(item["question"])
        if key in seen:
            return
        seen.add(key)
        generated.append(item)

    for chunk_index, chunk in enumerate(chunks, 1):
        passages = _passages(chunk)
        for passage_index, passage in enumerate(passages, 1):
            print(f"Generating: chunk {chunk_index}, passage {passage_index}")
            if "WHQ" in allowed_types:
                for item in generate_whq(passage, topic_index=chunk_index - 1):
                    if item.get("suggested_marks", 1) <= 1:
                        add(item)
            if "WHQ" in allowed_types and qg_model is not None and t5_passages_used < max_t5_passages:
                model_items = generate_whq_with_model(passage, qg_model)
                t5_passages_used += 1
                for item in model_items:
                    add({**item, "question_type": "WHQ", "suggested_marks": 1})

        # Descriptive and T/F rules use the entire chunk for sufficient evidence.
        if "WHQ" in allowed_types:
            for item in generate_whq(chunk, topic_index=chunk_index - 1):
                add(item)
        if "FILL_BLANK" in allowed_types:
            for item in generate_fill_blank(chunk):
                add(item)
        if "TRUE_FALSE" in allowed_types:
            tf_items = generate_true_false(chunk, start_index=tf_index)
            for item in tf_items:
                add(item)
            tf_index += len(tf_items)

    # MCQ distractors are selected across the full source, so collect them
    # once after processing all chunks.
    if "MCQ" in allowed_types:
        for item in generate_mcq(
            " ".join(str(chunk) for chunk in chunks), use_model=use_t5
        ):
            add(item)

    return generated


def _self_test():
    from aqz.generation.types.wh_questions.generator import _rule_questions
    from aqz.generation.types.true_false.generator import _true_false_questions
    # Keep original factual regression checks.
    assert _clean_question("What does a spam filter classify as?") is None
    assert _clean_question("What can a navigation application do?") is None
    assert _clean_question("What is Artificial Intelligence?")
    assert _clean_question("How can AI systems be designed for narrow tasks or broader problem-solving?") is None

    sample = (
        "Artificial Intelligence is a field of computer science concerned with intelligent systems. "
        "A spam filter, for example, classifies messages as unwanted or legitimate. "
        "A navigation application can estimate travel times and recommend routes. "
        "A speech recognition system converts spoken words into text. "
        "AI systems can assist with other tasks in education and healthcare."
    )
    questions = [q for q, _ in _rule_questions(sample)]
    assert "What is Artificial Intelligence?" in questions, questions
    assert "How does a spam filter classify messages?" in questions, questions

    tf = _true_false_questions(sample)
    assert tf and tf[0]["answer"] == "False", tf
    assert " is not " in tf[0]["question"], tf
    assert tf[0]["explanation"] in sample

    ml = (
        "Machine Learning is a branch of AI in which algorithms learn patterns from data. "
        "In traditional programming, a developer writes explicit rules for a task. "
        "In machine learning, a model is trained using examples to learn relationships in data. "
        "Supervised learning uses labelled examples, such as emails marked as spam or not spam. "
        "Unsupervised learning searches for structure in data without predefined labels, such as grouping customers. "
        "Reinforcement learning trains an agent through interaction, feedback, and rewards. "
        "Deep Learning is a specialized area of machine learning using neural networks with multiple layers. "
        "These networks can learn complex representations from large datasets. "
        "Deep learning models may require substantial computing resources and large amounts of data."
    )
    results = generate_questions([ml], use_t5=False)
    counts = Counter(q.get("question_type") for q in results)
    assert counts["WHQ"] >= 2, counts
    assert counts["WHQ"] >= 3, counts
    assert counts["TRUE_FALSE"] >= 1, counts
    assert counts["FILL_BLANK"] >= 1, counts
    assert any(
        q["question"].startswith("Fill in the blank:")
        for q in results if q["question_type"] == "FILL_BLANK"
    )
    for q in results:
        if q.get("answer_kind") == "source_marking_points":
            assert q["answer_valid"] and "1. " in q["answer"]

    networks = (
        "Computer networks connect computers and devices to share data and resources. "
        "Routers forward packets between different networks using routing tables. "
        "An Internet Protocol address identifies a device interface on a network. "
        "The Domain Name System translates domain names into IP addresses. "
        "Ethernet is a common technology for wired local area networks. "
        "Wireless networks use radio waves to communicate between devices. "
        "Network security protects data and services from unauthorized access. "
        "Firewalls examine network traffic according to configured security rules. "
        "Network administrators monitor performance and troubleshoot connection problems. "
        "Redundant links can improve the availability of important network services. "
        "Protocols define rules that devices follow when exchanging information."
    )
    network_results = generate_questions([networks], use_t5=False)
    network_counts = Counter(q.get("question_type") for q in network_results)
    assert network_counts["WHQ"] >= 1, network_counts
    assert network_counts["FILL_BLANK"] >= 1, network_counts
    print("Self-test PASSED: unified WHQ, fill-ups, TRUE_FALSE; source-grounded answers; offline")


def main():
    from collections import Counter
    parser = argparse.ArgumentParser(description="AQZ Question Generator V4 (all four question types)")
    parser.add_argument("pdf", nargs="?", help="PDF path (positional form)")
    parser.add_argument("--pdf", dest="pdf_flag", metavar="PATH", help="PDF path (flag form)")
    parser.add_argument("--max-chunks", type=int, default=1, help="0 = all chunks")
    parser.add_argument("--rules-only", action="store_true", help="No T5 model; quick template test")
    parser.add_argument("--self-test", action="store_true", help="Offline regression tests")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return
    if args.max_chunks < 0:
        parser.error("--max-chunks must be 0 or higher")
    pdf_path = args.pdf_flag or args.pdf
    if not pdf_path:
        parser.error("PDF required: --pdf PATH (or positional path)")
    from pathlib import Path
    if not Path(pdf_path).is_file():
        parser.error(f"PDF not found: {pdf_path}")
    from aqz.ingestion.pdf_service import extract_pdf_text
    from aqz.preprocessing.text_processor import preprocess_text, chunk_text
    chunks = chunk_text(preprocess_text(extract_pdf_text(pdf_path)))
    if args.max_chunks > 0:
        chunks = chunks[:args.max_chunks]
    results = generate_questions(chunks, use_t5=not args.rules_only)
    counts = Counter(q.get("question_type", "WHQ") for q in results)
    print("\nCandidate counts by type:", dict(counts))
    for i, item in enumerate(results, 1):
        print(f"\n{i}. [{item.get('question_type', 'WHQ')}] {item['question']}")
        if item.get("answer") is not None:
            print("Answer:", item["answer"])


if __name__ == "__main__":
    main()


