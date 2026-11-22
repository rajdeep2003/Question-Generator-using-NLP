
"""AQZ answer extraction - Version 2 (original RoBERTa model).

Keeps generate_answer(question, context) -> str for the original app.
Adds generate_answer_details() for debugging and evaluation.
Extractive QA is intended for short factual answers, not essay writing.
"""

import argparse
import re
from functools import lru_cache
from unittest.mock import patch

from aqz.config import ANSWER_MODEL, DEVICE


MISSING_ANSWER = "Answer could not be confidently extracted."
MIN_CONFIDENCE = 0.12  # Tune against teacher-labelled examples, not a guarantee.


@lru_cache(maxsize=1)
def _get_answer_pipeline():
    """Load the same AQZ RoBERTa model once, only when needed."""
    from transformers import pipeline

    print(f"Loading AQZ answer model: {ANSWER_MODEL}")
    return pipeline(
        task="question-answering",
        model=ANSWER_MODEL,
        tokenizer=ANSWER_MODEL,
        device=DEVICE,
    )


def _normalize(text):
    """Normalize PDF whitespace without changing original content."""
    return re.sub(r"\s+", " ", str(text)).strip()


def _answer_in_context(answer, context):
    """Require the extracted answer to occur in the supplied source."""
    answer = _normalize(answer).casefold()
    context = _normalize(context).casefold()
    if not answer:
        return False
    return bool(re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", context))


def _is_descriptive(question):
    """RoBERTa-SQuAD2 cannot provide a supported full essay answer."""
    return bool(re.match(
        r"^(?:explain|discuss|describe|compare|contrast|differentiate|"
        r"evaluate|analyse|analyze|design|develop|justify|write a short note|"
        r"critically assess|critically evaluate)\b",
        _normalize(question), flags=re.I,
    ))


def _rejected(reason, score=0.0):
    return {
        "answer": MISSING_ANSWER,
        "confidence": round(float(score), 4),
        "valid": False,
        "reason": reason,
    }


def generate_answer_details(question, context, min_confidence=MIN_CONFIDENCE):
    """Return a source-checked answer and a diagnostic reason.

    Uses no second model. Rejects unsupported/unanswerable responses.
    This checks extraction grounding, not complete factual correctness.
    """
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (float, int)):
        raise ValueError("min_confidence must be between 0 and 1")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 and 1")

    q = _normalize(question)
    source = _normalize(context)
    if not q:
        return _rejected("empty_question")
    if not source:
        return _rejected("empty_context")
    if re.match(r"^(?:true\s*(?:/|or)\s*false)\s*[:?]", q, flags=re.I):
        return _rejected("true_false_requires_fact_verification")
    if _is_descriptive(q):
        return _rejected("descriptive_answer_not_supported_by_extract_model")

    # The pipeline handles long documents with token windows (doc_stride).
    result = _get_answer_pipeline()(
        question=q,
        context=source,
        handle_impossible_answer=True,
        max_seq_len=384,
        doc_stride=96,
        max_answer_len=48,
        top_k=1,
    )
    if isinstance(result, list):
        result = result[0] if result else {}
    if not isinstance(result, dict):
        return _rejected("unexpected_model_response")

    answer = _normalize(result.get("answer") or "")
    try:
        score = float(result.get("score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    if not answer:
        return _rejected("model_abstained", score)
    if answer.casefold() in {"unknown", "none", "n/a", "it", "they", "this", "that"}:
        return _rejected("vague_answer", score)
    if len(answer.split()) > 48:
        return _rejected("answer_too_long", score)
    if score < min_confidence:
        return _rejected("low_model_score", score)
    if not _answer_in_context(answer, source):
        return _rejected("answer_not_found_in_source", score)

    return {
        "answer": answer,
        "confidence": round(score, 4),
        "valid": True,
        "reason": "source_span_extracted",
    }


def generate_answer(question, context):
    """Backward-compatible API expected by original question_processor.py."""
    return generate_answer_details(question, context)["answer"]


def _self_test():
    """Offline tests with simulated RoBERTa responses (no download)."""
    source = (
        "Machine Learning uses labelled examples. "
        "The Dartmouth workshop took place in 1956."
    )

    def mock_result(answer, score):
        def fake_pipeline(**kwargs):
            assert kwargs["context"] == source
            assert kwargs["handle_impossible_answer"] is True
            return {"answer": answer, "score": score}
        return fake_pipeline

    with patch(__name__ + "._get_answer_pipeline", return_value=mock_result("1956", 0.88)):
        details = generate_answer_details("When did the Dartmouth workshop take place?", source)
        assert details["valid"] is True
        assert details["answer"] == "1956"
        assert generate_answer("When did the Dartmouth workshop take place?", source) == "1956"

    for response, score, reason in [
        ("", 0.90, "model_abstained"),
        ("1956", 0.01, "low_model_score"),
        ("2024", 0.90, "answer_not_found_in_source"),
        ("it", 0.90, "vague_answer"),
    ]:
        with patch(__name__ + "._get_answer_pipeline", return_value=mock_result(response, score)):
            assert generate_answer_details("When was the workshop held?", source)["reason"] == reason

    with patch(__name__ + "._get_answer_pipeline", side_effect=AssertionError("Model must not load")):
        assert generate_answer("Explain machine learning in detail.", source) == MISSING_ANSWER
        assert generate_answer("", source) == MISSING_ANSWER
        assert generate_answer("What is ML?", "") == MISSING_ANSWER
        assert generate_answer("True or False: ML uses examples.", source) == MISSING_ANSWER

    assert _answer_in_context("1956", source)
    assert not _answer_in_context("2024", source)
    print("Self-test PASSED: source checks, confidence, abstention, descriptive safety, compatibility")


def main():
    parser = argparse.ArgumentParser(description="AQZ original RoBERTa answer extractor")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--question", type=str)
    parser.add_argument("--context", type=str)
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    elif args.question is not None and args.context is not None:
        result = generate_answer_details(args.question, args.context)
        print(f"Answer: {result['answer']}")
        print(f"Valid: {result['valid']}")
        print(f"Model score: {result['confidence']}")
        print(f"Reason: {result['reason']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
