"""Source-grounded multiple-choice question generation."""

import re

from aqz.generation.types.common import _sentences


_DEFINITION = re.compile(
    r"^(?P<subject>[A-Z][A-Za-z0-9 &'/-]{1,55}?)\s+"
    r"(?:is|are)\s+(?P<definition>(?:a|an|the)\s+[^.!?]{8,150})[.!?]$"
)


def _definitions(chunk):
    """Extract concise subject/definition pairs stated directly in the source."""
    found = []
    seen = set()
    for sentence in _sentences(chunk):
        match = _DEFINITION.match(sentence.strip())
        if not match:
            continue
        subject = match.group("subject").strip()
        definition = match.group("definition").strip().rstrip(".!?")
        key = (subject.casefold(), definition.casefold())
        if key in seen:
            continue
        seen.add(key)
        found.append((subject, definition, sentence))
    return found


def generate(chunk, max_count=10):
    """Build four-option definition questions using only source statements.

    A chunk needs four distinct source definitions to provide one correct
    option and three source-grounded distractors.
    """
    definitions = _definitions(chunk)
    if len(definitions) < 4:
        return []

    questions = []
    for index, (subject, correct, context) in enumerate(definitions):
        distractors = []
        for offset in range(1, len(definitions)):
            candidate = definitions[(index + offset) % len(definitions)][1]
            if candidate.casefold() != correct.casefold() and candidate not in distractors:
                distractors.append(candidate)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            continue

        # Rotate option order so the correct answer is not always first.
        correct_position = index % 4
        options = distractors[:]
        options.insert(correct_position, correct)
        questions.append({
            "question": f"Which option best describes {subject}?",
            "context": context,
            "question_type": "MCQ",
            "options": options,
            "answer": correct,
            "answer_index": correct_position,
            "answer_valid": True,
            "answer_kind": "source_definition",
            "explanation": context,
            "suggested_marks": 1,
        })
        if len(questions) >= max_count:
            break
    return questions
