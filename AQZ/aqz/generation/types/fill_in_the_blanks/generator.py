"""Fill in the blanks generation backed by extractable source phrases."""

import re
from aqz.generation.types.common import _sentences

def _fill_blank_questions(chunk, max_count=3):

    """Create short fill-ups by blanking source phrases with extractable answers."""

    results = []

    seen = set()

    patterns = (

        r"\b(?:is|are|was|were)\s+(?:called|known as|defined as)\s+"

        r"(?P<answer>[^,.;]{2,48})",

        r"\b(?:consists of|consist of|is composed of|are composed of|"

        r"is made up of|are made up of)\s+(?P<answer>[^,.;]{2,48})",

        r"^(?:the\s+|a\s+|an\s+)?(?P<answer>[A-Z][A-Za-z0-9./-]*"

        r"(?:\s+[A-Za-z][A-Za-z0-9./-]*){0,3})\s+"

        r"(?:is|are|uses|use|has|have|provides|contains|consists|"

        r"connects|transmits|forwards|routes)\b",

    )

    for sentence in _sentences(chunk):

        if not sentence.endswith((".", "!", "?")) or not 6 <= len(sentence.split()) <= 32:

            continue

        answer = None

        for pattern in patterns:

            match = re.search(pattern, sentence)

            if match:

                candidate = match.group("answer").strip().rstrip(".?!")

                if (

                    1 <= len(candidate.split()) <= 5

                    and not re.match(

                        r"^(?:this|that|these|those|it|they|we|you|"

                        r"instructions?|examples?|chapter|section|figure|table)\b",

                        candidate,

                        flags=re.I,

                    )

                ):

                    answer = candidate

                    break

        if not answer or answer.casefold() in seen:

            continue

        blanked, count = re.subn(

            re.escape(answer), "________", sentence, count=1, flags=re.I

        )

        if not count:

            continue

        seen.add(answer.casefold())

        results.append({

            "question": f"Fill in the blank: {blanked}",

            "context": sentence,

            "question_type": "FILL_BLANK",

            "answer": answer,

            "answer_valid": True,

            "answer_kind": "source_phrase",

            "suggested_marks": 1,

        })

        if len(results) >= max_count:

            break

    return results


def generate(chunk, max_count=3):
    return _fill_blank_questions(chunk, max_count=max_count)
