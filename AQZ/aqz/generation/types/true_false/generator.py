"""True/false generation backed by verified source assertions."""

import re
from aqz.generation.types.common import _sentences

def _true_false_questions(chunk, start_index=0, max_count=2):

    """Use source facts; make false statements only by negating 'is/are'."""

    candidates = []

    for sentence in _sentences(chunk):

        # Restrict to stand-alone unhedged declarative assertions.

        if re.match(r"^\d+[.)]\s", sentence):

            continue  # A numbered section heading is not part of the assertion.

        if not (8 <= len(sentence.split()) <= 28):

            continue

        if any(re.search(r"\b" + word + r"\b", sentence, re.I)

               for word in ("not", "never", "may", "might", "could", "sometimes", "often", "however")) or "rather than" in sentence.casefold():

            continue

        match = re.fullmatch(

            r"(?P<subj>[A-Z][^.!?]{3,65}?)\s+"

            r"(?P<verb>is|are)\s+"

            r"(?P<pred>[^.!?]{15,135})\.", sentence,

        )

        if not match:

            continue

        subj = match.group("subj").strip()

        if subj.lower().startswith(("in ", "for ", "because ", "although ", "such ", "these ", "this ", "those ", "it ", "they ")):

            continue

        # PDF heading accidentally concatenated with a repeated subject.

        if len(re.findall(r"\b[A-Z][a-z]+\b", subj)) > 6:

            continue

        subj_words = re.findall(r"[a-z]+", subj.casefold())

        if any(tuple(subj_words[i:i+n]) == tuple(subj_words[j:j+n])

               for n in (2, 3)

               for i in range(len(subj_words) - n + 1)

               for j in range(i + n, len(subj_words) - n + 1)):

            continue

        if re.search(r"\b(?:and|or)\s+(?:is|are)\b", match.group("pred"), re.I):

            continue

        candidates.append((sentence, match))



    results = []

    for offset, (sentence, match) in enumerate(candidates[:max_count]):

        # False / True alternate globally across chunks.

        make_false = (start_index + offset) % 2 == 0

        claim = sentence

        if make_false:

            claim = (

                match.group("subj") + " " + match.group("verb") + " not "

                + match.group("pred") + "."

            )

        results.append({

            "question": f"True or False: {claim}",

            "context": sentence,

            "question_type": "TRUE_FALSE",

            "answer": "False" if make_false else "True",

            "answer_valid": True,

            "answer_kind": "source_verified_true_false",

            "explanation": sentence,

            "suggested_marks": 1,

        })

    return results


def generate(chunk, start_index=0, max_count=2):
    return _true_false_questions(chunk, start_index=start_index, max_count=max_count)
