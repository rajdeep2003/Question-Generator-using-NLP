"""Shared text and question validation utilities."""

import re

QUESTION_STARTS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "should", "would", "will", "define", "describe", "explain", "discuss",
    "compare", "differentiate", "analyze", "analyse", "evaluate", "identify",
    "list", "state", "name", "mention", "illustrate", "justify", "critically",
}


def _sentences(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    boilerplate = (
        "instructions for testing",
        "question generation test material",
        "this is a text-based study document",
        "sample source material",
        "educational test material",
    )
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text)
        if len(sentence.split()) >= 5
        and not any(marker in sentence.casefold() for marker in boilerplate)
    ]


def _passages(chunk, max_words=70):
    result, group, size = [], [], 0
    for sentence in _sentences(chunk):
        count = len(sentence.split())
        if count > max_words:
            continue
        if group and size + count > max_words:
            if size >= 12:
                result.append(" ".join(group))
            group, size = [], 0
        group.append(sentence)
        size += count
    if group and size >= 12:
        result.append(" ".join(group))
    return result


def _clean_question(value):
    q = re.sub(r"\s+", " ", str(value)).strip()
    q = re.sub(r"^(?:question\s*\d*\s*[:.)-]|q\d+\s*[:.)-])\s*", "", q, flags=re.I)
    q = re.sub(r"^(?:<pad>|</s>|<s>)+\s*", "", q, flags=re.I)
    q = q.strip(" \t\n\r\"'`-:;,")
    if not q.endswith("?"):
        return None
    q = re.sub(r"\?+$", "?", q)
    words = q[:-1].split()
    if not 3 <= len(words) <= 30 or words[0].lower() not in QUESTION_STARTS:
        return None
    if re.search(r"[.!?]\s+[A-Z]", q[:-1]):
        return None
    if re.search(r"\b(?:as|like|of|about|for|by|from|to|with)\?$", q, re.I):
        return None
    if re.search(r"^what (?:does|do|did)\b.+\bnot (?:necessarily )?\w+\b", q, re.I):
        return None
    if re.search(r"\b(?:what|which) does\b.{0,110}\bwhether\b", q, re.I):
        return None
    if re.search(r"\b(?:do|does|did)\b.{0,70}\b(?:receives|uses|attempts|produces|helps)\b", q, re.I):
        return None
    if re.search(r"\b(?:do|does|did)\s+not\s+(?:necessarily\s+)?think\s+like\b", q, re.I):
        return None
    if re.search(r"^how\s+(?:can|could)\b.+\bbe\s+(?:designed|built|developed|created)\b", q, re.I):
        return None
    if re.fullmatch(r"what\s+can\s+.{2,110}\s+do\?", q, re.I):
        return None
    return q


def _split_questions(raw):
    raw = re.sub(r"</s>|<pad>", "", str(raw), flags=re.I)
    return [q for part in re.split(r"<sep>", raw, flags=re.I)
            if (q := _clean_question(part))]


def _remove_repeated_title(term):
    words = term.split()
    for size in range(1, len(words) // 2 + 1):
        if [w.lower() for w in words[-2*size:-size]] == [w.lower() for w in words[-size:]]:
            return " ".join(words[-size:])
    return term


def _key(question):
    return re.sub(r"[^a-z0-9]", "", str(question).casefold())


def _evidence(sentences, keywords):
    return [s for s in sentences if any(k in s.casefold() for k in keywords)]


def _outline(sentences, limit=12):
    """Answer-key marking points are verbatim sentences, never invented content."""
    chosen = list(dict.fromkeys(sentences))[:limit]
    return "\n".join(f"{i}. {s}" for i, s in enumerate(chosen, 1))


def _has(sentences, *phrases):
    joined = " ".join(sentences).casefold()
    return all(p in joined for p in phrases)


def _topic_from_sentences(sentences):
    verbs = (
        r"is|are|was|were|has|have|uses|use|consists|contain|contains|"
        r"provide|provides|mean|means|refer|refers|enable|enables|allow|"
        r"allows|support|supports|connect|connects|transmit|transmits|"
        r"forward|forwards|route|routes|store|stores|send|sends"
    )
    for sentence in sentences:
        match = re.match(
            rf"^(?P<subject>[^,.;!?]{{2,55}}?)\s+(?:{verbs})\b",
            sentence,
            flags=re.I,
        )
        if not match:
            continue
        topic = re.sub(
            r"^(?:the|a|an)\s+", "", match.group("subject").strip(),
            flags=re.I,
        )
        if ":" in topic:
            continue
        if 1 <= len(topic.split()) <= 5 and topic.casefold() not in {
            "it", "this", "they", "these", "there",
        } and not re.match(
            r"^(?:instructions?|objectives?|contents?|figures?|tables?|"
            r"examples?|chapter|section|page)\b",
            topic,
            flags=re.I,
        ):
            return topic
    return None


