"""Unified short and long WH-question generation."""

import re
from aqz.generation.types.common import (
    _clean_question, _evidence, _has, _outline, _remove_repeated_title,
    _sentences, _split_questions, _topic_from_sentences,
)

def _generate_descriptive_questions(chunk, topic_index=0):
    """Conservative educational templates backed by explicit source sentences.

    Topic-targeted patterns are used only when those exact concepts occur.
    Other PDFs receive limited, source-length-gated generic WH-question fallbacks.
    Extended questions are generated only when the source supports enough marking points.
    """
    sentences = [s for s in _sentences(chunk) if s.endswith(('.', '!', '?'))]
    if len(sentences) < 3:
        return []

    questions = []

    def add(prompt, kind, evidence, min_points):
        evidence = list(dict.fromkeys(evidence))
        if len(evidence) < min_points or not _clean_question(prompt):
            return
        context = " ".join(evidence)
        questions.append({
            "question": prompt,
            "context": context,
            "question_type": "WHQ",
            "_answer_level": kind,
            "answer": _outline(evidence, limit=12),
            "answer_valid": True,
            "answer_kind": "source_marking_points",
            "suggested_marks": 5 if kind == "SAQ" else 15,
        })

    # Compare teaching methods when BOTH are explained in the source.
    if _has(sentences, "traditional programming", "in machine learning"):
        ev = _evidence(sentences, ["traditional programming", "in machine learning"])
        add("How does traditional programming differ from machine learning?", "SAQ", ev, 2)

    if _has(sentences, "supervised learning", "unsupervised learning"):
        ev = _evidence(sentences, ["supervised learning", "unsupervised learning"])
        add("How do supervised and unsupervised learning differ, and what examples illustrate each?", "SAQ", ev, 2)

    if _has(sentences, "deep learning", "neural networks"):
        ev = _evidence(sentences, ["deep learning", "neural networks", "these networks"])
        if _has(sentences, "widely used", "computing resources"):
            add("What is Deep Learning, and what are one application and one limitation?", "SAQ", ev, 3)
        else:
            add("What is Deep Learning, and what role do multi-layer neural networks play?", "SAQ", ev, 2)

    if _has(sentences, "natural language processing", "typical tasks"):
        ev = _evidence(sentences, ["natural language processing", "typical tasks"])
        add("What is Natural Language Processing, and what are three typical tasks?", "SAQ", ev, 2)

    if _has(sentences, "a spam filter", "a navigation application", "speech recognition"):
        ev = _evidence(sentences, ["artificial intelligence is", "a spam filter", "a navigation application", "speech recognition"])
        add("What is Artificial Intelligence, and what are three everyday applications?", "SAQ", ev, 4)

    if _has(sentences, "healthcare", "education", "teachers"):
        ev = _evidence(sentences, ["in healthcare", "in education", "teachers remain", "qualified professionals"])
        add("How is AI used in healthcare and education, and what role does human review play?", "SAQ", ev, 3)

    if _has(sentences, "privacy risks", "biased or incomplete data", "organizations should"):
        ev = _evidence(sentences, ["biased or incomplete", "privacy risks", "organizations should"])
        add("What are two risks of AI, and how can organizations address them?", "SAQ", ev, 3)

    if _has(sentences, "dartmouth", "john mccarthy"):
        ev = _evidence(sentences, ["dartmouth", "john mccarthy", "research over subsequent decades"])
        add("What happened at the Dartmouth workshop, and why was it significant in AI history?", "SAQ", ev, 2)

    # Extended WH-questions require several points from ONE substantial chunk/section.
    # Only complete sentences are admissible as marking points.
    words = len(chunk.split())
    if words >= 105 and len(sentences) >= 6:
        if _has(sentences, "supervised learning", "unsupervised learning", "reinforcement learning", "deep learning"):
            ev = _evidence(sentences, ["machine learning", "traditional programming", "supervised learning", "unsupervised learning", "reinforcement learning", "deep learning", "these networks", "neural networks", "training data"])
            if _has(sentences, "widely used", "computing resources"):
                add("What are the main approaches to machine learning, and what are the principles, applications, and limitations of Deep Learning?", "LONG", ev, 6)
            else:
                add("What are the main approaches to machine learning, and how does Deep Learning work?", "LONG", ev, 6)

        if _has(sentences, "in healthcare", "in education", "in banking", "in agriculture"):
            ev = _evidence(sentences, ["in healthcare", "in education", "in banking", "in transportation", "in agriculture", "their reliability", "teachers remain", "qualified professionals"])
            add("How is AI applied in healthcare, education, banking, transportation, and agriculture?", "LONG", ev, 6)

        if _has(sentences, "privacy risks", "security", "human oversight", "responsible ai"):
            ev = _evidence(sentences, ["biased or incomplete", "privacy risks", "difficult to explain", "security", "attacked", "organizations should", "responsible ai"])
            add("What risks does AI create, and how can responsible practices manage them?", "LONG", ev, 6)

        if _has(sentences, "natural language processing", "typical tasks", "chatbot", "sentiment analysis"):
            ev = _evidence(sentences, ["natural language processing", "nlp systems", "typical tasks", "chatbot", "machine translation", "sentiment analysis", "these applications"])
            add("What are the main tasks and practical applications of Natural Language Processing?", "LONG", ev, 6)

    # Domain-agnostic fallback: a topic must be explicitly defined, and
    # substantial source evidence must exist. Never force an extended question just
    # to fill a quota when the uploaded material has too little detail.
    match = re.search(
        r"\b([A-Z][A-Za-z-]+(?:\s+[A-Z][A-Za-z-]+){0,3})"
        r"\s*(?:\([A-Z]{2,8}\))?\s+is\s+(?:a|an)\s+"
        r"(?:field|branch|area|method|process|technique|approach|system|framework)\b",
        " ".join(sentences),
    )
    if match:
        topic = _remove_repeated_title(match.group(1))
        if 1 <= len(topic.split()) <= 4:
            related = [s for s in sentences if topic.casefold() in s.casefold()]
            # Follow-up sentences can say 'these', 'they' or 'it' instead of
            # repeating the topic. Use the nearby source sentences when the
            # definition begins a coherent educational passage.
            if len(related) < 3 and len(sentences) >= 4:
                related = sentences[:4]
            if not any(q["_answer_level"] == "SAQ" for q in questions):
                add(f"What is {topic}, and what are two related concepts or examples?", "SAQ", related, 3)
            has_examples = bool(re.search(
                r"\b(?:such as|for example|examples? include|applications? include|including)\b",
                chunk, flags=re.I,
            ))
            if (not any(q["_answer_level"] == "LONG" for q in questions)
                    and words >= 125 and len(sentences) >= 7 and has_examples):
                add(f"What are the key characteristics of {topic}, and what examples does the source provide?",
                    "LONG", sentences[:12], 7)

    topic = _topic_from_sentences(sentences)
    if topic:
        if not any(q["_answer_level"] == "SAQ" for q in questions):
            saq_prompts = (
                f"What are the main components and functions of {topic}?",
                f"How does {topic} operate, and what are its main features?",
                f"What is the purpose of {topic}, and how is it used?",
            )
            add(
                saq_prompts[topic_index % len(saq_prompts)],
                "SAQ", sentences[:5], 3,
            )
        if (
            not any(q["_answer_level"] == "LONG" for q in questions)
            and words >= 100 and len(sentences) >= 6
        ):
            long_prompts = (
                f"What are the key concepts and processes of {topic}, and what examples illustrate them?",
                f"How is {topic} structured and operated, according to the supplied material?",
                f"What are the design, operation, and significance of {topic}?",
            )
            add(
                long_prompts[topic_index % len(long_prompts)],
                "LONG", sentences[:12], 6,
            )
    return questions


def generate(chunk, topic_index=0):
    # Factual recall questions are part of the same WH-question pool. Their
    # suggested mark count distinguishes them from descriptive questions.
    factual = [
        {
            "question": question,
            "context": sentence,
            "question_type": "WHQ",
            "suggested_marks": 1,
        }
        for question, sentence in _rule_questions(chunk)
    ]
    items = _generate_descriptive_questions(chunk, topic_index=topic_index)
    for item in items:
        item.pop("_answer_level", None)
    return factual + items




def _rule_questions(passage):

    """Preserved V3 factual templates; output pairs (question, source sentence)."""

    results = []

    for sentence in _sentences(passage):

        pattern = (r"\b(?P<term>(?:[A-Z][A-Za-z-]*\s+){0,5}[A-Z][A-Za-z-]*)"

                   r"\s*(?:\((?P<abbr>[A-Z]{2,8})\))?\s+is\s+(?:a|an)\s+"

                   r"(?:field|branch|area|type|method|technique|approach|process|"

                   r"subset|specialized area|specialized branch|part|system|model)\b")

        for match in re.finditer(pattern, sentence):

            term = _remove_repeated_title(match.group("term"))

            abbr = match.group("abbr")

            if abbr:

                words = term.split()

                for start in range(len(words)):

                    suffix = words[start:]

                    if "".join(w[0].upper() for w in suffix) == abbr:

                        term = " ".join(suffix)

                        break

            if len(term.split()) <= 4 and term.lower() not in {"this", "that", "it"}:

                results.append((f"What is {term}?", sentence))



        match = re.search(

            r"\b(?P<event>The\s+[A-Z][^.]{6,85}?)\s+"

            r"(?:took place|was held|occurred)\s+in\s+(?P<year>(?:18|19|20)\d{2})\b",

            sentence)

        if match:

            event = match.group("event").strip()

            results.append((f"When did {event[0].lower() + event[1:]} take place?", sentence))



        match = re.search(

            r"\b(?P<agent>(?:A|An|The)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,4})"

            r"(?:,\s*for example,)?\s+classifies\s+"

            r"(?P<object>[a-z][a-z\s-]{1,45}?)\s+as\s+[^.]+\.", sentence)

        if match:

            agent = match.group("agent")

            obj = match.group("object").strip()

            results.append((f"How does {agent[0].lower() + agent[1:]} classify {obj}?", sentence))



        match = re.search(

            r"\b(?P<agent>(?:A|An|The)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,4})"

            r"\s+converts\s+(?P<object>[^.]{2,60}?)\s+into\s+[^.]+\.", sentence)

        if match:

            agent = match.group("agent")

            obj = match.group("object").strip()

            results.append((f"What does {agent[0].lower() + agent[1:]} convert {obj} into?", sentence))



        match = re.search(

            r"\b(?P<agent>(?:A|An|The)\s+[a-z][a-z-]*(?:\s+[a-z][a-z-]*){0,4})"

            r"\s+can\s+(?P<verb>estimate|recommend|identify|detect|predict)\s+[^.]{3,65}\.", sentence)

        if match:

            agent, verb = match.group("agent"), match.group("verb")

            results.append((f"What can {agent[0].lower() + agent[1:]} {verb}?", sentence))

    return [(q, sentence) for q, sentence in results if _clean_question(q)]


def generate_with_model(passage, model):
    """Generate factual WH questions using the configured text2text model."""
    results = model(
        "generate questions: " + passage,
        max_new_tokens=160,
        num_beams=4,
        no_repeat_ngram_size=3,
        do_sample=False,
        truncation=True,
    )
    if not results:
        return []
    return [
        {
            "question": question,
            "context": passage,
            "question_type": "WHQ",
            "suggested_marks": 1,
        }
        for question in _split_questions(results[0].get("generated_text", ""))
    ]



