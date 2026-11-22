
"""AQZ Text Processor - Version 2.

Sentence-aware and section-aware PDF text processing.

Original interfaces preserved:
    preprocess_text(text)
    chunk_text(text)

No LLM or additional transformer is required.
"""

import argparse
import re

from aqz.config import CHUNK_WORDS, CHUNK_OVERLAP


# -----------------------------------------------
# 1. Identify numbered chapter headings
# -----------------------------------------------

SECTION_HEADER = re.compile(
    r"^\s*(?:"
    # Chapter 2: Title / Chapter 2. Title / Chapter 2 Title
    r"chapter\s+\d{1,2}(?:\.\d+)*(?:\s*[:.)-]\s*|\s+)"
    # 1. Title / 1) Title / 1.2 Title / 1.2.3 Title
    r"|\d{1,2}(?:\.\d+)*[.)]?\s+)"
    r"[A-Z][^.!?]{1,120}$",
    re.I,
)


def is_heading(line):
    # Markdown heading markers are common in pasted notes and PDF exports.
    line = re.sub(r"^\s*#{1,6}\s*", "", line.strip())
    return bool(SECTION_HEADER.fullmatch(line))


# -----------------------------------------------
# 2. PDF text preprocessing
# -----------------------------------------------

def preprocess_text(text):
    """
    Clean PDF extraction artifacts while
    preserving chapter and paragraph boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("PDF text must be a string")

    text = text.replace("\xa0", " ")
    text = text.replace("\u00ad", "")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\f", "\n\n")

    # Remove invisible control characters
    text = re.sub(
        r"[\x00-\x08\x0b\x0e-\x1f]",
        "",
        text,
    )

    # Fix hyphenated words broken across lines
    text = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "-",
        text,
    )

    paragraphs = []
    current = []
    seen_prose = False

    def looks_like_cover_title(line):
        words = re.findall(r"[A-Za-z]+", line)

        if not words or len(words) > 14:
            return False

        if line.endswith((".", "!", "?")):
            return False

        capitals = sum(
            word[0].isupper()
            for word in words
        )

        return capitals / len(words) >= 0.65

    def flush():
        if current:
            paragraphs.append(
                " ".join(current).strip()
            )
            current.clear()

    for raw_line in text.split("\n"):

        line = re.sub(
            r"\s+", " ", raw_line
        ).strip()

        if not line:
            flush()
            continue

        # Remove Markdown markers before preserving a recognized heading.
        line = re.sub(r"^\s*#{1,6}\s*", "", line)

        # Ignore standalone PDF page numbers
        if re.fullmatch(
            r"(?:page\s+)?\d{1,3}",
            line,
            flags=re.I,
        ):
            continue

        # Remove bullet symbols only
        line = re.sub(
            r"^[•●]\s*", "", line
        )

        if is_heading(line):
            flush()

            paragraphs.append(
                "\n" + line + "\n"
            )

        elif (
            not seen_prose
            and not current
            and looks_like_cover_title(line)
        ):
            # Avoid treating the cover title as a fact
            continue

        else:
            current.append(line)
            seen_prose = True

    flush()

    return "\n\n".join(paragraphs).strip()


# -----------------------------------------------
# 3. Sentence splitting
# -----------------------------------------------

def split_sentences(paragraph):
    """
    Split at sentence boundaries, rather than
    arbitrarily cutting the text by word count.
    """

    paragraph = re.sub(
        r"\s+", " ", paragraph
    ).strip()

    if not paragraph:
        return []

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[\"'(]?[A-Z0-9])",
            paragraph,
        )
        if sentence.strip()
    ]


# -----------------------------------------------
# 4. Build sentence-aligned chunks
# -----------------------------------------------

def build_chunks(
    sentences,
    max_words,
    overlap,
    min_words,
):

    chunks = []
    current = []
    count = 0

    def flush_chunk():
        nonlocal current, count

        if count >= min_words:
            chunks.append(
                " ".join(current)
            )

        current = []
        count = 0

    for sentence in sentences:

        words = sentence.split()

        if not words:
            continue

        word_count = len(words)

        # Preserve an unusually long sentence
        # rather than silently truncating it.
        if word_count > max_words:
            flush_chunk()
            chunks.append(sentence)
            continue

        if (
            current
            and count + word_count > max_words
        ):

            previous = current[:]
            flush_chunk()

            # Include only complete previous sentences
            # in the overlap.
            if overlap > 0:

                tail = []
                tail_words = 0

                for old in reversed(previous):

                    old_count = len(old.split())

                    if tail_words + old_count > overlap:
                        break

                    tail.insert(0, old)
                    tail_words += old_count

                if (
                    tail_words + word_count
                    <= max_words
                ):
                    current = tail
                    count = tail_words

        current.append(sentence)
        count += word_count

    flush_chunk()

    return chunks


# -----------------------------------------------
# 5. Main chunking function
# -----------------------------------------------

def chunk_text(
    text,
    max_words=None,
    overlap=None,
    min_words=1,
):
    """
    Split cleaned educational material
    into complete-sentence chunks.

    The original call still works:
        chunks = chunk_text(text)
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    if max_words is None:
        max_words = CHUNK_WORDS

    if overlap is None:
        overlap = CHUNK_OVERLAP

    if (
        type(max_words) is not int
        or max_words <= 0
        or type(overlap) is not int
        or overlap < 0
        or overlap >= max_words
        or type(min_words) is not int
        or min_words <= 0
    ):
        raise ValueError(
            "Invalid max_words, overlap or min_words"
        )

    if not text.strip():
        return []

    sections = []
    active = []
    heading_seen = False

    for paragraph in re.split(
        r"\n\s*\n", text
    ):

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        # Start a new group when a heading appears
        if is_heading(paragraph):

            if active:

                preface_text = " ".join(active)

                # Skip short document-description text
                # before the first numbered chapter.
                preface = (
                    not heading_seen
                    and len(preface_text.split()) < 60
                    and bool(re.search(
                        r"\b(?:this document|this material|"
                        r"sample source material|"
                        r"educational test material)\b",
                        preface_text,
                        re.I,
                    ))
                )

                if not preface:
                    sections.append(active)

                active = []

            heading_seen = True

        else:
            active.extend(
                split_sentences(paragraph)
            )

    if active:
        sections.append(active)

    chunks = []

    for sentences in sections:

        section_chunks = build_chunks(
            sentences,
            max_words,
            overlap,
            min_words,
        )

        chunks.extend(section_chunks)

    return chunks


# -----------------------------------------------
# 6. Self-test
# -----------------------------------------------

def self_test():

    sample = (
        "1. Machine Learning and Deep Learning\n"
        "Machine Learning uses labelled examples "
        "to learn useful patterns.\n"
        "Unsupervised learning discovers structure "
        "in data without predefined labels.\n"
        "2. Natural Language Processing\n"
        "Natural Language Processing helps "
        "computers process human language.\n"
        "Chatbots interpret user requests "
        "and provide responses to users."
    )

    cleaned = preprocess_text(sample)

    chunks = chunk_text(
        cleaned,
        max_words=25,
        overlap=5,
        min_words=5,
    )

    assert chunks, "No chunks produced"

    # Headings must not become model sentences
    assert not any(
        "1. Machine Learning" in chunk
        or "2. Natural Language" in chunk
        for chunk in chunks
    )

    # Separate sections must not merge
    assert not any(
        "Unsupervised learning" in chunk
        and "Chatbots interpret" in chunk
        for chunk in chunks
    )

    # Important source information remains
    combined = " ".join(chunks)

    assert "labelled examples" in combined
    assert "Chatbots interpret" in combined

    # Empty text
    assert chunk_text("") == []

    # Broken PDF word
    assert (
        preprocess_text("A multi-\nlayer system.")
        == "A multi-layer system."
    )

    # Invalid overlap
    try:
        chunk_text(
            cleaned,
            max_words=20,
            overlap=20,
        )
        raise AssertionError(
            "Invalid overlap was accepted"
        )

    except ValueError:
        pass

    print(
        "Self-test PASSED: PDF cleaning, "
        "section boundaries, complete sentences, "
        "overlap"
    )


# -----------------------------------------------
# 7. Command-line tests
# -----------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="AQZ text processor tester"
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--pdf",
        type=str,
        help="Path to educational PDF",
    )

    parser.add_argument(
        "--max-chunks",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()

    elif args.pdf:

        from aqz.ingestion.pdf_service import extract_pdf_text

        text = extract_pdf_text(args.pdf)

        cleaned = preprocess_text(text)

        chunks = chunk_text(cleaned)

        print(
            f"Total chunks: {len(chunks)}"
        )

        for index, chunk in enumerate(
            chunks[:max(0, args.max_chunks)],
            start=1,
        ):

            print(
                f"\n--- Chunk {index} "
                f"({len(chunk.split())} words) ---"
            )

            print(chunk)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
