
"""AQZ Classifier - Version 2.

Hybrid Bloom classification:
1. Rule-based cognitive-action detection
2. Original BART transformer fallback

Keeps original function interfaces.
"""

import argparse
import re
from functools import lru_cache

from aqz.config import CLASSIFIER_MODEL, DEVICE


# Bloom's taxonomy descriptions for BART
BLOOM_DESCRIPTIONS = {
    "Remember": "recall a learned fact, term, date or definition",
    "Understand": "explain an idea or describe how something works",
    "Apply": "use a method or procedure to solve a concrete problem",
    "Analyze": "compare, contrast, or examine relationships and components",
    "Evaluate": "judge a solution using evidence or criteria",
    "Create": "design or develop a new solution or plan",
}


# ------------------------------------------------
# 1. Rule-based Bloom's taxonomy classification
# ------------------------------------------------

ACTION_RULES = [
    (
        "Create",
        r"(?:design|develop|construct|formulate|propose|devise|create|build)\b"
    ),
    (
        "Create",
        r"write\s+(?:an?\s+)?(?:algorithm|program|pseudocode)\b"
    ),
    (
        "Evaluate",
        r"(?:critically\s+evaluate|evaluate|justify|critique|assess|defend|recommend|argue)\b"
    ),
    (
        "Analyze",
        r"(?:compare|differentiate|distinguish|analy[sz]e|examine|contrast|categorize|classify)\b"
    ),
    (
        "Analyze",
        r"what\s+(?:is|are)\s+(?:the\s+)?(?:differences?|distinctions?)\b"
    ),
    (
        "Apply",
        r"(?:calculate|compute|solve|determine|demonstrate|apply|perform|trace|execute)\b"
    ),
    (
        "Apply",
        r"(?:find|use)\b.+\b(?:value|result|output|profit|complexity|answer|using|algorithm)\b"
    ),
    (
        "Apply",
        r"(?:given|for\s+the\s+given)\b.+\b(?:find|calculate|compute|solve|determine)\b"
    ),
    (
        "Understand",
        r"(?:explain|describe|discuss|summarize|interpret|outline)\b"
    ),
    (
        "Understand",
        r"(?:how\s+(?:do|does|did|can)|why\b)"
    ),
    (
        "Remember",
        r"(?:what\s+(?:is|are|was|were|does|do|can)|which\b|who\b|whom\b|when\b|where\b|define\b|state\b|list\b|name\b|identify\b|mention\b|how\s+many\b|true\s+or\s+false\b|fill\s+in\s+the\s+blank\b)"
    ),
]


def _clean(text):
    text = re.sub(
        r"\s+", " ", str(text)
    ).strip()

    # Remove question numbering
    text = re.sub(
        r"^(?:Q\s*\d+\s*[.):]\s*|\d+\s*[.):]\s*)",
        "",
        text,
        flags=re.I
    )

    return text


def _rule_level(question):
    """Identify Bloom's level from cognitive action verbs."""

    q = _clean(question).lower()

    for label, pattern in ACTION_RULES:

        if re.match(
            r"^(?:" + pattern + r")",
            q
        ):
            return label

    return None


# ------------------------------------------------
# 2. Load original BART classifier when needed
# ------------------------------------------------

@lru_cache(maxsize=1)
def _get_classifier():

    from transformers import pipeline

    print(
        f"Loading original AQZ classifier: "
        f"{CLASSIFIER_MODEL}"
    )

    return pipeline(
        "zero-shot-classification",
        model=CLASSIFIER_MODEL,
        device=DEVICE
    )


# ------------------------------------------------
# 3. BART classification fallback
# ------------------------------------------------

@lru_cache(maxsize=512)
def _bart_bloom(question):

    descriptions = list(
        BLOOM_DESCRIPTIONS.values()
    )

    result = _get_classifier()(
        question,
        candidate_labels=descriptions,
        hypothesis_template=(
            "To answer this question, "
            "a student must {}."
        ),
        multi_label=False
    )

    description_to_label = {
        description: level
        for level, description
        in BLOOM_DESCRIPTIONS.items()
    }

    best_description = result["labels"][0]

    level = description_to_label[
        best_description
    ]

    score = float(
        result["scores"][0]
    )

    return level, score


# ------------------------------------------------
# 4. Main Bloom classification function
# ------------------------------------------------

def classify_bloom(question):
    """
    Return Bloom's taxonomy level.

    Original output format preserved:
    {
        "level": "Remember",
        "confidence": 0.75
    }
    """

    if (
        not isinstance(question, str)
        or not question.strip()
    ):
        raise ValueError(
            "Question must be non-empty text."
        )

    # First try deterministic NLP rules
    level = _rule_level(question)

    if level:
        return {
            "level": level,
            "confidence": 0.75
        }

    # If no rule matches, use BART
    level, score = _bart_bloom(
        _clean(question)
    )

    return {
        "level": level,
        "confidence": round(score, 4)
    }


# ------------------------------------------------
# 5. Difficulty classification
# ------------------------------------------------

def classify_difficulty(question):
    """
    Estimate question difficulty using Bloom's
    cognitive level and complexity indicators.
    """

    if (
        not isinstance(question, str)
        or not question.strip()
    ):
        raise ValueError(
            "Question must be non-empty text."
        )

    q = _clean(question).lower()

    level = _rule_level(question)

    # High complexity
    if re.search(
        r"\b(?:prove|derive|optimi[sz]e|critically|worst.case|best.case)\b",
        q
    ):
        difficulty = "Hard"

    elif level in {
        "Create",
        "Evaluate"
    }:
        difficulty = "Hard"

    # Medium complexity
    elif level in {
        "Apply",
        "Analyze",
        "Understand"
    }:
        difficulty = "Medium"

    # Recall questions
    elif level == "Remember":
        difficulty = "Easy"

    else:
        difficulty = "Medium"

    return {
        "level": difficulty,
        "confidence": 0.70
    }


# ------------------------------------------------
# 6. Standalone tests
# ------------------------------------------------

def _self_test():

    checks = {
        "What is Artificial Intelligence?":
            ("Remember", "Easy"),

        "Fill in the blank: TCP uses ________ to transmit data.":
            ("Remember", "Easy"),

        "How does a spam filter classify messages?":
            ("Understand", "Medium"),

        "Explain supervised learning.":
            ("Understand", "Medium"),

        "Calculate the shortest path using Dijkstra's algorithm.":
            ("Apply", "Medium"),

        "Compare supervised and unsupervised learning.":
            ("Analyze", "Medium"),

        "Evaluate the ethical risks of AI.":
            ("Evaluate", "Hard"),

        "Design an AI-based learning system.":
            ("Create", "Hard"),

        "What is the difference between ML and DL?":
            ("Analyze", "Medium"),

        "Given three items, find the maximum profit.":
            ("Apply", "Medium"),
    }

    for question, expected in checks.items():

        bloom = classify_bloom(question)

        difficulty = classify_difficulty(
            question
        )

        result = (
            bloom["level"],
            difficulty["level"]
        )

        assert result == expected, (
            question,
            result,
            expected
        )

    print(
        f"Self-test PASSED: {len(checks)} "
        "classification examples "
        "(no model loaded)"
    )


# ------------------------------------------------
# 7. Test through terminal
# ------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test AQZ Bloom classification"
    )

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--question",
        type=str,
        default=None
    )

    args = parser.parse_args()

    if args.self_test:
        _self_test()

    elif args.question:

        print(
            "Bloom:",
            classify_bloom(args.question)
        )

        print(
            "Difficulty:",
            classify_difficulty(args.question)
        )

    else:
        parser.print_help()
