
"""AQZ Examination Blueprint.

Validates examination metadata, marking schemes, question types,
Bloom's taxonomy targets, and difficulty targets.

No transformer models are loaded in this module.
"""

import argparse
import copy
import json
import re
from datetime import date
from pathlib import Path


QUESTION_TYPES = {"FILL_BLANK", "TRUE_FALSE", "WHQ"}

BLOOM_LEVELS = {
    "Remember", "Understand", "Apply",
    "Analyze", "Evaluate", "Create"
}

DIFFICULTY_LEVELS = {"Easy", "Medium", "Hard"}


class BlueprintError(ValueError):
    """Raised for an invalid examination configuration."""
    pass


# ----------------------------------------------------
# 1. Common validation functions
# ----------------------------------------------------

def _required_text(data, key):
    value = data.get(key)

    if not isinstance(value, str) or not value.strip():
        raise BlueprintError(
            f"'{key}' must be a non-empty string."
        )

    return value.strip()


def _positive_int(value, field):
    if type(value) is not int or value <= 0:
        raise BlueprintError(
            f"'{field}' must be a positive integer."
        )

    return value


# ----------------------------------------------------
# 2. Bloom's taxonomy and difficulty targets
# ----------------------------------------------------

def _validate_targets(targets, field, count, allowed):

    if targets is None:
        return {}

    if not isinstance(targets, dict):
        raise BlueprintError(
            f"'{field}' must be a dictionary."
        )

    if not targets:
        return {}

    unknown = set(targets) - allowed

    if unknown:
        raise BlueprintError(
            f"Unknown {field} labels: {sorted(unknown)}"
        )

    for label, amount in targets.items():
        _positive_int(amount, f"{field}.{label}")

    if sum(targets.values()) != count:
        raise BlueprintError(
            f"'{field}' must total {count} offered "
            f"questions; got {sum(targets.values())}."
        )

    return dict(targets)


# ----------------------------------------------------
# 3. Validate full examination blueprint
# ----------------------------------------------------

def validate_blueprint(data):
    """
    Validates the teacher's exam configuration.

    offered_count = questions printed in the paper
    answer_count  = questions students must attempt

    Full marks = sum(answer_count * marks_each)

    Bloom and difficulty targets apply to OFFERED
    questions in each section.
    """

    if not isinstance(data, dict):
        raise BlueprintError(
            "Blueprint must be a dictionary."
        )

    paper = copy.deepcopy(data)

    # Dynamic examination information
    required_fields = (
        "college_name",
        "college_address",
        "course_name",
        "semester",
        "academic_year",
        "subject_name",
        "paper_code",
        "exam_date"
    )

    for field in required_fields:
        paper[field] = _required_text(paper, field)

    optional_fields = (
        "university_name",
        "exam_name",
        "branch",
        "teacher_prompt",
        "logo_path"
    )

    for field in optional_fields:
        value = paper.get(field, "")

        if not isinstance(value, str):
            raise BlueprintError(
                f"'{field}' must be a string."
            )

        paper[field] = value.strip()

    # Validate examination date
    if not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}",
        paper["exam_date"]
    ):
        raise BlueprintError(
            "'exam_date' must be YYYY-MM-DD."
        )

    try:
        date.fromisoformat(paper["exam_date"])
    except ValueError as error:
        raise BlueprintError(
            "'exam_date' must be a valid date."
        ) from error

    # Validate duration
    paper["duration_minutes"] = _positive_int(
        paper.get("duration_minutes"),
        "duration_minutes"
    )

    # Validate examination instructions
    instructions = paper.get("instructions", [])

    if not isinstance(instructions, list):
        raise BlueprintError(
            "'instructions' must be a list."
        )

    if any(
        not isinstance(item, str) or not item.strip()
        for item in instructions
    ):
        raise BlueprintError(
            "Instructions must contain non-empty strings."
        )

    paper["instructions"] = [
        item.strip() for item in instructions
    ]

    # Validate sections
    sections = paper.get("sections")

    if not isinstance(sections, list) or not sections:
        raise BlueprintError(
            "'sections' must be a non-empty list."
        )

    seen_ids = set()

    full_marks = 0
    total_offered = 0
    total_to_answer = 0

    for index, section in enumerate(sections, start=1):

        if not isinstance(section, dict):
            raise BlueprintError(
                f"Section {index} must be a dictionary."
            )

        section_id = _required_text(section, "id")

        if section_id.casefold() in seen_ids:
            raise BlueprintError(
                f"Duplicate section id: '{section_id}'."
            )

        seen_ids.add(section_id.casefold())

        section["title"] = _required_text(
            section, "title"
        )

        q_type = _required_text(
            section, "type"
        ).upper()

        if q_type not in QUESTION_TYPES:
            raise BlueprintError(
                f"Invalid question type: {q_type}"
            )

        section["type"] = q_type

        offered = _positive_int(
            section.get("offered_count"),
            f"{section_id}.offered_count"
        )

        answer = _positive_int(
            section.get("answer_count"),
            f"{section_id}.answer_count"
        )

        marks = _positive_int(
            section.get("marks_each"),
            f"{section_id}.marks_each"
        )

        if answer > offered:
            raise BlueprintError(
                f"Section {section_id}: answer_count "
                "cannot exceed offered_count."
            )

        # Validate Bloom distribution
        section["bloom_targets"] = _validate_targets(
            section.get("bloom_targets"),
            f"{section_id}.bloom_targets",
            offered,
            BLOOM_LEVELS
        )

        # Validate difficulty distribution
        section["difficulty_targets"] = _validate_targets(
            section.get("difficulty_targets"),
            f"{section_id}.difficulty_targets",
            offered,
            DIFFICULTY_LEVELS
        )

        # Calculate section marks
        section_marks = answer * marks

        section["section_marks"] = section_marks

        full_marks += section_marks
        total_offered += offered
        total_to_answer += answer

    # Verify teacher-specified full marks
    entered_marks = paper.get("full_marks")

    if entered_marks is not None:
        _positive_int(
            entered_marks, "full_marks"
        )

        if entered_marks != full_marks:
            raise BlueprintError(
                f"Full marks mismatch: entered "
                f"{entered_marks}, calculated {full_marks}."
            )

    # Store calculated values
    paper["full_marks"] = full_marks

    paper["total_offered_questions"] = total_offered

    paper["total_questions_to_answer"] = total_to_answer

    return paper


# ----------------------------------------------------
# 4. Load teacher's configuration from JSON
# ----------------------------------------------------

def load_blueprint(path):

    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return validate_blueprint(data)


# ----------------------------------------------------
# 5. Example examination blueprint
# ----------------------------------------------------

SAMPLE_BLUEPRINT = {

    "college_name": "Example Institute of Technology",

    "college_address": "Example City, India",

    "university_name": "Example University",

    "exam_name": "B.Tech Semester Examination",

    "course_name": "B.Tech",

    "branch": "Information Technology",

    "semester": "4",

    "academic_year": "2026-2027",

    "subject_name": "Design and Analysis of Algorithms",

    "paper_code": "PCC-CS404",

    "exam_date": "2026-12-15",

    "duration_minutes": 180,

    "full_marks": 70,

    "logo_path": "",

    "teacher_prompt": (
        "Cover important syllabus topics and "
        "avoid repeated concepts."
    ),

    "instructions": [
        "Figures in the margin indicate full marks.",
        "Answer the required number of questions "
        "from each group."
    ],

    "sections": [

        {
            "id": "A",
            "title": "Group A - WH Questions (1 mark)",
            "type": "WHQ",
            "offered_count": 12,
            "answer_count": 10,
            "marks_each": 1,

            "bloom_targets": {
                "Remember": 8,
                "Understand": 4
            },

            "difficulty_targets": {
                "Easy": 8,
                "Medium": 4
            }
        },

        {
            "id": "B",
            "title": "Group B - WH-Questions",
            "type": "WHQ",
            "offered_count": 5,
            "answer_count": 3,
            "marks_each": 5,

            "bloom_targets": {
                "Understand": 2,
                "Apply": 1,
                "Analyze": 2
            },

            "difficulty_targets": {
                "Easy": 1,
                "Medium": 3,
                "Hard": 1
            }
        },

        {
            "id": "C",
            "title": "Group B - WH-Questions (extended)",
            "type": "WHQ",
            "offered_count": 3,
            "answer_count": 3,
            "marks_each": 15,

            "bloom_targets": {
                "Analyze": 1,
                "Evaluate": 1,
                "Create": 1
            },

            "difficulty_targets": {
                "Medium": 1,
                "Hard": 2
            }
        }
    ]
}


# ----------------------------------------------------
# 6. Print examination blueprint
# ----------------------------------------------------

def print_summary(paper):

    print("\nAQZ EXAM BLUEPRINT")

    print(
        f"Paper: {paper['subject_name']} "
        f"({paper['paper_code']})"
    )

    print(f"College: {paper['college_name']}")

    print(f"Exam date: {paper['exam_date']}")

    print(
        f"Duration: {paper['duration_minutes']} minutes"
    )

    print("-" * 55)

    for section in paper["sections"]:

        print(
            f"{section['id']}: {section['type']} | "
            f"offered={section['offered_count']}, "
            f"answer={section['answer_count']}, "
            f"marks each={section['marks_each']}, "
            f"section marks={section['section_marks']}"
        )

    print("-" * 55)

    print(
        f"Questions offered: "
        f"{paper['total_offered_questions']}"
    )

    print(
        f"Questions to answer: "
        f"{paper['total_questions_to_answer']}"
    )

    print(f"Full marks: {paper['full_marks']}")

    print("Blueprint validation: PASSED")


# ----------------------------------------------------
# 7. Backend unit tests
# ----------------------------------------------------

def _self_test():

    sample = validate_blueprint(SAMPLE_BLUEPRINT)

    assert sample["full_marks"] == 70

    assert sample["total_offered_questions"] == 20

    assert sample["total_questions_to_answer"] == 16

    assert [
        s["section_marks"]
        for s in sample["sections"]
    ] == [10, 15, 45]

    # Test automatic full-mark calculation
    auto = copy.deepcopy(SAMPLE_BLUEPRINT)

    auto.pop("full_marks")

    assert validate_blueprint(auto)["full_marks"] == 70

    # Test True/False section
    extra = copy.deepcopy(SAMPLE_BLUEPRINT)

    extra["sections"].append({
        "id": "D",
        "title": "Group D - True or False",
        "type": "TRUE_FALSE",
        "offered_count": 5,
        "answer_count": 5,
        "marks_each": 1
    })

    extra["full_marks"] = 75

    assert validate_blueprint(extra)["full_marks"] == 75

    # Invalid blueprints must raise errors
    def must_fail(change):

        bad = copy.deepcopy(SAMPLE_BLUEPRINT)
        change(bad)

        try:
            validate_blueprint(bad)
        except BlueprintError:
            return

        raise AssertionError(
            "Invalid blueprint was accepted."
        )

    must_fail(
        lambda x: x.update(full_marks=80)
    )

    must_fail(
        lambda x: x["sections"][0].update(
            answer_count=13
        )
    )

    must_fail(
        lambda x: x["sections"][0].update(
            bloom_targets={"Remember": 5}
        )
    )

    must_fail(
        lambda x: x.update(
            exam_date="15/12/2026"
        )
    )

    must_fail(
        lambda x: x["sections"][0].update(
            marks_each=0
        )
    )

    must_fail(
        lambda x: x["sections"][1].update(
            id="A"
        )
    )

    print(
        "Self-test PASSED: valid and "
        "invalid exam blueprints"
    )


# ----------------------------------------------------
# 8. Command-line testing
# ----------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="AQZ exam blueprint validator"
    )

    group = parser.add_mutually_exclusive_group(
        required=True
    )

    group.add_argument(
        "--self-test",
        action="store_true"
    )

    group.add_argument(
        "--demo",
        action="store_true"
    )

    group.add_argument(
        "--write-sample",
        metavar="PATH"
    )

    group.add_argument(
        "--validate",
        metavar="PATH"
    )

    args = parser.parse_args()

    try:

        if args.self_test:
            _self_test()

        elif args.demo:
            print_summary(
                validate_blueprint(SAMPLE_BLUEPRINT)
            )

        elif args.write_sample:

            path = Path(args.write_sample)

            if path.exists():
                parser.error(
                    f"File already exists: {path}"
                )

            path.write_text(
                json.dumps(SAMPLE_BLUEPRINT, indent=2),
                encoding="utf-8"
            )

            print(
                f"Sample saved to {path}. "
                "Edit its values, then run --validate."
            )

        else:
            print_summary(
                load_blueprint(args.validate)
            )

    except (
        BlueprintError,
        OSError,
        json.JSONDecodeError
    ) as error:

        parser.error(str(error))


if __name__ == "__main__":
    main()
