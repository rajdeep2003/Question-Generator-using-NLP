import torch

DEVICE = 0 if torch.cuda.is_available() else -1

QUESTION_MODEL = "valhalla/t5-base-e2e-qg"

# MCQ generation uses the supplied multitask QA-QG model for answer/question
# pairs, then the supplied T5-base model for three distractors.
MCQ_QG_MODEL = "valhalla/t5-small-qa-qg-hl"
DISTRACTOR_MODEL = "fares7elsadek/t5-base-distractor-generation"

CLASSIFIER_MODEL = "facebook/bart-large-mnli"

ANSWER_MODEL = "deepset/roberta-base-squad2"

CHUNK_WORDS = 180

CHUNK_OVERLAP = 30

BLOOM_LEVELS = [
    "Remember",
    "Understand",
    "Apply",
    "Analyze",
    "Evaluate",
    "Create"
]

DIFFICULTY_LEVELS = [
    "Easy",
    "Medium",
    "Hard"
]
