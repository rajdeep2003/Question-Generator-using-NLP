import torch

DEVICE = 0 if torch.cuda.is_available() else -1

QUESTION_MODEL = "valhalla/t5-base-e2e-qg"

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