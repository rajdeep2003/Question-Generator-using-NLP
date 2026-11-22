from transformers import pipeline
## ignore this file

MODEL_NAME = "valhalla/t5-base-qa-qg-hl"

print("Loading model...")

nlp = pipeline(
"text2text-generation",
model=MODEL_NAME
)

paragraph = """
The binary search algorithm works on a sorted array.
It repeatedly divides the search interval into two halves.
Its time complexity is O(log n).
"""

print("\nGenerating questions...\n")

result = nlp(
"generate question: <hl> sorted array <hl> "
"The binary search algorithm works on a sorted array. "
"It repeatedly divides the search interval into two halves. "
"Its time complexity is O(log n). </s>",
max_new_tokens=64
)

for item in result:
    print(item["generated_text"])
