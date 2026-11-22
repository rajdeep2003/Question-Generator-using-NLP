"""Source grounded MCQ generation with T5 question and distractor models."""

import re
from functools import lru_cache

import torch

from aqz.config import DISTRACTOR_MODEL, MCQ_QG_MODEL
from aqz.generation.types.common import _sentences
from .trace import operation, _log, preview


@lru_cache(maxsize=1)
def _get_models():
    """Load the pasted MultiTaskQAQGPipeline and the HF distractor model once."""
    with operation("import Transformers model classes"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    with operation("import MultiTaskQAQGPipeline"):
        from .multitask_qg import MultiTaskQAQGPipeline

    with operation(f"load QA-QG tokenizer: {MCQ_QG_MODEL}"):
        qg_tokenizer = AutoTokenizer.from_pretrained(MCQ_QG_MODEL)
    with operation(f"load QA-QG model weights: {MCQ_QG_MODEL}"):
        qg_model = AutoModelForSeq2SeqLM.from_pretrained(MCQ_QG_MODEL)
    with operation("detect CUDA availability"):
        use_cuda = torch.cuda.is_available()
    with operation("initialize MultiTaskQAQGPipeline"):
        qg = MultiTaskQAQGPipeline(
            model=qg_model,
            tokenizer=qg_tokenizer,
            ans_model=qg_model,
            ans_tokenizer=qg_tokenizer,
            qg_format="highlight",
            use_cuda=use_cuda,
        )

    with operation(f"load distractor tokenizer: {DISTRACTOR_MODEL}"):
        distractor_tokenizer = AutoTokenizer.from_pretrained(DISTRACTOR_MODEL)
    with operation(f"load distractor model weights: {DISTRACTOR_MODEL}"):
        distractor_model = AutoModelForSeq2SeqLM.from_pretrained(DISTRACTOR_MODEL)
    with operation("select distractor model device"):
        device = torch.device("cuda" if use_cuda else "cpu")
    with operation(f"move distractor model to {device}"):
        distractor_model.to(device)
    return qg, distractor_tokenizer, distractor_model, device


def _generate_distractors_batch(rows, tokenizer, model, device):
    # The model card specifies question <sep> context <sep> correct answer.
    with operation("format distractor prompts"):
        prompts = [f"{question} <sep> {context} <sep> {answer}"
                   for question, context, answer in rows]
        for index, prompt in enumerate(prompts, start=1):
            _log(f"distractor input {index}: {preview(prompt, 320)!r}")
    with operation("tokenize distractor prompts"):
        encoded = tokenizer(
            prompts, return_tensors="pt", truncation=True, padding=True, max_length=256
        )
    with operation(f"move {len(encoded['input_ids'])} tokenized prompt(s) to {device}"):
        encoded = {key: value.to(device) for key, value in encoded.items()}
    token_count = encoded["input_ids"].shape[1]
    _log(f"distractor inference input shape={tuple(encoded['input_ids'].shape)} tokens={token_count}")
    with operation("run distractor model.generate (max_new_tokens=24, num_beams=1)"):
        with torch.inference_mode():
            # Greedy decoding keeps the CPU path responsive.
            outputs = model.generate(**encoded, max_new_tokens=24, num_beams=1)
    parsed = []
    with operation("decode and validate distractor outputs"):
        for row, output in zip(rows, outputs):
            decoded = tokenizer.decode(output, skip_special_tokens=True)
            values = re.split(r"\s*<sep>\s*", decoded)
            clean, seen = [], {row[2].strip().casefold()}
            for value in values:
                value = re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:")
                key = value.casefold()
                if len(value) < 2 or key in seen:
                    continue
                seen.add(key)
                clean.append(value)
                if len(clean) == 3:
                    break
            parsed.append(clean)
            _log(f"distractor output for answer={preview(row[2], 80)!r}: raw={preview(decoded, 260)!r}; parsed={clean!r}")
    return parsed


def _fallback_seeds(chunk):
    """Use explicit definitions if the QG model produces no answer pairs."""
    with operation("compile source-definition fallback pattern"):
        pattern = re.compile(
            r"^(?P<subject>[A-Z][A-Za-z0-9 &'/-]{1,55}?)\s+(?:is|are)\s+"
            r"(?P<definition>(?:a|an|the)\s+[^.!?]{8,150})[.!?]$"
        )
    with operation("extract source-definition fallback seeds"):
        return [
            (f"Which option best describes {match.group('subject')}?", sentence,
             match.group("definition").rstrip(".!?"))
            for sentence in _sentences(chunk)
            if (match := pattern.match(sentence.strip()))
        ]


def generate(chunk, max_count=10, use_model=True):
    """Generate MCQs with QA-QG answers and three T5 distractors.

    The optional ``use_model=False`` path is kept for fast/offline workflows;
    it uses source definitions as choices and never downloads a model.
    """
    _log(f"starting generation: use_model={use_model}, max_count={max_count}, source_chars={len(str(chunk))}")
    if use_model:
        try:
            with operation("get cached or load MCQ models"):
                qg, tokenizer, distractor_model, device = _get_models()
            # QG runs answer extraction once per sentence; bound CPU work for
            # long PDFs, then generate only as many distractor sets as needed.
            with operation("split source into sentences and cap at 32"):
                source_sentences = _sentences(chunk)
                source = " ".join(source_sentences[:32])
                _log(f"source stage: {len(source_sentences)} sentence(s) found; {min(len(source_sentences), 32)} passed to QG; chars={len(source)}; preview={preview(source)}")
            with operation("generate QA-QG answer/question pairs"):
                raw_seeds = qg(source)
                _log(f"QA-QG stage returned {len(raw_seeds)} raw seed(s)")
            with operation("validate and cap QA-QG seeds"):
                valid_items = [
                    item for item in raw_seeds
                    if all((item.get("question", "").strip(), item.get("context", "").strip(),
                            item.get("answer", "").strip()))
                ]
                seeds = [
                    (item.get("question", "").strip(), item.get("context", "").strip(),
                     item.get("answer", "").strip())
                    for item in valid_items
                ][:max_count]
                _log(f"seed validation: accepted={len(valid_items)}, missing_required_fields={len(raw_seeds) - len(valid_items)}, capped_to={len(seeds)}")
                for index, (question, context, answer) in enumerate(seeds[:10], start=1):
                    _log(f"seed {index}: question={preview(question)!r}; answer={preview(answer, 100)!r}; context={preview(context)!r}")
            if not seeds:
                with operation("create fallback seeds from source definitions"):
                    seeds = _fallback_seeds(source)
                    _log(f"definition fallback extracted {len(seeds)} seed(s)")
                    for index, seed in enumerate(seeds[:10], start=1):
                        _log(f"fallback seed {index}: question={preview(seed[0])!r}; answer={preview(seed[2], 100)!r}; context={preview(seed[1])!r}")
            _log(f"using {len(seeds)} seed(s), max_count={max_count}")
            questions = []
            for batch_start in range(0, len(seeds), 4):
                batch = seeds[batch_start:batch_start + 4]
                _log(f"starting distractor batch {batch_start + 1}-{batch_start + len(batch)} of {len(seeds)}")
                batch_options = _generate_distractors_batch(
                    batch, tokenizer, distractor_model, device
                )
                for (question, context, answer), options in zip(batch, batch_options):
                    if len(options) != 3:
                        _log(f"skipping MCQ seed: model returned {len(options)} distractors; expected 3")
                        continue
                    correct_position = len(questions) % 4
                    options.insert(correct_position, answer)
                    questions.append({
                        "question": question,
                        "context": context,
                        "question_type": "MCQ",
                        "options": options,
                        "answer": answer,
                        "answer_index": correct_position,
                        "answer_valid": True,
                        "answer_kind": "model_answer_source_context",
                        "explanation": context,
                        "suggested_marks": 1,
                    })
                    _log(f"assembled MCQ {len(questions)} with 4 options; correct_index={correct_position}")
                    if len(questions) >= max_count:
                        break
                if len(questions) >= max_count:
                    break
            _log(f"model MCQ generation complete: {len(questions)} question(s)")
            return questions
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            # Callers may still produce offline candidates from source definitions.
            # Keep the failure visible in logs while allowing the existing workflow.
            import logging
            logging.getLogger(__name__).warning("MCQ models unavailable: %s", exc)
            _log(f"model MCQ path failed; falling back to source definitions: {type(exc).__name__}: {exc}")

    with operation("run definition-based fallback generation"):
        definitions = _fallback_seeds(chunk)
    if len(definitions) < 4:
        _log(f"fallback skipped: only {len(definitions)} distinct definition seed(s)")
        return []
    questions = []
    for index, (question, context, correct) in enumerate(definitions):
        with operation(f"build fallback options for seed {index + 1}"):
            distractors = [row[2] for row in definitions if row[2].casefold() != correct.casefold()][:3]
        if len(distractors) != 3:
            continue
        correct_position = index % 4
        options = distractors
        options.insert(correct_position, correct)
        questions.append({
            "question": question, "context": context, "question_type": "MCQ",
            "options": options, "answer": correct, "answer_index": correct_position,
            "answer_valid": True, "answer_kind": "source_definition",
            "explanation": context, "suggested_marks": 1,
        })
        if len(questions) >= max_count:
            break
    _log(f"definition-based fallback complete: {len(questions)} question(s)")
    return questions
