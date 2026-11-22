"""Source grounded MCQ generation with T5 question and distractor models."""

import re
import os
import posixpath
import shutil
import tarfile
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

import torch

from aqz.config import DISTRACTOR_MODEL, MCQ_QG_MODEL
from aqz.generation.types.common import _sentences
from .trace import operation, _log, preview


_DISTRACTOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "their", "this", "to", "was", "were", "which", "with",
}
_SENSE2VEC_DOWNLOAD_URL = (
    "https://github.com/explosion/sense2vec/releases/download/v1.0.0/"
    "s2v_reddit_2015_md.tar.gz"
)
_SENSE2VEC_DOWNLOAD_LOCK = threading.Lock()


def _content_terms(text):
    return {
        token.casefold() for token in re.findall(r"[A-Za-z0-9]+", str(text))
        if len(token) > 2 and token.casefold() not in _DISTRACTOR_STOPWORDS
    }


def _download_sense2vec(model_path):
    """Download and safely extract the Reddit 2015 vectors on first use."""
    with _SENSE2VEC_DOWNLOAD_LOCK:
        if model_path.is_dir():
            return True
        archive_path = model_path.parent / "s2v_reddit_2015_md.tar.gz"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _log("downloading Sense2Vec Reddit 2015 vectors (about 573 MB)")
        extraction_started = False
        try:
            with urllib.request.urlopen(_SENSE2VEC_DOWNLOAD_URL, timeout=60) as response:
                total = int(response.headers.get("Content-Length", "0"))
                downloaded = 0
                last_reported = -1
                with partial_path.open("wb") as output:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        downloaded += len(block)
                        percent = int(downloaded * 100 / total) if total else 0
                        if percent >= last_reported + 10:
                            _log(f"Sense2Vec download progress: {percent}%")
                            last_reported = percent
            partial_path.replace(archive_path)

            extraction_started = True
            with tarfile.open(archive_path, "r:gz") as archive:
                members = archive.getmembers()
                for member in members:
                    normalized_name = member.name.replace("\\", "/")
                    normalized_name = posixpath.normpath(normalized_name)
                    member_path = Path(normalized_name)
                    if (member_path.is_absolute() or normalized_name == ".."
                            or normalized_name.startswith("../")
                            or ":" in normalized_name.split("/")[0]
                            or member.issym()
                            or member.islnk() or not (member.isdir() or member.isfile())):
                        raise ValueError(f"Unsafe Sense2Vec archive entry: {member.name}")
                    member.name = normalized_name
                archive.extractall(model_path.parent)

            extracted = model_path.parent / "s2v_reddit_2015_md"
            if not extracted.is_dir():
                extracted = model_path.parent / "s2v_old"
            if extracted.is_dir() and extracted.resolve() != model_path.resolve():
                if model_path.exists():
                    shutil.rmtree(model_path)
                extracted.replace(model_path)
            if not model_path.is_dir():
                raise OSError("Sense2Vec archive did not contain the expected model directory")
            _log(f"Sense2Vec vectors extracted to {model_path}")
            return True
        except (OSError, tarfile.TarError, ValueError, urllib.error.URLError) as exc:
            _log(f"Sense2Vec download failed; using lexical ranking: {exc}")
            if extraction_started:
                for incomplete_path in (model_path, model_path.parent / "s2v_reddit_2015_md"):
                    if incomplete_path.is_dir():
                        shutil.rmtree(incomplete_path, ignore_errors=True)
            return False
        finally:
            for temporary_path in (partial_path, archive_path):
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


@lru_cache(maxsize=1)
def _sense2vec_model_path():
    return Path(os.environ.get(
        "SENSE2VEC_MODEL_PATH",
        Path(__file__).resolve().parents[4] / "models" / "s2v_reddit_2015_md",
    ))


def _ensure_sense2vec_vectors():
    """Return the local Reddit 2015 vectors path, downloading if needed."""
    model_path = _sense2vec_model_path()
    legacy_archive_path = model_path.parent / "s2v_old"
    if not model_path.is_dir() and legacy_archive_path.is_dir():
        legacy_archive_path.replace(model_path)
    if not model_path.is_dir():
        try:
            import sense2vec  # noqa: F401
        except ImportError as exc:
            _log(f"Sense2Vec package is missing; using lexical ranking: {exc}")
            return None
        if not _download_sense2vec(model_path):
            return None
    return model_path


@lru_cache(maxsize=1)
def _load_sense2vec():
    """Load or automatically download the Reddit 2015 vectors."""
    model_path = _ensure_sense2vec_vectors()
    if model_path is None:
        return None
    try:
        from sense2vec import Sense2Vec
        model = Sense2Vec().from_disk(str(model_path))
        _log(f"loaded Sense2Vec 2015 vectors from {model_path}")
        return model
    except (ImportError, OSError, ValueError) as exc:
        _log(f"could not load Sense2Vec vectors; using lexical ranking: {exc}")
        return None


def _sense2vec_neighbors(answer, model):
    """Return a normalized similarity map for the answer's available senses."""
    if model is None:
        return {}
    phrase = re.sub(r"[^\w\s'-]", " ", answer.casefold())
    phrase = re.sub(r"\s+", "_", phrase.strip())
    if not phrase:
        return {}
    neighbors = {}
    for sense in ("NOUN", "PROPN", "VERB", "ADJ", "ADV"):
        key = f"{phrase}|{sense}"
        try:
            if key not in model:
                continue
            for neighbor, similarity in model.most_similar(key, n=100):
                value = neighbor.rsplit("|", 1)[0].replace("_", " ").strip()
                normalized = re.sub(r"\W+", " ", value).casefold().strip()
                neighbors[normalized] = max(float(similarity), neighbors.get(normalized, 0.0))
        except (KeyError, ValueError):
            continue
    return neighbors


def _rank_source_distractors(question, context, answer, answer_pool):
    """Rank extracted answers by relevance to the current question/source."""
    target_terms = _content_terms(f"{question} {context}") - _content_terms(answer)
    sense2vec = _load_sense2vec()
    semantic_neighbors = _sense2vec_neighbors(answer, sense2vec)
    ranked = []
    for alternative, alternative_context, alternative_question in answer_pool:
        if alternative.casefold() == answer.casefold():
            continue
        if len(alternative) > 100 or len(alternative.split()) > 12:
            continue
        candidate_terms = _content_terms(alternative)
        context_terms = _content_terms(alternative_context)
        question_terms = _content_terms(alternative_question)
        answer_overlap = len(candidate_terms & target_terms)
        context_overlap = len(context_terms & target_terms)
        question_overlap = len(question_terms & _content_terms(question))
        context_union = len(context_terms | target_terms) or 1
        question_union = len(question_terms | _content_terms(question)) or 1
        normalized_alternative = re.sub(
            r"\W+", " ", alternative
        ).casefold().strip()
        semantic_score = semantic_neighbors.get(normalized_alternative, 0.0)
        # Prefer concise answers tied to questions and contexts using the same
        # topic vocabulary. Sense2Vec similarity adds a semantic signal when
        # the 2015 vectors contain both terms; course-source overlap remains
        # necessary because only extracted source answers enter answer_pool.
        score = (4 * answer_overlap
                 + 2 * question_overlap / question_union
                 + context_overlap / context_union
                 + 8 * semantic_score)
        ranked.append((score, alternative.casefold(), alternative))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return [alternative for _, _, alternative in ranked]


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
                if (len(value) < 2 or len(value) > 100 or len(value.split()) > 12
                        or key in seen):
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


def _append_source_cloze_fallbacks(questions, seeds, max_count, answer_pool):
    """Fill a remaining MCQ shortage with source-verbatim cloze questions."""
    seen_questions = {
        re.sub(r"\W+", " ", item.get("question", "")).casefold().strip()
        for item in questions
    }
    for question, context, answer in seeds:
        if len(questions) >= max_count:
            break
        clean_answer = re.sub(r"\s+", " ", answer).strip(" .,:;-")
        if len(clean_answer) < 2 or len(clean_answer) > 100 or len(clean_answer.split()) > 12:
            continue
        # Replace only an exact phrase from this source context. This keeps the
        # fallback answer verifiable from the passage.
        match = re.search(re.escape(clean_answer), context, re.IGNORECASE)
        if not match:
            continue
        cloze = context[:match.start()] + "_____" + context[match.end():]
        fallback_question = f"Choose the answer that completes the source statement: {cloze}"
        normalized = re.sub(r"\W+", " ", fallback_question).casefold().strip()
        if normalized in seen_questions:
            continue
        distractors = _rank_source_distractors(
            fallback_question, context, clean_answer, answer_pool
        )[:3]
        if len(distractors) != 3:
            continue
        correct_position = len(questions) % 4
        options = list(distractors)
        options.insert(correct_position, clean_answer)
        questions.append({
            "question": fallback_question,
            "context": context,
            "question_type": "MCQ",
            "options": options,
            "answer": clean_answer,
            "answer_index": correct_position,
            "answer_valid": True,
            "answer_kind": "source_cloze_fallback",
            "explanation": context,
            "suggested_marks": 1,
        })
        seen_questions.add(normalized)
        _log(f"assembled source-cloze fallback MCQ {len(questions)}")
    return questions


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
                ]
                # Preserve a broad pool: poor distractor outputs may invalidate
                # several seeds before we assemble the requested count.
                seeds = seeds[:max(max_count * 8, max_count)]
                _log(f"seed validation: accepted={len(valid_items)}, missing_required_fields={len(raw_seeds) - len(valid_items)}, retained_for_recovery={len(seeds)}")
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
            answer_pool = []
            answer_seen = set()
            for answer_question, answer_context, answer in seeds:
                cleaned_answer = re.sub(r"\s+", " ", answer).strip(" .,:;-")
                key = cleaned_answer.casefold()
                if (2 <= len(cleaned_answer) <= 100
                        and len(cleaned_answer.split()) <= 12
                        and key not in answer_seen):
                    answer_seen.add(key)
                    answer_pool.append((cleaned_answer, answer_context, answer_question))
            seen_questions = set()
            for batch_start in range(0, len(seeds), 4):
                batch = seeds[batch_start:batch_start + 4]
                _log(f"starting distractor batch {batch_start + 1}-{batch_start + len(batch)} of {len(seeds)}")
                batch_options = _generate_distractors_batch(
                    batch, tokenizer, distractor_model, device
                )
                for (question, context, answer), options in zip(batch, batch_options):
                    # Top up weak model output with distinct, source-extracted
                    # answers from other candidate seeds.
                    if len(options) < 3:
                        ranked_alternatives = _rank_source_distractors(
                            question, context, answer, answer_pool
                        )
                        for alternative in ranked_alternatives:
                            if len(options) >= 3:
                                break
                            if alternative.casefold() != answer.casefold() and all(
                                alternative.casefold() != str(option).casefold() for option in options
                            ):
                                options.append(alternative)
                    if len(options) != 3:
                        _log(f"skipping MCQ seed: only {len(options)} distinct source-grounded distractors available")
                        continue
                    normalized_question = re.sub(r"\W+", " ", question).casefold().strip()
                    if normalized_question in seen_questions:
                        continue
                    seen_questions.add(normalized_question)
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
            if len(questions) < max_count:
                _log(f"MCQ shortage fallback: have {len(questions)}, requested {max_count}")
                questions = _append_source_cloze_fallbacks(
                    questions, seeds, max_count, answer_pool
                )
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
