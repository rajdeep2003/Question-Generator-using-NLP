"""Small local implementation of the supplied MultiTaskQAQGPipeline.

It follows the pasted pipeline's highlighted-answer format for QG and its
``question: ... context: ...`` format for QA. MCQ generation uses its QG path.
"""

import torch
from aqz.generation.types.common import _sentences
from .trace import operation, _log, preview


class MultiTaskQAQGPipeline:
    def __init__(self, model, tokenizer, ans_model, ans_tokenizer,
                 qg_format="highlight", use_cuda=True):
        with operation("assign MultiTaskQAQGPipeline model components"):
            self.model = model
            self.tokenizer = tokenizer
            self.ans_model = ans_model
            self.ans_tokenizer = ans_tokenizer
            self.qg_format = qg_format
        with operation("select MultiTaskQAQGPipeline device"):
            self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        with operation(f"move question-generation model to {self.device}"):
            self.model.to(self.device)
        if self.ans_model is not self.model:
            with operation(f"move answer-extraction model to {self.device}"):
                self.ans_model.to(self.device)
        with operation("detect MultiTaskQAQGPipeline model type"):
            self.model_type = "t5" if "T5" in self.model.__class__.__name__ else "bart"

    def __call__(self, inputs):
        if not isinstance(inputs, str):
            return self._extract_answer(inputs["question"], inputs["context"])
        with operation("normalize MultiTaskQAQGPipeline source text"):
            text = " ".join(inputs.split())
        with operation("split MultiTaskQAQGPipeline source sentences"):
            sentences = _sentences(text)
        if not sentences:
            _log("no usable sentences; returning no QA-QG pairs")
            return []
        _log(f"source has {len(sentences)} sentence(s), {len(text.split())} word(s)")
        _log(f"source preview: {preview(text)}")
        with operation("prepare highlighted answer-extraction inputs"):
            answer_inputs = []
            for index in range(len(sentences)):
                marked = [
                    f"<hl> {sentence} <hl>" if index == position else sentence
                    for position, sentence in enumerate(sentences)
                ]
                source = "extract answers: " + " ".join(marked)
                if self.model_type == "t5":
                    source += " </s>"
                answer_inputs.append(source)
        encoded = self._tokenize(answer_inputs, padding=True)
        with operation(f"move {len(encoded['input_ids'])} answer input(s) to {self.device}"):
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        with operation("run QA-QG answer-extraction model.generate"):
            with torch.inference_mode():
                answer_outputs = self.ans_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=32,
                )
        with operation("decode extracted answers and prepare question inputs"):
            pairs = []
            for sentence, output in zip(sentences, answer_outputs):
                decoded = self.ans_tokenizer.decode(output, skip_special_tokens=False)
                answers = decoded.split("<sep>")
                if answers and not answers[-1].strip():
                    answers.pop()
                for answer in answers:
                    answer = answer.strip()
                    if not answer or answer in {"</s>", "<pad>"}:
                        continue
                    marked_sentence = sentence.replace(answer, f"<hl> {answer} <hl>", 1)
                    source = f"generate question: {marked_sentence}"
                    if self.model_type == "t5":
                        source += " </s>"
                    pairs.append((answer, source, sentence))
            _log(f"answer extraction output: {len(pairs)} question-generation input(s)")
            for index, (answer, qg_input, context) in enumerate(pairs[:5], start=1):
                _log(f"QG input {index}: answer={preview(answer, 100)!r}; context={preview(context)!r}; prompt={preview(qg_input)!r}")
            if len(pairs) > 5:
                _log(f"QG input previews omitted: {len(pairs) - 5}")
        if not pairs:
            return []
        encoded = self._tokenize([source for _, source, _ in pairs], padding=True)
        with operation(f"move {len(encoded['input_ids'])} question input(s) to {self.device}"):
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        with operation("run QA-QG question-generation model.generate"):
            with torch.inference_mode():
                question_outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=32,
                    num_beams=4,
                )
        with operation("decode generated questions and assemble QA-QG pairs"):
            result = []
            for (answer, _, context), output in zip(pairs, question_outputs):
                question = self.tokenizer.decode(output, skip_special_tokens=True).strip()
                if question:
                    result.append({"answer": answer, "question": question, "context": context})
            _log(f"QA-QG output: {len(result)} question/answer/context pair(s)")
            for index, item in enumerate(result[:10], start=1):
                _log(f"QA-QG pair {index}: question={preview(item['question'])!r}; answer={preview(item['answer'], 100)!r}; context={preview(item['context'])!r}")
            if len(result) > 10:
                _log(f"QA-QG pair previews omitted: {len(result) - 10}")
        return result

    def _tokenize(self, inputs, padding=True):
        with operation(f"tokenize {len(inputs)} input(s) (padding={padding})"):
            encoded = self.tokenizer(
                inputs, max_length=512, truncation=True,
                padding="max_length" if padding else False, return_tensors="pt",
            )
        _log(f"tokenized tensor shape={tuple(encoded['input_ids'].shape)}")
        return encoded

    def _extract_answer(self, question, context):
        with operation("format QA-mode question/context input"):
            source = f"question: {question} context: {context}"
            if self.model_type == "t5":
                source += " </s>"
        encoded = self._tokenize([source], padding=False)
        with operation(f"move QA input to {self.device}"):
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)
        with operation("run QA-mode model.generate"):
            with torch.inference_mode():
                output = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=16,
                )
        with operation("decode QA-mode answer"):
            answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return answer
