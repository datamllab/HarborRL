"""Exact-prefix teacher and process-reward adapters, with explicit fake evidence."""
import operator
import re
from fractions import Fraction

from harborrl.distill.data import NUMBER
from harborrl.trajectories.native import digest


class FakeTeacher:
    """Known normalized test distribution; never presented as model inference."""
    revision = digest("fake-prefix-categorical-v1")
    evidence_kind = "synthetic"

    def __init__(self, identity, context):
        self.identity, self.context = identity, context

    def capabilities(self):
        return {"exact_token_scoring": True, "raw_logprob_semantics": True, "identity": self.identity,
                "max_context": self.context, "revision": self.revision, "evidence_kind": self.evidence_kind}

    def score_tokens(self, prompt, response, identity):
        import torch
        if identity != self.identity or len(prompt) + len(response) > self.context:
            raise ValueError("Fake scorer identity/context mismatch")
        logits = torch.linspace(-1, 1, identity["vocab_size"], dtype=torch.float64)
        base = logits.log_softmax(0)
        prefix, result = list(prompt), []
        for token in response:
            shift = sum(prefix) % identity["vocab_size"]
            result.append(float(base[(token - shift) % identity["vocab_size"]]))
            prefix.append(token)
        return result

    def unchanged(self):
        return True


class ModelTeacher:
    evidence_kind = "observed"

    def __init__(self, config, device, identity):
        from .models import load_student, parameter_digest
        self.model, self.tokenizer = load_student(config, device, teacher=True)
        self.identity = self.tokenizer.identity()
        if self.identity != identity:
            raise ValueError("Teacher and student tokenizer/template/vocabulary must match exactly")
        self.context = config["sampling"]["max_context"]
        self.revision = parameter_digest(self.model)

    def capabilities(self):
        return {"exact_token_scoring": True, "raw_logprob_semantics": True, "identity": self.identity,
                "max_context": self.context, "revision": self.revision, "evidence_kind": self.evidence_kind}

    def score_tokens(self, prompt, response, identity):
        import torch

        from .models import response_logprobs
        if identity != self.identity or len(prompt) + len(response) > self.context:
            raise ValueError("Teacher identity/context mismatch")
        with torch.no_grad():
            return response_logprobs(self.model, [prompt], [response], self.tokenizer.pad_token_id)[0].cpu().tolist()

    def unchanged(self):
        from .models import parameter_digest
        return all(parameter.grad is None for parameter in self.model.parameters()) and parameter_digest(self.model) == self.revision


def probe_teacher(teacher, tokenizer):
    """Exercise the actual adapter at the first-token boundary, EOS and longer prefixes."""
    import math
    capabilities = teacher.capabilities()
    prompt = tokenizer.encode("1 + 1 =")
    response = tokenizer.encode("2\n")[1:] + [tokenizer.eos_token_id]
    for tokens in ([tokenizer.eos_token_id], response):
        scores = teacher.score_tokens(prompt, tokens, tokenizer.identity())
        if len(scores) != len(tokens) or any(not math.isfinite(v) or v > 0 for v in scores):
            raise ValueError("Teacher scoring probe failed")
        for index in range(len(tokens)):
            separate = teacher.score_tokens(prompt + tokens[:index], [tokens[index]], tokenizer.identity())[0]
            if abs(separate - scores[index]) > 1e-4:
                raise ValueError("Teacher causal-prefix probe failed")
    return {**capabilities, "probe": "passed", "request_digest": digest({"prompt": prompt, "response": response})}


def process_rewards(kind, response_ids, tokenizer):
    """Each newline terminates a step; a boundary token belongs to that step."""
    if kind not in {"fake", "equation"} or not response_ids or len(response_ids) > 16384:
        raise ValueError("Invalid or oversized process-reward request")
    ends, previous = [], 0
    for index in range(len(response_ids)):
        count = tokenizer.decode(response_ids[:index + 1]).count("\n")
        if count > previous:
            ends.append(index + 1)
        previous = count
    if not ends or ends[-1] != len(response_ids):
        ends.append(len(response_ids))
    steps, start = [], 0
    for end in ends:
        text = tokenizer.decode(response_ids[start:end]).strip()
        if kind == "fake":
            score = (int(digest(response_ids[start:end])[:8], 16) % 9 + 1) / 10
            status = "synthetic"
        else:
            match = re.fullmatch(rf"\s*({NUMBER})\s*([+*/-])\s*({NUMBER})\s*=\s*({NUMBER})\s*", text)
            score, status = 0.0, "unsupported"
            if match and len(text) <= 1024:
                a, b, expected = Fraction(match[1]), Fraction(match[3]), Fraction(match[4])
                try:
                    actual = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv}[match[2]](a, b)
                    score = 1.0 if actual == expected else -1.0
                    status = "correct" if actual == expected else "incorrect"
                except ZeroDivisionError:
                    status = "unsupported"
        steps.append({"start": start, "end": end, "text": text, "score": score, "status": status})
        start = end
    values = [0.0] * len(response_ids)
    cumulative = 0.0
    for step in reversed(steps):
        cumulative += step["score"]
        values[step["start"]:step["end"]] = [cumulative] * (step["end"] - step["start"])
    return {"adapter": f"{kind}-prm-v1", "evidence_kind": "synthetic" if kind == "fake" else "observed",
            "steps": steps, "token_returns": values, "aggregation": "undiscounted-step-reward-to-go"}
