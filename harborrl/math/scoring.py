"""Token-aligned arithmetic process rewards; no training loop."""
import operator
import re
from fractions import Fraction

from harborrl.math.data import NUMBER
from harborrl.trajectories.native import digest


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
