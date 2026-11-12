"""Locked math tasks and bounded exact-rational verification; never eval text."""
import json
import re
from fractions import Fraction
from pathlib import Path

from harborrl.trajectories.native import digest, publish

NUMBER = r"[+-]?(?:\d{1,32}(?:\.\d{1,16})?|\d{1,32}/[1-9]\d{0,31})"
GSM_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3}){1,10}(?:\.\d{1,16})?|\d{1,32}(?:\.\d{1,16})?|\d{1,32}/[1-9]\d{0,31})"
VERIFIERS = {"exact_rational_v1", "gsm8k_numeric_v1"}
TASK_FIELDS = {"task_id", "problem_id", "prompt", "reference", "verifier", "split", "source", "license"}


def load_tasks(path):
    rows, identities, problems, prompts = [], set(), {}, {}
    source = Path(path)
    if source.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Task file exceeds the 16 MiB input limit")
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != TASK_FIELDS:
            raise ValueError(f"Task line {index} requires exactly {sorted(TASK_FIELDS)}")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError("Task fields must be nonempty strings")
        if row["split"] not in {"train", "validation", "test"} or row["verifier"] not in VERIFIERS:
            raise ValueError("Only split-locked tasks with a supported numeric verifier are supported")
        if len(row["prompt"]) > 8192 or not re.fullmatch(NUMBER, row["reference"]):
            raise ValueError("Unsupported prompt length or rational reference")
        if row["task_id"] in identities:
            raise ValueError("Duplicate task_id")
        identities.add(row["task_id"])
        for key, table in ((row["problem_id"], problems), (digest(row["prompt"]), prompts)):
            if key in table and table[key] != row["split"]:
                raise ValueError("Problem or prompt leakage across splits")
            table[key] = row["split"]
        rows.append(row)
    if not rows:
        raise ValueError("Empty math task source")
    return rows


def _gsm_answer(response):
    candidates = []
    for pattern in (r"(?m)^[ \t]*####[ \t]+([^\n]{1,100})$", r"\\boxed\{([^{}]{1,100})\}"):
        for match in re.finditer(pattern, response):
            text = match[1].strip()
            for wrapper in ("**", "$"):
                if text.startswith(wrapper) and text.endswith(wrapper) and len(text) >= 2 * len(wrapper):
                    text = text[len(wrapper):-len(wrapper)].strip()
            if re.fullmatch(GSM_NUMBER, text):
                candidates.append((match.start(), text.replace(",", "")))
    if candidates:
        return max(candidates)[1]
    text = response.strip()
    return text if re.fullmatch(NUMBER, text) else ""


def verify(response, reference, verifier="exact_rational_v1"):
    if verifier not in VERIFIERS:
        raise ValueError(f"Unsupported numeric verifier: {verifier}")
    record = {"verifier": verifier, "input_digest": digest({"response": response, "reference": reference}),
              "status": "unparseable", "reward": 0.0, "parsed": None}
    if not isinstance(response, str) or len(response) > 16384:
        record["status"] = "unsupported"
        return record
    text = response.strip()
    if verifier == "gsm8k_numeric_v1":
        text = _gsm_answer(text)
    elif "####" in text:
        text = text.rsplit("####", 1)[1].strip()
    elif "\\boxed{" in text:
        match = re.search(r"\\boxed\{([^{}]{1,100})\}\s*$", text)
        text = match[1] if match else ""
    if not re.fullmatch(NUMBER, text):
        return record
    correct = Fraction(text) == Fraction(reference)
    incorrect_reward = 0.0 if verifier == "gsm8k_numeric_v1" else -1.0
    record.update(status="correct" if correct else "incorrect", reward=1.0 if correct else incorrect_reward, parsed=text)
    return record


def import_samples(tasks_path, samples_path, output):
    tasks = {row["task_id"]: row for row in load_tasks(tasks_path)}
    root = Path(output)
    if root.exists():
        raise ValueError("Dataset output must be a new directory")
    if Path(samples_path).stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Sample file exceeds the 16 MiB input limit")
    raw = Path(samples_path).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if not rows:
        raise ValueError("Empty sample source")
    prepared = []
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"task_id", "response", "source", "license"}:
            raise ValueError("Imported row requires task_id, response, source and license")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError("Sample values must be nonempty strings")
        task = tasks[row["task_id"]]
        record = {"schema_version": "math-sample-v1", "task_id": task["task_id"], "problem_id": task["problem_id"],
                  "split": task["split"], "prompt": task["prompt"], "response": row["response"],
                  "source_kind": "curated", "evidence_kind": "declared", "status": "collected",
                  "source": row["source"], "license": row["license"], "source_digest": digest(raw.hex()),
                  "task_digest": digest(task), "parent_digest": None, "token_evidence": None}
        record["sample_id"] = digest(record)
        if record["sample_id"] in identities:
            raise ValueError("Duplicate imported sample")
        identities.add(record["sample_id"])
        prepared.append((record, verify(row["response"], task["reference"], task["verifier"])))
    root.mkdir(parents=True)
    for sample, verification in prepared:
        publish(root / "samples" / f"{sample['sample_id']}.json", sample)
        publish(root / "verification" / f"{sample['sample_id']}.json", {**verification, "sample_digest": digest(sample)})
    manifest = {"schema_version": "math-dataset-v1", "tasks_digest": digest(list(tasks.values())),
                "samples": [{"sample_id": s["sample_id"], "digest": digest(s), "verification_digest": digest({**v, "sample_digest": digest(s)})}
                            for s, v in prepared], "training_eligibility": "not_an_on_policy_batch"}
    publish(root / "manifest.json", manifest)
    return manifest
