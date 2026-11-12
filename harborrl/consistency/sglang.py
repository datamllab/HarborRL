"""Exact-token teacher-forcing probes against every synchronized inference engine."""
import json
from pathlib import Path
from urllib import request

from harborrl.trajectories.native import digest, finite, publish, read_json

from .config import environment, violations
from .metrics import merge
from .report import assess, decide


def http_json(url, payload=None, timeout=120):
    opener = request.build_opener(request.ProxyHandler({}))
    req = request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def score(endpoint, prompt, response, version, *, transport=http_json, label="Canary"):
    tokens = prompt + response
    payload = {"input_ids": tokens, "sampling_params": {"temperature": 1.0, "top_p": 1.0,
               "top_k": -1, "max_new_tokens": 0}, "return_logprob": True, "logprob_start_len": 0}
    output = transport(endpoint.rstrip("/") + "/generate", payload)
    meta = output.get("meta_info", {})
    if str(meta.get("weight_version", "")) != str(version):
        raise ValueError(f"{label} generation-time weight version mismatch")
    rows = meta.get("input_token_logprobs")
    if (not isinstance(rows, list) or len(rows) != len(tokens)
            or any(not isinstance(row, (tuple, list)) or len(row) < 2 for row in rows)
            or [row[1] for row in rows] != tokens):
        raise ValueError(f"{label} input logprobs must preserve every exact token and position")
    values = [row[0] for row in rows[len(prompt):]]
    if any(not finite(value) or value > 0 for value in values):
        raise ValueError(f"Invalid {label} raw logprob")
    return values


def check_pool(states, reference, config, *, transport=http_json):
    expected = reference["weight_version"]
    endpoints = [s["endpoint"] for s in states]
    if not states or len(set(endpoints)) != len(endpoints):
        raise ValueError("Canary requires a nonempty unique engine pool")
    if any(not s.get("healthy") or str(s.get("weight_version")) != str(expected) for s in states):
        raise ValueError("Canary pool contains stale or unhealthy engines")
    parts = []
    for endpoint in endpoints:
        try:
            rows = [(case["logprobs"], score(endpoint, case["prompt_ids"], case["response_ids"], expected,
                                            transport=transport), [1] * len(case["response_ids"])) for case in reference["cases"]]
            part = assess(rows, config, metadata={"engine": endpoint, "weight_version": expected})
            # Detect a version transition across scoring, in addition to each response's version.
            if str(transport(endpoint + "/get_weight_version").get("weight_version")) != str(expected):
                raise ValueError("Engine changed weights during canary scoring")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            part = {"statistics": None, "error": str(exc), "metadata": {"engine": endpoint}}
        parts.append(part)
    report = decide(parts, config, phase="synchronized-pool")
    # Mean/fraction limits apply independently to every engine.
    engine_errors = [f"{p['metadata']['engine']}: {error}" for p in parts if p['statistics']
                     for error in violations(merge([p['statistics']]), config)]
    report['numeric_violations'] = engine_errors
    if engine_errors and report['status'] != 'failed':
        report['status'] = 'failed' if config['mode'] == 'enforce' else 'warning'
    return report


def admit_pool(states, run_dir):
    config = environment()
    if config["mode"] == "off":
        return
    versions = {str(s["weight_version"]) for s in states}
    if len(versions) != 1:
        raise ValueError("Cannot admit a mixed-version pool")
    version = versions.pop()
    if not version.isdecimal():
        raise ValueError("Invalid numeric serving version")
    root = Path(run_dir) / "consistency"
    reference = read_json(root / f"sync-{version}.json")
    if reference["weight_version"] != version or reference["digest"] != digest({k: v for k, v in reference.items() if k != "digest"}):
        raise ValueError("Canary reference identity mismatch")
    admission = digest({'states': states, 'config': config, 'reference': reference['digest']})
    path = root / f"pool-{version}-{admission}.json"
    if path.exists():
        report = read_json(path)
    else:
        report = check_pool(states, reference, config)
    report["reference_digest"] = reference["digest"]
    publish(path, report)
    if report["status"] == "failed":
        raise ValueError(f"Serving canary rejected version {version}")
