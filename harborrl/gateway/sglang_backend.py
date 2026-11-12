"""Direct /generate adapter with explicit tokenizer and generation-time evidence.

The live engine must return weight_version in meta_info and be exclusively
updated through the caller's drained pool barrier. Before/after probes alone
are deliberately insufficient. This adapter does not load models on import.
"""
import json
from urllib import request

from harborrl.trajectories.native import digest, finite

from .messages import ProtocolError


class SGLangBackend:
    def __init__(self, endpoint, tokenizer, model_adapter, *, tokenizer_digest,
                 template_digest, timeout=120, max_context=32768,
                 audited_raw_logprobs=False, sampling_seed=None):
        if sampling_seed is not None and (type(sampling_seed) is not int or not 0 <= sampling_seed < 2**31):
            raise ValueError('sampling_seed must be null or an integer in 0..2147483647')
        self.endpoint = endpoint.rstrip("/")
        self.tokenizer = tokenizer
        self.model_adapter = model_adapter
        self.tokenizer_digest, self.template_digest = tokenizer_digest, template_digest
        self.timeout, self.max_context = timeout, max_context
        self.audited_raw_logprobs = audited_raw_logprobs
        self.sampling_seed = sampling_seed
        self.opener = request.build_opener(request.ProxyHandler({}))

    def prepare(self, converted):
        ids = self.model_adapter.render_chat(self.tokenizer, converted)
        if len(ids) + converted["sampling"]["max_new_tokens"] > self.max_context:
            raise ProtocolError("context budget exceeded; implicit truncation is forbidden")
        return ids

    def count_tokens(self, converted):
        return len(self.prepare(converted))

    def weight_version(self, endpoint=None):
        target = endpoint or self.endpoint
        req = request.Request(target + "/get_weight_version")
        with self.opener.open(req, timeout=self.timeout) as response:
            payload = json.load(response)
        version = payload.get("weight_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError("missing serving weight version")
        return version

    def pool_states(self):
        req = request.Request(self.endpoint + "/workers")
        with self.opener.open(req, timeout=self.timeout) as response:
            payload = json.load(response)
        workers = payload.get("workers") if isinstance(payload, dict) else None
        if not isinstance(workers, list) or not workers:
            raise ValueError("serving router returned no workers")
        states = []
        for worker in workers:
            endpoint = worker.get("url") if isinstance(worker, dict) else None
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ValueError("serving router returned an invalid worker URL")
            endpoint = endpoint.rstrip("/")
            states.append({
                "endpoint": endpoint,
                "healthy": True,
                "weight_version": self.weight_version(endpoint),
            })
        return states

    def generate(self, converted, identity, response_id):
        ids = self.prepare(converted)
        sampling = dict(converted['sampling'])
        if self.sampling_seed is not None:
            sampling['sampling_seed'] = int(digest({
                'seed': self.sampling_seed, 'batch': identity.batch_id,
                'group': identity.group_id, 'slot': identity.slot_id,
                'task': identity.task_digest, 'input_ids': ids,
            }), 16) % 2**31
        payload = {"input_ids": ids, "sampling_params": sampling,
                   "return_logprob": True, "stream": False}
        req = request.Request(self.endpoint + "/generate", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
        with self.opener.open(req, timeout=self.timeout) as response:
            output = json.load(response)
        return self.decode(output, payload, converted, identity, response_id)

    def decode(self, output, payload, converted, identity, response_id):
        meta = output.get("meta_info", {})
        if str(meta.get("weight_version", "")) != identity.policy_version:
            raise ValueError("missing or mismatched generation-time weight version")
        rows = meta.get("output_token_logprobs")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(r, (list, tuple)) or len(r) < 2 or not finite(r[0]) or r[0] > 0
            or type(r[1]) is not int or r[1] < 0 for r in rows
        ):
            raise ValueError("missing or malformed generation logprobs")
        text = output.get("text")
        if not isinstance(text, str):
            raise TypeError("missing generation text")
        output_ids = [row[1] for row in rows]
        reported_ids = output.get("output_ids")
        if reported_ids is not None and reported_ids != output_ids:
            raise ValueError("serving output token IDs disagree with logprob evidence")
        decoded = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        finish = meta.get("finish_reason", {})
        if not isinstance(finish, dict) or finish.get("type") not in ("stop", "length"):
            raise ValueError("unsupported serving finish reason")
        # SGLang may omit the explicitly matched terminal stop string from text.
        # Nothing else may be dropped, appended, or whitespace-normalized.
        matched = finish.get("matched")
        stop_sequence = matched if (finish["type"] == "stop" and isinstance(matched, str)
                                    and matched in converted["sampling"]["stop"]) else None
        if decoded != text and not (stop_sequence and decoded == text + stop_sequence):
            raise ValueError("serving text disagrees with output token IDs")
        blocks = self.model_adapter.parse_output(
            text, response_id=response_id, tools=converted["tools"]
        )
        has_tools = any(b["type"] == "tool_use" for b in blocks)
        declared = {tool['function']['name'] for tool in converted['tools']}
        unavailable = [{'id': b['id'], 'name': b['name']} for b in blocks
                       if b['type'] == 'tool_use' and b['name'] not in declared]
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        eos = finish["type"] == "stop" and finish.get("matched") == eos_id
        reason = "max_tokens" if finish["type"] == "length" else (
            "tool_use" if has_tools else "stop_sequence" if stop_sequence else "end_turn")
        return {"input_ids": payload["input_ids"], "output_ids": output_ids,
                "logprobs": [r[0] for r in rows], "logprob_semantics": "raw_model" if self.audited_raw_logprobs else "unverified",
                "engine_id": self.endpoint, "policy_version": str(meta["weight_version"]),
                "tokenizer_digest": self.tokenizer_digest, "template_digest": self.template_digest,
                "serving_input": payload, "evidence_kind": "serving", "content": blocks,
                "finish_reason": reason, "stop_sequence": stop_sequence, "eos": eos,
                "sampling": converted["sampling"],
                "model_output_text": text, "unavailable_tool_calls": unavailable,
                "serving_output_digest": digest(output)}
