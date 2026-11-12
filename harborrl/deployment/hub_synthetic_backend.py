"""Synthetic serving backend for transport-only rollout verification.

This backend never touches a model: it exists so the hub -> worker -> gateway
-> runner path can be exercised end to end without a GPU serving pool. Its
output is marked synthetic, and trajectory assembly downgrades such evidence
to EVAL_ONLY, so it can never be mistaken for real RL evidence.
"""
from __future__ import annotations

from harborrl.gateway.messages import ProtocolError


class SyntheticServingBackend:
    """Minimal backend implementing the two calls Gateway makes."""

    def __init__(self, *, context=32768):
        self.max_context = context
        self.endpoint = "synthetic://local"

    def prepare(self, converted):
        ids = list(converted.get("input_ids") or [1, 2, 3])
        if len(ids) + converted["sampling"]["max_new_tokens"] > self.max_context:
            raise ProtocolError("context budget exceeded")
        return ids

    def count_tokens(self, converted):
        return len(self.prepare(converted))

    def weight_version(self, endpoint=None):
        return "synthetic"

    def pool_states(self):
        return [{"endpoint": self.endpoint, "healthy": True,
                 "weight_version": "synthetic"}]

    def generate(self, converted, identity, response_id):
        return {
            "input_ids": [1, 2],
            "output_ids": [3, 4],
            "logprobs": [-0.2, -0.3],
            "logprob_semantics": "raw_model",
            "policy_version": identity.policy_version,
            "tokenizer_digest": "synthetic-tokenizer",
            "template_digest": "synthetic-template",
            "engine_id": "synthetic",
            "serving_input": {"input_ids": [1, 2]},
            "sampling": converted["sampling"],
            "finish_reason": "end_turn",
            "evidence_kind": "synthetic",
            "content": [{"type": "text", "text": "synthetic rollout output"}],
        }
