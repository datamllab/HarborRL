"""Qwen tokenizer/template and generated tool-call protocol."""
from __future__ import annotations

from harborrl.gateway.messages import ProtocolError, parse_qwen_output


def normalize_chat_template_ids(rendered):
    """Accept both legacy lists and modern ``BatchEncoding`` results."""
    if hasattr(rendered, "get"):
        rendered = rendered.get("input_ids")
    if isinstance(rendered, (list, tuple)):
        if rendered and isinstance(rendered[0], (list, tuple)):
            rendered = rendered[0]
        return list(rendered)
    return rendered


class QwenModelAdapter:
    model_id = "qwen"

    def render_chat(self, tokenizer, converted):
        rendered_ids = tokenizer.apply_chat_template(
            converted["messages"],
            tools=converted["tools"] or None,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_ids = normalize_chat_template_ids(rendered_ids)
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(type(token_id) is not int or token_id < 0 for token_id in token_ids)
        ):
            raise ProtocolError("tokenizer did not produce actual input IDs")
        return token_ids

    def parse_output(self, text, *, response_id, tools):
        return parse_qwen_output(text, response_id=response_id, tools=tools, allow_unknown_tools=True)
