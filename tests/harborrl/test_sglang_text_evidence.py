"""No token prefix/suffix can masquerade as the text delivered to the Agent."""
from types import SimpleNamespace

import pytest

from harborrl.gateway.sglang_backend import SGLangBackend


def decode(decoded, text, *, finish=None, stops=()):
    tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: decoded, eos_token_id=99)
    model = SimpleNamespace(parse_output=lambda value, **_kwargs: [{"type": "text", "text": value}])
    backend = SGLangBackend("http://unused", tokenizer, model,
                            tokenizer_digest="fixture", template_digest="fixture")
    return backend.decode(
        {"text": text, "meta_info": {"weight_version": "1", "output_token_logprobs": [[-.2, 13]],
                                      "finish_reason": finish or {"type": "stop", "matched": 99}}},
        {"input_ids": [1]}, {"tools": [], "sampling": {"stop": list(stops)}},
        SimpleNamespace(policy_version="1"), "response")


@pytest.mark.parametrize("decoded,text", [
    ("done hidden suffix", "done"),
    ("done", "done extra visible text"),
    ("done\n", "done"),
    ("done", "done\n"),
    ("hidden", ""),
    ("", "unbacked text"),
    ("   ", ""),
])
def test_rejects_prefixes_and_unproved_whitespace_changes(decoded, text):
    with pytest.raises(ValueError, match="text disagrees"):
        decode(decoded, text)


@pytest.mark.parametrize("text", ["done", "done\n", "\n done  ", "你好", ""])
def test_accepts_exact_visible_text(text):
    result = decode(text, text)
    assert result["content"] == [{"type": "text", "text": text}]
    assert result["output_ids"] == [13]


def test_accepts_only_the_explicit_matched_terminal_stop_omission():
    finish = {"type": "stop", "matched": "<END>"}
    for decoded in ("done", "done<END>"):
        result = decode(decoded, "done", finish=finish, stops=["<END>"])
        assert result["stop_sequence"] == "<END>"
        assert result["finish_reason"] == "stop_sequence"
    for decoded in ("done<END>hidden", "done<END><END>", "done\n<END>"):
        with pytest.raises(ValueError, match="text disagrees"):
            decode(decoded, "done", finish=finish, stops=["<END>"])
    with pytest.raises(ValueError, match="text disagrees"):
        decode("done<END>", "done", finish=finish)  # undeclared stop
    with pytest.raises(ValueError, match="text disagrees"):
        decode("done<END>", "done", finish={"type": "length", "matched": "<END>"}, stops=["<END>"])
