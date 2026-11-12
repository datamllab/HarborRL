"""Keep the native training example compatible with the CLI prompt budget."""
from types import SimpleNamespace

import pytest
import yaml

from harborrl.config import ROOT
from harborrl.gateway.messages import ProtocolError
from harborrl.gateway.sglang_backend import SGLangBackend


def test_native_example_reserves_room_for_cli_system_and_tools():
    config = yaml.safe_load((ROOT / "examples/native/train_qwen_native.yaml").read_text())
    sampling = config["sampling"]
    # A controlled CPU rendering floor based on the pinned CLI's measured prompt.
    # This checks the example budget, not tokenizer/model/GPU compatibility.
    model = SimpleNamespace(render_chat=lambda *_: [1] * 20000)
    backend = SGLangBackend("http://unused", None, model,
                            tokenizer_digest="fixture", template_digest="fixture",
                            max_context=sampling["max_context"])
    converted = {"sampling": {"max_new_tokens": sampling["max_tokens"]}}
    assert len(backend.prepare(converted)) == 20000
    backend.max_context = 8192
    with pytest.raises(ProtocolError, match="context budget exceeded"):
        backend.prepare(converted)
