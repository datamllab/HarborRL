"""Rejected model outputs are evidence, never retry or training inputs."""
import errno
import json
from copy import deepcopy

import pytest
from test_native_gateway import payload, setup

from harborrl.gateway.messages import ProtocolError
from harborrl.gateway.sglang_backend import SGLangBackend
from harborrl.models.qwen import QwenModelAdapter
from harborrl.trajectories.native import digest, read_json


def backend_for(root, output):
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [11, 12]

        def decode(self, ids, skip_special_tokens):
            return output['text']

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(output).encode()

    class Opener:
        calls = 0

        def open(self, request, timeout):
            self.calls += 1
            return Response()

    backend = SGLangBackend('http://engine', Tokenizer(), QwenModelAdapter(),
        tokenizer_digest='tok', template_digest='tpl', rejected_root=root)
    backend.opener = Opener()
    return backend


def serving_output(text):
    return {'text': text, 'meta_info': {'weight_version': '1',
        'output_token_logprobs': [[-.2, 13]], 'finish_reason': {'type': 'stop'}}}


@pytest.mark.parametrize('fault', ['json', 'incomplete', 'version', 'logprobs', 'finish'])
def test_rejected_output_is_preserved_but_not_delivered_or_retryable(tmp_path, fault):
    output = serving_output('done')
    if fault == 'json':
        output['text'] = '<tool_call>{"name":"Edit","arguments":oops}</tool_call>'
    elif fault == 'incomplete':
        output['text'] = '<tool_call>{"name":'
    elif fault == 'version':
        output['meta_info']['weight_version'] = 'wrong'
    elif fault == 'logprobs':
        output['meta_info']['output_token_logprobs'] = [[None, 13]]
    else:
        output['meta_info']['finish_reason'] = {'type': 'abort'}
    original = deepcopy(output)
    registry, ident, token, gateway = setup(tmp_path)
    rejected = tmp_path / 'rejected'
    backend = gateway.backend = backend_for(rejected, output)
    with pytest.raises(ValueError):
        gateway.generate(token, payload())
    assert backend.opener.calls == 1 and output == original
    paths = list(rejected.glob('*/*.json'))
    assert len(paths) == 1
    record = read_json(paths[0])
    assert record['identity'] == ident.to_dict()
    assert record['serving_output'] == original
    assert record['serving_output_digest'] == digest(original)
    assert record['serving_input']['input_ids'] == [11, 12]
    assert record['converted_request']['messages'] == [{'role': 'user', 'content': 'hello'}]
    assert record['training_use'] is False and token not in paths[0].read_text()
    assert paths[0].parent.name == digest(ident.to_dict())
    assert paths[0].stem == digest(record['response_id'])
    registry.close(ident.attempt_id)
    turns, seal = registry.seal(ident.attempt_id)
    assert turns == [] and seal['errors']
    assert registry.has_nontransport_failure(ident.attempt_id)
    assert not registry.transport_failure_only(ident.attempt_id)


def test_valid_output_creates_no_rejected_evidence(tmp_path):
    _registry, ident, token, gateway = setup(tmp_path)
    rejected = tmp_path / 'rejected'
    gateway.backend = backend_for(rejected, serving_output('done'))
    aid, message = gateway.generate(token, payload())
    assert aid == ident.attempt_id and message['content'] == [{'type': 'text', 'text': 'done'}]
    assert not rejected.exists()


@pytest.mark.parametrize('error', [OSError(errno.ECONNRESET, 'write failed'),
                                  OSError(errno.ENOSPC, 'full'), ValueError('conflict')])
def test_diagnostic_write_failure_cannot_authorize_transport_retry(tmp_path, monkeypatch, error):
    from harborrl.gateway import sglang_backend

    registry, ident, token, gateway = setup(tmp_path)
    gateway.backend = backend_for(tmp_path / 'rejected', serving_output('<tool_call>bad</tool_call>'))
    def fail(*args):
        raise error
    monkeypatch.setattr(sglang_backend, 'publish', fail)
    with pytest.raises(ValueError, match='could not be persisted'):
        gateway.generate(token, payload())
    registry.close(ident.attempt_id)
    turns, seal = registry.seal(ident.attempt_id)
    assert turns == [] and seal['errors'] == ['ValueError']
    assert not registry.transport_failure_only(ident.attempt_id)
    assert registry.has_nontransport_failure(ident.attempt_id)


def test_record_without_sink_keeps_protocol_rejection(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    gateway.backend = backend_for(None, serving_output('<tool_call>bad</tool_call>'))
    with pytest.raises(ProtocolError, match='invalid generated tool JSON'):
        gateway.generate(token, payload())
    registry.close(ident.attempt_id)
    assert registry.seal(ident.attempt_id)[1]['errors'] == ['ProtocolError']
