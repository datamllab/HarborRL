import json
import threading
from types import SimpleNamespace
from urllib import error, request

import pytest
from test_native_contracts import identity

from harborrl.gateway.messages import (
    ProtocolError,
    convert,
    parse_qwen_output,
    sse_events,
)
from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry
from harborrl.rollout.harbor_job.audit import audit_session
from harborrl.trajectories.native import digest


def payload(**changes):
    return {"model": "policy", "max_tokens": 64, "messages": [{"role": "user", "content": "hello"}], **changes}


class Backend:
    """Test-only synthetic generation; no model quality or GPU evidence."""
    def count_tokens(self, converted):
        return 2

    def generate(self, converted, ident, response_id):
        return {"input_ids": [1, 2], "output_ids": [3], "logprobs": [-.2], "logprob_semantics": "raw_model",
                "policy_version": ident.policy_version, "tokenizer_digest": "tok", "template_digest": "tpl",
                "engine_id": "synthetic", "serving_input": {"input_ids": [1, 2]}, "sampling": converted["sampling"],
                "finish_reason": "end_turn", "evidence_kind": "synthetic", "content": [{"type": "text", "text": "done"}]}


def setup(tmp_path):
    registry = TraceRegistry(tmp_path / "traces")
    ident = identity()
    token = registry.register(ident, "trial")
    return registry, ident, token, Gateway(registry, Backend(), model="policy")


@pytest.mark.parametrize("change", [{"model": "external"}, {"thinking": {"type": "enabled"}},
    {"temperature": .7}, {"top_p": .9}, {"top_k": 1}, {"max_tokens": True}, {"stream": 1},
    {"messages": []}, {"messages": [{"role": "assistant", "content": [{"type": "image"}]}]},
    {"tool_choice": {"type": "any"}}])
def test_unsupported_request(change):
    with pytest.raises(ProtocolError):
        convert(payload(**change), model="policy")


@pytest.mark.parametrize("temperature", [1, 1.0], ids=["int", "float"])
@pytest.mark.parametrize("top_p", [1, 1.0], ids=["int", "float"])
def test_equivalent_sampling_numbers_share_attempt_lock(tmp_path, temperature, top_p):
    registry, ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload(temperature=temperature, top_p=top_p))
    registry.delivery(aid, message["id"], "sent")
    registry.consume(aid, message["id"], message["content"], "session")
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert seal["errors"] == []
    assert digest(turns[0]["sampling"]) == ident.sampling_digest
    for key, value in (("temperature", temperature), ("top_p", top_p)):
        assert type(turns[0]["sampling"][key]) is float
        assert type(turns[0]["client_request"][key]) is type(value)


def test_tool_roundtrip():
    tools = [{"name": "bash", "input_schema": {"type": "object"}}]
    p = payload(tools=tools, messages=[
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "bash", "input": {"command": "false"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "exit 1", "is_error": True}]}])
    converted = convert(p, model="policy")
    assert converted["messages"][-1]["content"] == "[tool_error]\nexit 1"
    out = parse_qwen_output('<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>', response_id="m", tools=converted["tools"])
    assert out[0]["input"] == {"command": "pwd"}
    assert out[0]["id"] == "toolu_m_0"
    p["messages"][-1]["content"][0]["tool_use_id"] = "other"
    with pytest.raises(ProtocolError, match="orphan"):
        convert(p, model="policy")


@pytest.mark.parametrize("effort", ["high", "xhigh"])
def test_claude_2_1_control_turns_are_accepted_without_hidden_sampling(effort):
    converted = convert(payload(
        output_config={"effort": effort},
        thinking={"type": "adaptive"},
        context_management={"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        system=[{"type": "text", "text": "locked system prompt"}],
        messages=[
            {"role": "user", "content": "run"},
            {"role": "system", "content": [{"type": "text", "text": "session update",
                                            "cache_control": {"type": "ephemeral"}}]},
        ]), model="policy")
    assert converted["messages"][0] == {"role": "system", "content": "locked system prompt"}
    assert converted["messages"][1] == {"role": "user", "content": "run"}
    assert converted["messages"][2] == {"role": "system", "content": "session update"}
    assert converted["sampling"]["temperature"] == 1
    with pytest.raises(ProtocolError, match="thinking"):
        convert(payload(thinking={"type": "enabled", "budget_tokens": 1}), model="policy")
    with pytest.raises(ProtocolError, match="context management"):
        convert(payload(context_management={"edits": []}), model="policy")


@pytest.mark.parametrize("text", ['<tool_call>{broken}</tool_call>', '<tool_call>{}', '<think>x</think>', '<tool_call>{"name":"unknown","arguments":{}}</tool_call>'])
def test_model_output_never_repaired(text):
    with pytest.raises(ProtocolError):
        parse_qwen_output(text, response_id="m", tools=[])


def test_trace_requires_actual_consumption_and_is_immutable(tmp_path):
    registry, _ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    registry.close(aid)
    _turns, seal = registry.seal(aid)
    assert seal["errors"] == ["unconfirmed_consumption"]
    with pytest.raises(ValueError, match="sealed"):
        registry.consume(aid, message["id"], message["content"], "session")
    with pytest.raises(PermissionError):
        gateway.generate(token, payload())


def test_session_audit_covers_final_response(tmp_path):
    registry, _ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    session = tmp_path / "session.jsonl"
    session.write_text(json.dumps({"sessionId": "session", "type": "assistant", "message": message}) + "\n")
    audited = audit_session(session, registry, aid)
    assert audited["consumed_response_ids"] == [message["id"]]
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert seal["errors"] == []
    assert turns[0]["delivery"] == "consumed_confirmed"


def test_session_audit_merges_split_assistant_blocks(tmp_path):
    class MultiBlockBackend(Backend):
        def generate(self, converted, ident, response_id):
            turn = super().generate(converted, ident, response_id)
            turn["content"] = [*turn["content"], {"type": "text", "text": "tail"}]
            return turn

    registry, _ident, token, gateway = setup(tmp_path)
    gateway.backend = MultiBlockBackend()
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    assert len(message["content"]) == 2
    session = tmp_path / "session.jsonl"
    session.write_text("".join(
        json.dumps({"sessionId": "session", "type": "assistant", "message": part}) + "\n"
        for part in ({**message, "content": [block]} for block in message["content"])))
    audited = audit_session(session, registry, aid)
    assert audited["consumed_response_ids"] == [message["id"]]
    registry.close(aid)
    assert registry.seal(aid)[1]["errors"] == []


def test_session_audit_rejects_blocks_the_gateway_never_served(tmp_path):
    registry, _ident, token, gateway = setup(tmp_path)
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message["id"], "sent")
    forged = {**message, "content": [{"type": "text", "text": "different"}]}
    session = tmp_path / "session.jsonl"
    session.write_text("".join(
        json.dumps({"sessionId": "session", "type": "assistant", "message": part}) + "\n"
        for part in (message, forged)))
    with pytest.raises(ValueError, match="differs"):
        audit_session(session, registry, aid)


def test_unknown_auxiliary_request_poison_seal(tmp_path):
    registry, ident, token, gateway = setup(tmp_path)
    with pytest.raises(ProtocolError):
        gateway.generate(token, payload(model="haiku"))
    registry.close(ident.attempt_id)
    assert registry.seal(ident.attempt_id)[1]["errors"] == ["ProtocolError"]


def test_pending_generation_blocks_version_change(tmp_path):
    registry, _ident, token, _gateway = setup(tmp_path)
    aid, rid, _, _ = registry.begin(token)
    with pytest.raises(TimeoutError):
        registry.drain(.01)
    with pytest.raises(ValueError):
        registry.publish_version([{"healthy": True, "weight_version": "2"}], "2")
    registry.finish(aid, rid, error="cancelled")
    registry.seal(aid)
    registry.publish_version([{"healthy": True, "weight_version": "2", "endpoint": "a"},
                              {"healthy": True, "weight_version": "2", "endpoint": "b"}], "2")
    with pytest.raises(ValueError, match="version"):
        registry.register(identity("1"), "trial2")


def test_pool_mismatch_keeps_admission_closed(tmp_path):
    registry = TraceRegistry(tmp_path)
    registry.drain(.1)
    with pytest.raises(RuntimeError):
        registry.publish_version([{"healthy": True, "weight_version": "1"}, {"healthy": True, "weight_version": "2"}], "2")
    assert not registry.accepting


def test_http_overload_rejects_before_any_attempt_state(tmp_path):
    _registry, _ident, token, gateway = setup(tmp_path)
    started, release = threading.Event(), threading.Event()

    class BlockingBackend(Backend):
        def generate(self, converted, ident, response_id):
            started.set()
            release.wait(5)
            return super().generate(converted, ident, response_id)

    gateway.backend = BlockingBackend()
    try:
        server = make_server(gateway, max_inflight=1)
    except PermissionError:
        pytest.skip("this CPU sandbox does not permit loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/messages"
    opener = request.build_opener(request.ProxyHandler({}))
    results, errors = [], []

    def post():
        try:
            with opener.open(request.Request(url, json.dumps(payload()).encode(),
                                             {"x-api-key": token}), timeout=10) as response:
                results.append(json.load(response))
        except Exception as exc:  # noqa: BLE001 - test collector
            errors.append(exc)

    try:
        first = threading.Thread(target=post)
        first.start()
        assert started.wait(5), "first request never reached the backend"
        with pytest.raises(error.HTTPError) as exc:
            opener.open(request.Request(url, json.dumps(payload()).encode(),
                                        {"x-api-key": token}), timeout=5)
        assert exc.value.code == 503
        assert exc.value.read()  # overloaded_error body is present
        release.set()
        first.join(10)
        assert not errors and results and results[0]["role"] == "assistant"
        # The rejected request never began a generation, so a retry is safe.
        post()
        assert len(results) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_http_sse_and_auth(tmp_path):
    _registry, _ident, token, gateway = setup(tmp_path)
    try:
        server = make_server(gateway)
    except PermissionError:
        pytest.skip("this CPU sandbox does not permit loopback socket binding")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1/messages"
    ready_url = f"http://127.0.0.1:{server.server_port}/readyz"
    opener = request.build_opener(request.ProxyHandler({}))
    try:
        with opener.open(ready_url, timeout=5) as response:
            assert json.load(response) == {"ok": True, "model": "policy"}
        req = request.Request(url, json.dumps(payload(stream=True)).encode(), {"x-api-key": token})
        with opener.open(req, timeout=5) as response:
            events = [json.loads(line[6:]) for line in response.read().decode().splitlines() if line.startswith("data: ")]
        assert events[0]["type"] == "message_start"
        assert events[-1]["type"] == "message_stop"
        assert events[2]["delta"]["text"] == "done"
        with pytest.raises(error.HTTPError) as exc:
            opener.open(request.Request(url, b'{}', {"x-api-key": "bad"}), timeout=5)
        assert exc.value.code == 401
        query = request.Request(url + "?beta=true", json.dumps(payload()).encode(),
                                {"x-api-key": token})
        with opener.open(query, timeout=5) as response:
            assert json.load(response)["role"] == "assistant"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_sse_tool_parameters_are_json_delta():
    message = {"id": "m", "role": "assistant", "type": "message", "model": "p", "content": [
        {"type": "tool_use", "id": "t", "name": "bash", "input": {"command": "pwd"}}],
        "stop_reason": "tool_use", "stop_sequence": None, "usage": {"input_tokens": 2, "output_tokens": 3}}
    events = list(sse_events(message))
    assert json.loads(events[2][1]["delta"]["partial_json"]) == {"command": "pwd"}
    assert events[-2][1]["delta"]["stop_reason"] == "tool_use"


@pytest.mark.parametrize("tools", [None, {}, "tool"])
def test_invalid_tool_list_is_protocol_error(tools):
    with pytest.raises(ProtocolError):
        convert(payload(tools=tools), model="policy")


def test_nonfinite_generated_tool_is_rejected():
    with pytest.raises(ProtocolError):
        parse_qwen_output('<tool_call>{"name":"bash","arguments":{"x":NaN}}</tool_call>',
                          response_id="m", tools=[{"function": {"name": "bash"}}])


def test_generated_tool_errors_distinguish_name_and_arguments():
    tools = [{"function": {"name": "bash"}}]
    with pytest.raises(ProtocolError, match="unknown generated tool"):
        parse_qwen_output('<tool_call>{"name":"sh","arguments":{}}</tool_call>',
                          response_id="m", tools=tools)
    with pytest.raises(ProtocolError, match="arguments are not an object"):
        parse_qwen_output('<tool_call>{"name":"bash","arguments":[]}</tool_call>',
                          response_id="m", tools=tools)


def test_generated_tool_name_alias_is_canonicalized_only_when_unique():
    tools = [{"function": {"name": "Bash"}}]
    output = parse_qwen_output('<tool_call>{"name":"bash","arguments":{"command":"pwd"}}</tool_call>',
                               response_id="m", tools=tools)
    assert output[0]["name"] == "Bash"
    with pytest.raises(ProtocolError, match="unknown generated tool"):
        parse_qwen_output('<tool_call>{"name":"sh","arguments":{}}</tool_call>',
                          response_id="m", tools=tools)


def test_qwen_finish_marker_maps_to_empty_end_turn():
    output = parse_qwen_output('<tool_call>{"name":"finish","arguments":{}}</tool_call>',
                               response_id="m", tools=[{"function": {"name": "Bash"}}])
    assert output == []


def test_model_adapter_preserves_unavailable_action_but_history_requires_native_error():
    from harborrl.gateway.messages import unavailable_tool_error
    from harborrl.models.qwen import QwenModelAdapter

    tools = [{'name': 'Bash', 'input_schema': {'type': 'object'}}]
    converted = convert(payload(tools=tools), model='policy')
    text = '<tool_call>{"name":"Check","arguments":{"target":"solution.py"}}</tool_call>'
    blocks = QwenModelAdapter().parse_output(text, response_id='r', tools=converted['tools'])
    assert blocks == [{'type': 'tool_use', 'id': 'toolu_r_0', 'name': 'Check',
                       'input': {'target': 'solution.py'}}]
    history = [*payload()['messages'], {'role': 'assistant', 'content': blocks},
               {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': blocks[0]['id'],
                                           'content': unavailable_tool_error('Check'), 'is_error': True}]}]
    result = convert(payload(tools=tools, messages=history), model='policy')
    assert result['messages'][-1]['content'] == '[tool_error]\n' + unavailable_tool_error('Check')
    history[-1]['content'][0]['is_error'] = False
    with pytest.raises(ProtocolError, match='exact native rejection'):
        convert(payload(tools=tools, messages=history), model='policy')
    history[-1]['content'][0].update(is_error=True, content='command executed')
    with pytest.raises(ProtocolError, match='exact native rejection'):
        convert(payload(tools=tools, messages=history), model='policy')


@pytest.mark.parametrize('rejection', ['valid', 'absent', 'success', 'wrong_tool', 'duplicate'])
def test_unavailable_action_requires_audited_native_rejection(tmp_path, rejection):
    from harborrl.gateway.messages import unavailable_tool_error

    class UnavailableBackend(Backend):
        def generate(self, converted, ident, response_id):
            turn = super().generate(converted, ident, response_id)
            turn.update(finish_reason='tool_use', content=[{'type': 'tool_use', 'id': 'u1',
                                                         'name': 'Check', 'input': {}}])
            return turn

    registry, _ident, token, gateway = setup(tmp_path)
    gateway.backend = UnavailableBackend()
    aid, message = gateway.generate(token, payload())
    registry.delivery(aid, message['id'], 'sent')
    result = {'type': 'tool_result', 'tool_use_id': 'u1', 'content': unavailable_tool_error('Check'),
              'is_error': True}
    if rejection == 'success':
        result['is_error'] = False
    if rejection == 'wrong_tool':
        result['content'] = unavailable_tool_error('Other')
    results = [] if rejection == 'absent' else [result] * (2 if rejection == 'duplicate' else 1)
    path = tmp_path / 'session.jsonl'
    path.write_text('\n'.join(map(json.dumps, [
        {'type': 'assistant', 'sessionId': 's', 'message': message},
        {'type': 'user', 'sessionId': 's', 'message': {'role': 'user', 'content': results}}])))
    if rejection == 'valid':
        audit_session(path, registry, aid)
        registry.close(aid)
        turns, seal = registry.seal(aid)
        assert seal['errors'] == []
        assert turns[0]['tool_rejection_ref'].startswith(str(path) + '#sha256=')
        assert turns[0]['output_ids'] == [3] and turns[0]['logprobs'] == [-.2]
    else:
        with pytest.raises(ValueError, match='exact native rejection'):
            audit_session(path, registry, aid)
        registry.close(aid)
        assert 'unconfirmed_tool_rejection' in registry.seal(aid)[1]['errors']


def test_sglang_generation_version_and_logprob_evidence():
    from harborrl.gateway.sglang_backend import SGLangBackend
    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return [11, 12]
        def decode(self, token_ids, skip_special_tokens):
            return "done"
    from harborrl.models.qwen import QwenModelAdapter
    backend = SGLangBackend(
        "http://engine", Tokenizer(), QwenModelAdapter(),
        tokenizer_digest="tok", template_digest="tpl",
    )
    converted = convert(payload(), model="policy")
    serving = {"input_ids": backend.prepare(converted)}
    output = {"text": "done", "meta_info": {"weight_version": "1", "output_token_logprobs": [[-.2, 13]],
                                               "finish_reason": {"type": "stop"}}}
    decoded = backend.decode(output, serving, converted, identity(), "msg")
    assert decoded["output_ids"] == [13]
    assert decoded["logprob_semantics"] == "unverified"
    output["meta_info"]["weight_version"] = "2"
    with pytest.raises(ValueError, match="version"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["meta_info"]["weight_version"] = "1"
    output["meta_info"]["output_token_logprobs"] = [[None, 13]]
    with pytest.raises(ValueError, match="logprobs"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["meta_info"]["output_token_logprobs"] = [[-.2, 13]]
    output["output_ids"] = [14]
    with pytest.raises(ValueError, match="token IDs"):
        backend.decode(output, serving, converted, identity(), "msg")
    output["output_ids"] = [13]
    output["text"] = "different"
    with pytest.raises(ValueError, match="text"):
        backend.decode(output, serving, converted, identity(), "msg")
    backend.max_context = 10
    with pytest.raises(ProtocolError, match="context"):
        backend.prepare(converted)


def test_sglang_weight_version_is_explicit():
    from harborrl.gateway.sglang_backend import SGLangBackend

    class Response:
        def read(self):
            return json.dumps({"weight_version": "7"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    class Opener:
        def open(self, request, timeout):
            assert request.full_url == "http://engine/get_weight_version"
            assert timeout == 30
            return Response()

    from harborrl.models.qwen import QwenModelAdapter
    backend = SGLangBackend(
        "http://engine", SimpleNamespace(), QwenModelAdapter(),
        tokenizer_digest="tok", template_digest="tpl", timeout=30,
    )
    backend.opener = Opener()
    assert backend.weight_version() == "7"


def test_sglang_unavailable_tool_keeps_sampled_tokens_logprobs_and_text():
    from harborrl.gateway.sglang_backend import SGLangBackend
    from harborrl.models.qwen import QwenModelAdapter

    text = '<tool_call>{"name":"Check","arguments":{}}</tool_call>'
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: text)
    backend = SGLangBackend('http://engine', tokenizer, QwenModelAdapter(),
                            tokenizer_digest='tok', template_digest='tpl', audited_raw_logprobs=True)
    converted = convert(payload(), model='policy')
    output = {'text': text, 'output_ids': [13, 14], 'meta_info': {
        'weight_version': '1', 'output_token_logprobs': [[-.2, 13], [-.3, 14]],
        'finish_reason': {'type': 'stop'}}}
    result = backend.decode(output, {'input_ids': [11, 12]}, converted, identity(), 'msg')
    assert result['model_output_text'] == text
    assert result['output_ids'] == [13, 14] and result['logprobs'] == [-.2, -.3]
    assert result['finish_reason'] == 'tool_use'
    assert result['unavailable_tool_calls'] == [{'id': 'toolu_msg_0', 'name': 'Check'}]
    assert result['logprob_semantics'] == 'raw_model'


def test_sglang_request_seed_is_independent_of_dispatch_order_and_retry():
    from harborrl.gateway.sglang_backend import SGLangBackend
    from harborrl.models.qwen import QwenModelAdapter

    sent = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{}'

    class Opener:
        def open(self, request, timeout):
            sent.append(json.loads(request.data))
            return Response()

    backend = SGLangBackend('http://engine', SimpleNamespace(), QwenModelAdapter(),
                            tokenizer_digest='tok', template_digest='tpl', sampling_seed=0)
    backend.opener = Opener()
    backend.prepare = lambda converted: converted['input_ids']
    backend.decode = lambda output, serving, *args: serving
    converted = {'input_ids': [11, 12], 'sampling': {'temperature': 1.0}}
    first = backend.generate(converted, identity(), 'response-a')
    backend.generate(converted, identity('1'), 'response-b')
    retry = backend.generate(converted, identity(attempt_id='retry', run_id='other'), 'response-c')
    assert first == retry
    assert first['sampling_params']['sampling_seed'] != sent[1]['sampling_params']['sampling_seed']
    assert converted['sampling'] == {'temperature': 1.0}
    for changed in (identity(batch_id='next'), identity(group_id='next'), identity(task_digest='next')):
        assert backend.generate(converted, changed, 'response')['sampling_params'] != first['sampling_params']
    assert backend.generate({**converted, 'input_ids': [11, 13]}, identity(), 'response') != first
    backend.sampling_seed = 1
    assert backend.generate(converted, identity(), 'response') != first
    backend.sampling_seed = None
    assert 'sampling_seed' not in backend.generate(converted, identity(), 'response')['sampling_params']


@pytest.mark.parametrize('seed', [True, -1, 2**31, 1.5])
def test_sglang_rejects_invalid_sampling_seed(seed):
    from harborrl.gateway.sglang_backend import SGLangBackend
    with pytest.raises(ValueError, match='sampling_seed'):
        SGLangBackend('http://engine', None, None, tokenizer_digest='tok',
                      template_digest='tpl', sampling_seed=seed)


def test_native_gateway_thread_does_not_own_asyncio_default_executor():
    from types import SimpleNamespace

    from harborrl.rollout.runtime import GatewayShard, GatewayShardSpec

    started, stopped = threading.Event(), threading.Event()

    class Server:
        def serve_forever(self):
            started.set()
            stopped.wait(5)

    shard = GatewayShard(GatewayShardSpec("gw0", 49184, "http://127.0.0.1:49184"),
                         registry=SimpleNamespace(), gateway=SimpleNamespace(),
                         server=Server(), trace_dir="unused")
    shard.start()
    try:
        assert started.wait(1)
        assert shard.thread.daemon is True
    finally:
        stopped.set()
        shard.thread.join(1)
