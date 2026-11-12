"""Regression coverage for the full coordinator success path and empty finals."""
import asyncio
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_native_contracts import identity, make_ir
from test_native_lifecycle import PROFILE

from harborrl.gateway.trace import TraceRegistry
from harborrl.harnesses.claude_code import ClaudeCodeHarnessAdapter
from harborrl.rollout.harbor_job.coordinator import run_attempt
from harborrl.trajectories.native import RewardProfile, digest, require_ready


def test_casefold_tool_alias_must_be_unique():
    from harborrl.gateway.messages import ProtocolError, parse_qwen_output
    with pytest.raises(ProtocolError, match="ambiguous"):
        parse_qwen_output('<tool_call>{"name":"BASH","arguments":{}}</tool_call>',
                          response_id="r", tools=[{"function": {"name": "Bash"}},
                                                  {"function": {"name": "bash"}}])


@pytest.mark.parametrize("empty,budget,reward,continuations", [
    (False, False, 1, 0), (True, False, 1, 0), (False, True, 0, 0), (False, True, 1, 0),
    (False, True, 0, 1), (False, True, 1, 1), (False, True, 0, 2),
])
def test_coordinator_publishes_complete_success(tmp_path, monkeypatch, empty, budget, reward, continuations):
    ident = identity()
    trial_id = "trial-" + ident.attempt_id
    root = tmp_path / "attempt"
    trial = root / "runner/harbor/trial"
    template = make_ir(tmp_path / "fixture", nturns=2 + continuations)
    profile = {**PROFILE, 'cli_version': '2.1.141', 'max_turns': 2} if budget else PROFILE

    class Registry(TraceRegistry):
        def register(self, ident, trial_id, **kwargs):
            token = super().register(ident, trial_id, **kwargs)
            turns = template["turns"]
            messages = [{'role': 'user', 'content': 'Read the file.'}]
            for index, original in enumerate(turns):
                if budget:
                    original['client_request']['messages'] = deepcopy(messages)
                    original['finish_reason'] = 'max_tokens' if index < continuations else 'tool_use'
                    original['client_request']['tools'] = [{'name': 'Read', 'input_schema': {'type': 'object'}}]
                    original['client_request_digest'] = digest(original['client_request'])
                    original['response'] = {'stop_reason': original['finish_reason'],
                        'content': [{'type': 'text', 'text': 'Inspecting'}] if index < continuations else
                            [{'type': 'tool_use', 'id': f'tool-{index}',
                              'name': 'Read', 'input': {'file_path': '/workspace/code.py'}}]}
                    original['response_digest'] = digest(original['response'])
                    messages.append({'role': 'assistant', 'content': original['response']['content']})
                    messages.append({'role': 'user', 'content': [{'type': 'text',
                        'text': ('Output token limit hit. Resume directly — no apology, no recap of what you were doing. '
                                 'Pick up mid-thought if that is where the cut happened. '
                                 'Break remaining work into smaller pieces.')}]})
                aid, request, _, _ = self.begin(token)
                turn = {**original, "request_id": request, "delivery": "sent"}
                if empty and index == 1:
                    turn["response"] = {"content": []}
                    turn["response_digest"] = digest(turn["response"])
                self.finish(aid, request, turn)
                self.delivery(aid, turn["response_id"], "sent")
            session = trial / "agent/sessions/projects/session.jsonl"
            session.parent.mkdir(parents=True)
            session.write_text("\n".join(json.dumps({
                "type": "assistant", "sessionId": "session", "message": {
                    "id": turn["response_id"], "role": "assistant",
                    "content": turn["response"]["content"]}})
                for turn in (turns[:1] if empty else turns)))
            final = {
                "type": "result", "subtype": "success", "is_error": False,
                "result": "", "stop_reason": "end_turn", "session_id": "session",
                "terminal_reason": "completed"}
            events = []
            if budget:
                final.update(subtype='error_max_turns', is_error=True, stop_reason='tool_use',
                             terminal_reason='max_turns', num_turns=3, permission_denials=[],
                             errors=['Reached maximum number of turns (2)'])
                events = [{'type': 'system', 'subtype': 'init', 'session_id': 'session',
                           'claude_code_version': '2.1.141'}]
                events += [{'type': 'assistant', 'session_id': 'session',
                            'message': {'id': turn['response_id']}} for turn in turns]
            (trial / "agent/claude-code.txt").write_text('\n'.join(map(json.dumps, [*events, final])))
            (trial / "verifier").mkdir()
            (trial / "verifier/reward.json").write_text(json.dumps({'reward': reward}))
            (trial / "result.json").write_text(json.dumps({
                "id": trial_id, "finished_at": "2026-10-19T00:00:00Z",
                "verifier_environment_mode": "shared",
                "verifier_result": {"rewards": {"reward": reward}},
                **({'exception_info': {'exception_type': 'NonZeroAgentExitCodeError'}} if budget else {})}))
            return token

    worker = tmp_path / "worker.py"
    terminal = {"type": "terminal", "trial_id": trial_id,
                "trial_dir": str(trial), "result_path": str(trial / "result.json")}
    worker.write_text("import json, sys\njson.loads(sys.stdin.readline())\n"
                      f"print({json.dumps({'type': 'prepared', 'trial_id': trial_id})!r}, flush=True)\n"
                      "sys.stdin.readline()\n"
                      f"print({json.dumps(terminal)!r}, flush=True)\n")
    spawn = asyncio.create_subprocess_exec
    practice_path = str(tmp_path / 'deployment-practice')
    monkeypatch.setenv('PYTHONPATH', practice_path)

    async def fake_spawn(*args, **kwargs):
        expected = str(Path(__file__).resolve().parents[2])
        assert kwargs['env']['PYTHONPATH'].split(os.pathsep) == [expected, practice_path]
        return await spawn(sys.executable, "-S", str(worker), **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_spawn)
    ir = asyncio.run(run_attempt(
        ident, task_path=tmp_path, profile=profile, gateway_url="http://gateway",
        runner_python=sys.executable, root=root, registry=Registry(tmp_path / "trace"),
        reward_profile=RewardProfile()))
    assert ir["readiness"] == "CONTRACT_READY"
    assert json.loads((root / "trajectory.v2.json").read_text()) == ir
    assert (root / "evaluation.json").is_file()
    assert not (root / "cleanup-required.json").exists()
    assert all(turn["delivery"] == "consumed_confirmed" for turn in ir["turns"])
    assert ir['evaluation']['raw_reward'] == reward
    assert ir['termination'] == ('budget_truncated' if budget else 'completed')
    if budget:
        assert ir['termination_evidence'] == ir['evaluation']['termination_evidence']
        assert ir['termination_evidence']['max_turns'] == 2
        assert ir['termination_evidence']['policy_response_count'] == 2 + continuations
        assert len(ir['termination_evidence']['continuations']) == continuations
        assert [turn['output_ids'] for turn in ir['turns']] == [[3, 4]] * (2 + continuations)
        assert [turn['logprobs'] for turn in ir['turns']] == [[-.2, -.3]] * (2 + continuations)
        for mutation in ('missing_receipt', 'result_digest', 'consumed', 'completed'):
            changed = deepcopy(ir)
            if mutation == 'missing_receipt':
                changed['evaluation'].pop('termination_evidence')
            elif mutation == 'result_digest':
                changed['termination_evidence']['source_result_digest'] = 'different'
            elif mutation == 'consumed':
                changed['termination_evidence']['consumed_response_ids'] = []
            else:
                changed['termination'] = 'completed'
            with pytest.raises(ValueError, match='termination_evidence|truncation_evidence'):
                require_ready(changed, allow_synthetic=True)


def test_coordinator_reports_harbor_failure_before_session_creation(tmp_path, monkeypatch):
    ident = identity()
    trial_id = 'trial-' + ident.attempt_id
    root = tmp_path / 'attempt'
    trial = root / 'runner/harbor/trial'
    result_path = trial / 'result.json'
    terminal = {'type': 'terminal', 'trial_id': trial_id, 'trial_dir': str(trial),
                'result_path': str(result_path)}
    worker = tmp_path / 'worker.py'
    worker.write_text('import sys\nsys.stdin.readline()\n'
        f"print({json.dumps({'type': 'prepared', 'trial_id': trial_id})!r}, flush=True)\n"
        'sys.stdin.readline()\n'
        f'print({json.dumps(terminal)!r}, flush=True)\n')
    spawn = asyncio.create_subprocess_exec

    async def fake_spawn(*args, **kwargs):
        return await spawn(sys.executable, '-S', str(worker), **kwargs)

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', fake_spawn)
    registry = TraceRegistry(tmp_path / 'trace')
    original_register = registry.register

    def register(*args, **kwargs):
        trial.mkdir(parents=True)
        result_path.write_text(json.dumps({'exception_info': {'exception_type': 'EnvironmentStartTimeoutError'}}))
        return original_register(*args, **kwargs)

    monkeypatch.setattr(registry, 'register', register)
    with pytest.raises(RuntimeError, match='before session audit: EnvironmentStartTimeoutError'):
        asyncio.run(run_attempt(ident, task_path=tmp_path, profile=PROFILE, gateway_url='http://gateway',
            runner_python=sys.executable, root=root, registry=registry, reward_profile=RewardProfile()))
    assert not (root / 'cleanup-required.json').exists()
    assert not (root / 'trajectory.v2.json').exists()


@pytest.mark.parametrize("change", [
    {"content": [{"type": "text", "text": "missing"}]},
    {"finish_reason": "length"}, {"is_final": False},
])
def test_empty_final_rejects_other_missing_responses(tmp_path, change):
    class Registry:
        def unconsumed_responses(self, aid):
            return [{"response_id": "r", "content": [], "finish_reason": "end_turn",
                     "is_final": True, **change}]
    with pytest.raises(ValueError, match="only one empty final"):
        ClaudeCodeHarnessAdapter().audit_empty_final(tmp_path / "absent", Registry(), "a", {})


@pytest.mark.parametrize("change", [{"session_id": "wrong"}, {"is_error": True},
    {"subtype": "error_max_turns"}, {"result": "nonempty"}, {"stop_reason": "max_tokens"}])
def test_empty_final_requires_cli_consumption_evidence(tmp_path, change):
    class Registry:
        def unconsumed_responses(self, aid):
            return [{"response_id": "r", "content": [], "finish_reason": "end_turn", "is_final": True}]
    path = tmp_path / "claude-code.txt"
    path.write_text(json.dumps({"type": "result", "subtype": "success", "is_error": False,
        "result": "", "stop_reason": "end_turn", "session_id": "session",
        "terminal_reason": "completed", **change}))
    with pytest.raises(ValueError, match="matching session result"):
        ClaudeCodeHarnessAdapter().audit_empty_final(path, Registry(), "a", {"session_id": "session"})


@pytest.mark.parametrize('mutation', [
    'cli_version', 'init_version', 'init_session', 'missing_init', 'duplicate_result',
    'not_final', 'session', 'reason', 'subtype', 'errors', 'turn_count', 'consumed',
    'assistant_session', 'exception', 'permissions', 'is_error',
])
def test_budget_termination_rejects_unverified_evidence(tmp_path, mutation):
    from harborrl.rollout.harbor_job.audit import audit_termination

    profile = {**PROFILE, 'cli_version': '2.1.141', 'max_turns': 2}
    audit = {'session_id': 'session', 'consumed_response_ids': ['r0', 'r1'], 'evidence_ref': 'session#sha'}
    initial = {'type': 'system', 'subtype': 'init', 'session_id': 'session', 'claude_code_version': '2.1.141'}
    responses = [{'type': 'assistant', 'session_id': 'session', 'message': {'id': rid}}
                 for rid in audit['consumed_response_ids']]
    final = {'type': 'result', 'subtype': 'error_max_turns', 'is_error': True,
             'terminal_reason': 'max_turns', 'stop_reason': 'tool_use', 'num_turns': 3,
             'session_id': 'session', 'permission_denials': [],
             'errors': ['Reached maximum number of turns (2)']}
    result = {'exception_info': {'exception_type': 'NonZeroAgentExitCodeError'}}
    events = [initial, *responses, final]
    if mutation == 'cli_version':
        profile['cli_version'] = '2.1.0'
    elif mutation == 'init_version':
        initial['claude_code_version'] = '2.1.0'
    elif mutation == 'init_session':
        initial['session_id'] = 'other'
    elif mutation == 'missing_init':
        events.remove(initial)
    elif mutation == 'duplicate_result':
        events.append(final)
    elif mutation == 'not_final':
        events.append({'type': 'late_event'})
    elif mutation == 'session':
        final['session_id'] = 'other'
    elif mutation == 'reason':
        final['terminal_reason'] = 'timeout'
    elif mutation == 'subtype':
        final['subtype'] = 'error_during_execution'
    elif mutation == 'errors':
        final['errors'].append('connection failure')
    elif mutation == 'turn_count':
        final['num_turns'] = 2
    elif mutation == 'consumed':
        audit['consumed_response_ids'] = ['r0']
    elif mutation == 'assistant_session':
        responses[-1]['session_id'] = 'other'
    elif mutation == 'exception':
        result['exception_info']['exception_type'] = 'AgentTimeoutError'
    elif mutation == 'permissions':
        final['permission_denials'] = ['Bash']
    else:
        final['is_error'] = 1
    path, result_path = tmp_path / 'stream.jsonl', tmp_path / 'result.json'
    path.write_text('\n'.join(map(json.dumps, events)))
    result_path.write_text(json.dumps(result))
    with pytest.raises(ValueError):
        audit_termination(path, result_path, profile, audit, turns=[
            {'response_id': 'r0', 'finish_reason': 'tool_use'},
            {'response_id': 'r1', 'finish_reason': 'tool_use'}])


def test_collector_binds_budget_evidence_to_original_result(tmp_path):
    from test_native_contracts import receipt

    from harborrl.rollout.harbor_job.collector import collect

    ident = identity()
    receipt(tmp_path, ident, 0)
    root = tmp_path / ident.attempt_id
    result = json.loads((root / 'result.json').read_text())
    result['exception_info'] = {'exception_type': 'NonZeroAgentExitCodeError'}
    (root / 'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='verified termination'):
        collect(ident, result['id'], root / 'result.json', root, RewardProfile(),
                termination_evidence={'profile': 'claude_code_2.1.141_max_turns_v1',
                                      'source_result_digest': 'another result'})


@pytest.mark.parametrize('mutation', [None, 'changed', 'missing', 'true', 'numeric', 'unknown',
                                      'required', 'schema_missing', 'tool', 'id', 'text'])
def test_native_consumption_only_accepts_declared_optional_defaults(tmp_path, mutation):
    ident = identity()
    registry = TraceRegistry(tmp_path / 'trace')
    token = registry.register(ident, 'trial-' + ident.attempt_id)
    aid, request, _, _ = registry.begin(token)
    turn = make_ir(tmp_path / 'fixture')['turns'][0]
    block = {'type': 'tool_use', 'id': 'tool-1', 'name': 'Edit',
             'input': {'file_path': '/workspace/code.py', 'old_string': 'a', 'new_string': 'b'}}
    schema = {'type': 'object', 'properties': {'replace_all': {'type': 'boolean', 'default': False}},
              'required': ['file_path', 'old_string', 'new_string']}
    turn.update(request_id=request, delivery='sent', response={'content': [block]},
                client_request={'tools': [{'name': 'Edit', 'input_schema': schema}]})
    content = [deepcopy(block)]
    content[0]['input']['replace_all'] = False
    if mutation == 'changed':
        content[0]['input']['new_string'] = 'c'
    elif mutation == 'missing':
        content[0]['input'].pop('old_string')
    elif mutation == 'true':
        content[0]['input']['replace_all'] = True
    elif mutation == 'numeric':
        content[0]['input']['replace_all'] = 0
    elif mutation == 'unknown':
        content[0]['input']['extra'] = False
    elif mutation == 'required':
        schema['required'].append('replace_all')
    elif mutation == 'schema_missing':
        schema['properties']['replace_all'].pop('default')
    elif mutation == 'tool':
        content[0]['name'] = 'Write'
    elif mutation == 'id':
        content[0]['id'] = 'different'
    elif mutation == 'text':
        content.append({'type': 'text', 'text': 'injected'})
    registry.finish(aid, request, turn)
    registry.delivery(aid, turn['response_id'], 'sent')
    if mutation:
        with pytest.raises(ValueError):
            registry.consume(aid, turn['response_id'], content, 'session#sha')
        return
    registry.consume(aid, turn['response_id'], content, 'session#sha')
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert not seal['errors']
    assert turns[0]['response'] == turn['response']
    assert 'replace_all' not in turns[0]['response']['content'][0]['input']
    proof = turns[0]['consumption_tool_defaults']
    assert proof['changes'][0]['defaults'] == {'replace_all': False}
    assert proof['served_content_digest'] == digest(turn['response']['content'])
    assert proof['consumed_content_digest'] == digest(content)


@pytest.mark.parametrize('tool', ['Edit', 'Write'])
@pytest.mark.parametrize('mutation', [None, 'wrong_version', 'missing_version', 'markdown',
    'field', 'indent', 'newline', 'path', 'name', 'id', 'schema', 'default', 'extra', 'old_string'])
def test_pinned_cli_whitespace_audit_preserves_exact_sample(tmp_path, tool, mutation):
    from harborrl.rollout.harbor_job.audit import audit_session

    registry = TraceRegistry(tmp_path / 'trace')
    ident = identity()
    token = registry.register(ident, 'trial')
    aid, request, _, _ = registry.begin(token)
    turn = make_ir(tmp_path / 'fixture')['turns'][0]
    field = 'new_string' if tool == 'Edit' else 'content'
    path = '/workspace/code.MDX' if mutation == 'markdown' else '/workspace/code.py'
    text = 'def f():  \r\n    return 1\t\r    \n# end\u00a0'
    normalized = 'def f():\r\n    return 1\r\n# end'
    inputs = {'file_path': path, field: text}
    properties = {'file_path': {'type': 'string'}, field: {'type': 'string'}}
    if tool == 'Edit':
        inputs['old_string'] = 'old  \n'
        properties.update(old_string={'type': 'string'}, replace_all={'type': 'boolean', 'default': False})
    block = {'type': 'tool_use', 'id': 'tool-1', 'name': tool, 'input': inputs}
    turn.update(request_id=request, response={'content': [block]},
                client_request={'tools': [{'name': tool, 'input_schema': {
                    'type': 'object', 'properties': properties, 'required': list(inputs)}}]})
    content = [deepcopy(block)]
    consumed = content[0]['input']
    consumed[field] = normalized
    if tool == 'Edit':
        consumed['replace_all'] = False
    version = '2.1.141'
    if mutation == 'wrong_version':
        version = '2.1.140'
    elif mutation == 'missing_version':
        version = None
    elif mutation == 'field':
        consumed[field] += 'changed'
    elif mutation == 'indent':
        consumed[field] = normalized.replace('    return', 'return')
    elif mutation == 'newline':
        consumed[field] = normalized.replace('\r\n', '\n')
    elif mutation == 'path':
        consumed['file_path'] = '/workspace/other.py'
    elif mutation == 'name':
        content[0]['name'] = 'Other'
    elif mutation == 'id':
        content[0]['id'] = 'other'
    elif mutation == 'schema':
        properties[field]['type'] = 'number'
    elif mutation == 'default':
        consumed['replace_all'] = True
    elif mutation == 'extra':
        consumed['extra'] = False
    elif mutation == 'old_string':
        consumed['old_string'] = 'old\n'
    snapshot = deepcopy(turn)
    registry.finish(aid, request, turn)
    registry.delivery(aid, turn['response_id'], 'sent')
    session = tmp_path / 'native-session.jsonl'
    session.write_text(json.dumps({'type': 'assistant', 'sessionId': 'session', 'version': version,
        'message': {'role': 'assistant', 'id': turn['response_id'], 'content': content}}) + '\n')
    if mutation:
        with pytest.raises(ValueError):
            audit_session(session, registry, aid, cli_version='2.1.141')
        assert registry.attempts[aid]['turns'][0]['response'] == snapshot['response']
        return
    audit_session(session, registry, aid, cli_version='2.1.141')
    registry.close(aid)
    turns, seal = registry.seal(aid)
    assert not seal['errors']
    for key in ('response', 'output_ids', 'text', 'logprobs'):
        assert turns[0].get(key) == snapshot.get(key)
    proof = turns[0]['consumption_tool_normalization']
    assert proof['kind'] == 'claude_code_2.1.141_tool_inputs_v1'
    assert proof['changes'][0]['field'] == field
    assert proof['served_content_digest'] == digest(snapshot['response']['content'])
    assert proof['consumed_content_digest'] == digest(content)
