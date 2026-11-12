"""Pinned CLI logical turns require exact consumed length-stop continuations."""
import json
from copy import deepcopy

import pytest

from harborrl.rollout.harbor_job.audit import audit_termination
from harborrl.trajectories.native import digest

CONTINUATION = ('Output token limit hit. Resume directly — no apology, no recap of what you were doing. '
                'Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.')


def fixture(lengths):
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Inspect the file.'}]}]
    turns = []
    for index in range(2 + len(lengths)):
        length = index in lengths
        content = ([{'type': 'text', 'text': 'Still inspecting'}] if length else
                   [{'type': 'tool_use', 'id': f'tool-{index}', 'name': 'Read', 'input': {'file_path': '/code.py'}}])
        turn = {'response_id': f'r{index}', 'finish_reason': 'max_tokens' if length else 'tool_use',
                'client_request': {'messages': deepcopy(messages)},
                'response': {'content': content, 'stop_reason': 'max_tokens' if length else 'tool_use'}}
        turns.append(turn)
        messages.append({'role': 'assistant', 'content': deepcopy(content)})
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': CONTINUATION}] if length else
                         [{'type': 'tool_result', 'tool_use_id': f'tool-{index}', 'content': 'value = 1'}]})
    return turns


def audit(tmp_path, turns):
    for turn in turns:
        for key in ('client_request', 'response'):
            turn.setdefault(key + '_digest', digest(turn[key]))
    session = {'session_id': 'session', 'consumed_response_ids': [turn['response_id'] for turn in turns],
               'evidence_ref': 'native-session#sha256=diagnostic'}
    events = [{'type': 'system', 'subtype': 'init', 'session_id': 'session', 'claude_code_version': '2.1.141'}]
    events.extend({'type': 'assistant', 'session_id': 'session', 'message': {'id': turn['response_id']}}
                  for turn in turns)
    events.append({'type': 'result', 'subtype': 'error_max_turns', 'is_error': True,
        'terminal_reason': 'max_turns', 'stop_reason': 'tool_use', 'num_turns': 3,
        'session_id': 'session', 'permission_denials': [], 'errors': ['Reached maximum number of turns (2)']})
    path, result = tmp_path / 'stream.jsonl', tmp_path / 'result.json'
    path.write_text('\n'.join(map(json.dumps, events)))
    result.write_text(json.dumps({'exception_info': {'exception_type': 'NonZeroAgentExitCodeError'}}))
    return audit_termination(path, result, {'cli_version': '2.1.141', 'max_turns': 2}, session, turns=turns)


@pytest.mark.parametrize('lengths', [set(), {0}, {1}, {0, 1}, {0, 2}])
def test_logical_turns_preserve_every_consumed_response(tmp_path, lengths):
    turns = fixture(lengths)
    # The pinned CLI moves ephemeral cache metadata and serializes older text as strings.
    if lengths == {0, 1}:
        turns[2]['client_request']['messages'][2]['content'] = CONTINUATION
    for turn in turns:
        turn['client_request']['messages'][-1]['content'][-1]['cache_control'] = {'type': 'ephemeral'}
    snapshot = deepcopy(turns)
    evidence = audit(tmp_path, turns)
    assert evidence['logical_turns'] == evidence['max_turns'] == 2
    assert evidence['policy_response_count'] == len(turns) == 2 + len(lengths)
    assert evidence['consumed_response_ids'] == [turn['response_id'] for turn in turns]
    assert len(evidence['continuations']) == len(lengths)
    assert [{key: value for key, value in turn.items() if not key.endswith('_digest')}
            for turn in turns] == snapshot
    for proof in evidence['continuations']:
        index = int(proof['response_id'][1:])
        assert proof['next_response_id'] == turns[index + 1]['response_id']
        assert proof['continuation_request_digest'] == digest(turns[index + 1]['client_request'])


@pytest.mark.parametrize('mutation', [
    'arbitrary_extra', 'missing_continuation', 'changed_continuation', 'continuation_role',
    'changed_history', 'missing_history', 'extra_history', 'changed_assistant', 'assistant_role',
    'extra_field', 'extra_block', 'unknown_cache', 'null_cache', 'malformed_message',
    'malformed_block', 'tool_length', 'empty_length', 'stop_reason',
    'end_turn', 'final_length', 'missing_logical_turn', 'extra_logical_turn',
    'request_digest', 'response_digest', 'next_digest', 'reordered',
])
def test_unlinked_or_unprobed_extra_response_rejected(tmp_path, mutation):
    turns = fixture({0})
    following = turns[1]['client_request']['messages']
    if mutation == 'arbitrary_extra':
        turns[0]['finish_reason'] = 'tool_use'
    elif mutation == 'missing_continuation':
        following.pop()
    elif mutation == 'changed_continuation':
        following[-1]['content'][0]['text'] += ' modified'
    elif mutation == 'continuation_role':
        following[-1]['role'] = 'assistant'
    elif mutation == 'changed_history':
        following[0]['content'][0]['text'] += ' modified'
    elif mutation == 'missing_history':
        turns[0]['client_request'].pop('messages')
    elif mutation == 'extra_history':
        following.insert(0, {'role': 'user', 'content': 'injected'})
    elif mutation == 'changed_assistant':
        following[-2]['content'][0]['text'] += ' modified'
    elif mutation == 'assistant_role':
        following[-2]['role'] = 'user'
    elif mutation == 'extra_field':
        following[-1]['injected'] = True
    elif mutation == 'extra_block':
        following[-1]['content'].append({'type': 'text', 'text': 'injected'})
    elif mutation == 'unknown_cache':
        following[0]['content'][0]['cache_control'] = {'type': 'ephemeral', 'injected': True}
    elif mutation == 'null_cache':
        following[0]['content'][0]['cache_control'] = None
    elif mutation == 'malformed_message':
        following[0] = None
    elif mutation == 'malformed_block':
        turns[0]['response']['content'] = [None]
    elif mutation == 'tool_length':
        turns[0]['response']['content'] = turns[1]['response']['content']
    elif mutation == 'empty_length':
        turns[0]['response']['content'] = []
    elif mutation == 'stop_reason':
        turns[0]['response']['stop_reason'] = 'tool_use'
    elif mutation == 'end_turn':
        turns[0]['finish_reason'] = 'end_turn'
    elif mutation == 'final_length':
        turns[-1]['finish_reason'] = 'max_tokens'
    elif mutation == 'missing_logical_turn':
        turns.pop()
    elif mutation == 'extra_logical_turn':
        turns.append({**deepcopy(turns[-1]), 'response_id': 'extra'})
    elif mutation == 'request_digest':
        turns[0]['client_request_digest'] = 'conflicting'
    elif mutation == 'response_digest':
        turns[0]['response_digest'] = 'conflicting'
    elif mutation == 'next_digest':
        turns[1]['client_request_digest'] = 'conflicting'
    else:
        turns.reverse()
    with pytest.raises(ValueError):
        audit(tmp_path, turns)
