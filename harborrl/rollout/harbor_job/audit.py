"""Audit persisted Claude native session JSONL, including final assistant output.

Never infer consumption from HTTP write success or matching request timestamps.
Unknown auxiliary sessions, summaries and conflicting duplicate messages reject
an attempt. Claude Code 2.1.141 may persist one logical assistant message as
multiple session lines (one content block per line, parallel tool calls); the
adapter merges those splits before the immutable trace consumption check. This
parser must be probed against the locked real CLI version.
"""
import hashlib
import json
from pathlib import Path

from harborrl.trajectories.native import digest, read_json

_CONTINUATION = ('Output token limit hit. Resume directly — no apology, no recap of what you were doing. '
                 'Pick up mid-thought if that is where the cut happened. Break remaining work into smaller pieces.')


def _message_content(messages):
    """Ignore only probed cache placement and string/text-block serialization."""
    result = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('turn budget continuation lacks native message content')  # noqa: TRY004 - audit rejects malformed evidence
        content = message.get('content')
        if isinstance(content, str):
            content = [{'type': 'text', 'text': content}]
        if not isinstance(content, list) or any(not isinstance(block, dict) for block in content):
            raise ValueError('turn budget continuation lacks native message content')
        blocks = []
        for block in content:
            cache = block.get('cache_control')
            if 'cache_control' in block and cache != {'type': 'ephemeral'}:
                raise ValueError('turn budget continuation has unprobed cache metadata')
            blocks.append({key: value for key, value in block.items() if key != 'cache_control'})
        result.append({**message, 'content': blocks})
    return result


def _budget_continuations(turns, budget):
    """Every extra response must be a consumed length stop with a linked retry."""
    if not turns or turns[-1].get('finish_reason') != 'tool_use':
        raise ValueError('turn budget differs from sealed policy trace')
    proofs = []
    for index, turn in enumerate(turns):
        if turn.get('finish_reason') == 'tool_use':
            continue
        response = turn.get('response', {})
        if (turn.get('finish_reason') != 'max_tokens' or response.get('stop_reason') != 'max_tokens'
                or index + 1 == len(turns)):
            raise ValueError('turn budget includes an unverified extra response')
        content = response.get('content')
        if (not isinstance(content, list) or not content
                or any(not isinstance(block, dict) or block.get('type') != 'text' for block in content)):
            raise ValueError('turn budget continuation requires a text-only length stop')
        following = turns[index + 1]
        before = turn.get('client_request', {}).get('messages')
        after = following.get('client_request', {}).get('messages')
        if not isinstance(before, list) or not before or not isinstance(after, list):
            raise ValueError('turn budget continuation lacks request history')
        expected = [*before, {'role': 'assistant', 'content': content},
                    {'role': 'user', 'content': [{'type': 'text', 'text': _CONTINUATION}]}]
        if _message_content(after) != _message_content(expected):
            raise ValueError('turn budget continuation differs from the consumed length stop')
        for item in (turn, following):
            for key in ('client_request', 'response'):
                if item.get(key + '_digest') != digest(item.get(key)):
                    raise ValueError('turn budget continuation has conflicting trace digests')
        proofs.append({'response_id': turn['response_id'], 'next_response_id': following['response_id'],
                       'response_digest': turn['response_digest'],
                       'continuation_request_digest': following['client_request_digest']})
    if len(turns) - len(proofs) != budget:
        raise ValueError('turn budget differs from sealed policy trace')
    return proofs


def audit_termination(path, result_path, profile, audit, *, turns):
    """Recognize the probed CLI turn-budget exit from matching native evidence."""
    result_path = Path(result_path)
    result = read_json(result_path)
    exception = result.get('exception_info')
    if not exception:
        return None
    if (profile['cli_version'] != '2.1.141'
            or exception.get('exception_type') != 'NonZeroAgentExitCodeError'):
        raise ValueError('exceptional trial requires a verified termination profile')
    path = Path(path)
    raw = path.read_bytes()
    events = [json.loads(line) for line in raw.splitlines() if line.strip()]
    results = [event for event in events if event.get('type') == 'result']
    inits = [event for event in events if event.get('type') == 'system' and event.get('subtype') == 'init']
    if len(results) != 1 or events[-1] != results[0] or len(inits) != 1:
        raise ValueError('turn budget requires one native init and final result')
    final, initial = results[0], inits[0]
    budget = profile['max_turns']
    expected = {'subtype': 'error_max_turns', 'terminal_reason': 'max_turns',
                'stop_reason': 'tool_use', 'session_id': audit['session_id'],
                'errors': [f'Reached maximum number of turns ({budget})']}
    if (any(final.get(key) != value for key, value in expected.items())
            or final.get('is_error') is not True or final.get('permission_denials') != []
            or type(final.get('num_turns')) is not int or final['num_turns'] != budget + 1
            or initial.get('session_id') != audit['session_id']
            or initial.get('claude_code_version') != profile['cli_version']):
        raise ValueError('turn budget lacks matching CLI termination evidence')
    responses = [event for event in events if event.get('type') == 'assistant']
    ids = list(dict.fromkeys(event.get('message', {}).get('id') for event in responses))
    if (ids != audit['consumed_response_ids'] or ids != [turn['response_id'] for turn in turns]
            or any(event.get('session_id') != audit['session_id'] for event in responses)
            or not ids or not all(isinstance(value, str) and value for value in ids)):
        raise ValueError('turn budget differs from consumed native responses')
    continuations = _budget_continuations(turns, budget)
    return {'profile': 'claude_code_2.1.141_max_turns_v1', 'reason': 'max_turns',
            'session_id': audit['session_id'], 'max_turns': budget,
            'policy_response_count': len(turns), 'logical_turns': budget, 'continuations': continuations,
            'consumed_response_ids': ids, 'session_evidence_ref': audit['evidence_ref'],
            'source_result_digest': hashlib.sha256(result_path.read_bytes()).hexdigest(),
            'cli_stream_ref': str(path), 'cli_stream_digest': hashlib.sha256(raw).hexdigest()}


def audit_session(path, registry, attempt_id, *, cli_version=None):
    path = Path(path)
    raw = path.read_bytes()
    evidence = f"{path}#sha256={hashlib.sha256(raw).hexdigest()}"
    sessions = set()
    tool_results = []
    merged = {}  # message id -> merged content blocks, in first-seen order
    versions = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("isSidechain") or event.get("type") == "summary" or event.get("subtype") == "compact_boundary":
            raise ValueError("auxiliary generation or compact is not enabled")
        if event.get("sessionId"):
            sessions.add(event["sessionId"])
        if event.get('type') == 'user':
            content = event.get('message', {}).get('content')
            if isinstance(content, list):
                tool_results.extend(block for block in content
                                    if isinstance(block, dict) and block.get('type') == 'tool_result')
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        if message.get("role") != "assistant" or not isinstance(message.get("content"), list) or not message.get("id"):
            raise ValueError("assistant consumption lacks ID or content")
        rid, content = message["id"], message["content"]
        versions.setdefault(rid, set()).add(event.get('version'))
        if rid not in merged:
            merged[rid] = list(content)
            continue
        current = merged[rid]
        if content == current or current[:len(content)] == content:
            continue  # identical rewrite or a shorter recap of the same message
        if content[:len(current)] == current:
            merged[rid] = list(content)  # growing incremental update of one message
        else:
            current.extend(content)  # per-block split of one logical message
    records = [(rid, content) for rid, content in merged.items()]
    if len(sessions) != 1 or not records:
        raise ValueError("expected one complete native session")
    # Validate the complete file before mutating the registry; seal catches omitted responses.
    registry.audit_tool_rejections(attempt_id, tool_results, evidence)
    for rid, content in records:
        verified_version = cli_version if versions[rid] == {cli_version} else None
        registry.consume(attempt_id, rid, content, evidence, cli_version=verified_version)
    return {"session_id": next(iter(sessions)), "consumed_response_ids": list(merged), "evidence_ref": evidence}
