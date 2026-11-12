"""Frozen Qwen math PRM identity, step alignment and scoring receipts."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.consistency.sglang import http_json
from harborrl.trajectories.native import digest, finite, read_json

ADAPTER = 'qwen-math-prm-v1'
MARKER = '<extra_0>'
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'
FIELDS = {'endpoint', 'checkpoint', 'revision'}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('process_reward requires endpoint, checkpoint and revision')
    if any(not isinstance(value, str) or not value.strip() for value in config.values()):
        raise ValueError('process_reward fields must be nonempty strings')
    url = urlsplit(config['endpoint'])
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ('', '/')):
        raise ValueError('process_reward.endpoint must be a direct HTTP(S) origin')
    if not re.fullmatch(r'[0-9a-f]{40}', config['revision']):
        raise ValueError('process_reward.revision must be an immutable 40-character commit')


def freeze(config):
    """Pin local weights, tokenizer and executable model sources by content."""
    validate_config(config)
    root = Path(config['checkpoint']).resolve()
    model = read_json(root / 'config.json')
    if (model.get('architectures') != ['Qwen2ForProcessRewardModel']
            or model.get('auto_map') != {
                'AutoConfig': 'configuration_qwen2_rm.Qwen2RMConfig',
                'AutoModel': 'modeling_qwen2_rm.Qwen2ForProcessRewardModel'}):
        raise ValueError('process_reward requires the Qwen2 math process classification model')
    names = {'config.json', 'tokenizer.json', 'tokenizer_config.json',
             'configuration_qwen2_rm.py', 'modeling_qwen2_rm.py'}
    index = root / 'model.safetensors.index.json'
    if index.is_file():
        weights = read_json(index).get('weight_map', {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError('PRM requires a nonempty safetensors index')
        names.update(weights.values())
        names.add(index.name)
    else:
        names.add('model.safetensors')
    names.update(name for name in ('special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt')
                 if (root / name).is_file())
    files = {}
    for name in sorted(names):
        if not isinstance(name, str) or Path(name).name != name or not (root / name).is_file():
            raise ValueError('PRM requires a complete local checkpoint and model sources')
        value = hashlib.sha256()
        with (root / name).open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                value.update(chunk)
        files[name] = value.hexdigest()
    frozen = {**config, 'checkpoint': str(root), 'endpoint': config['endpoint'].rstrip('/'),
              'adapter': ADAPTER, 'files': files}
    # Endpoint is transport configuration, not part of the model's identity.
    return {**frozen, 'identity': digest({key: value for key, value in frozen.items() if key != 'endpoint'})}


def load(path):
    frozen = read_json(path)
    expected = digest({key: value for key, value in frozen.items() if key not in ('identity', 'endpoint')})
    if frozen.get('identity') != expected or frozen.get('adapter') != ADAPTER:
        raise ValueError('PRM manifest identity changed')
    validate_config({key: frozen[key] for key in FIELDS})
    return frozen


def check(frozen, *, transport=http_json):
    info = transport(frozen['endpoint'] + '/model_info')
    if any(info.get(key) != frozen[key] for key in ('identity', 'revision', 'adapter', 'checkpoint')):
        raise ValueError('PRM scorer identity changed')
    if info.get('frozen') is not True or info.get('is_generation') is not False:
        raise ValueError('PRM must be a frozen inference-only scorer')
    return {key: info[key] for key in ('identity', 'revision', 'adapter', 'checkpoint', 'frozen')}


def split_steps(response_ids, tokenizer):
    """Double-newline steps; an indivisible boundary token stays in its step."""
    if (not response_ids or len(response_ids) > 16384
            or any(type(token) is not int or token < 0 for token in response_ids)):
        raise ValueError('PRM requires a nonempty bounded response token sequence')
    text = tokenizer.decode(response_ids, skip_special_tokens=True)
    ends, steps, char_start, seen = [], [], 0, 0
    for index in range(1, len(response_ids) + 1):
        prefix = tokenizer.decode(response_ids[:index], skip_special_tokens=True)
        # Byte-level tokenizers can temporarily end in an incomplete code point.
        if not text.startswith(prefix):
            continue
        count = prefix.count('\n\n')
        if count < seen:
            raise ValueError('PRM tokenizer changed an established step boundary')
        if count > seen and prefix[char_start:].strip():
            ends.append(index)
            steps.append(prefix[char_start:])
            char_start = len(prefix)
        seen = count
    tail = text[char_start:]
    if tail.strip():
        ends.append(len(response_ids))
        steps.append(tail)
    elif steps:
        ends[-1] = len(response_ids)
        steps[-1] += tail
    else:
        raise ValueError('PRM cannot score a response with no visible reasoning')
    if ''.join(steps) != text:
        raise ValueError('PRM step alignment changed the visible response')
    return ends, steps


def token_returns(ends, probabilities, response_length):
    """Signed rewards 2*p-1, undiscounted reward-to-go within each token span."""
    if (type(response_length) is not int or response_length < 1 or not ends
            or len(ends) != len(probabilities) or ends[-1] != response_length
            or any(type(end) is not int for end in ends)
            or any(left >= right for left, right in zip((0, *ends[:-1]), ends, strict=True))):
        raise ValueError('PRM step ends must partition every response token exactly once')
    if any(not finite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError('PRM probabilities must be finite and in [0, 1]')
    rewards = [2 * value - 1 for value in probabilities]
    values = [0.] * response_length
    starts = (0, *ends[:-1])
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        values[start:end] = [math.fsum(rewards[index:])] * (end - start)
    return values


def request_payload(identity, query, steps):
    if (not isinstance(query, str) or not query.strip() or not isinstance(steps, (tuple, list))
            or not steps or any(not isinstance(step, str) or not step.strip() for step in steps)
            or any(MARKER in value for value in (query, *steps))):
        raise ValueError('PRM query and nonempty steps must not contain the reward marker')
    return {'identity': identity, 'query': query, 'steps': list(steps)}


def validate_receipt(frozen, receipt, query, response_ids, tokenizer):
    ends, steps = split_steps(response_ids, tokenizer)
    payload = request_payload(frozen['identity'], query, steps)
    if (receipt.get('identity') != frozen['identity']
            or receipt.get('request_digest') != digest(payload)
            or receipt.get('response_tokens_digest') != digest(response_ids)
            or receipt.get('step_ends') != ends or receipt.get('steps') != steps):
        raise ValueError('PRM scorer, request or exact response-token evidence changed')
    values = token_returns(ends, receipt.get('probabilities', []), len(response_ids))
    if receipt.get('token_returns') != values:
        raise ValueError('PRM reward-to-go evidence changed')
    if (not isinstance(receipt.get('scorer_input_digest'), str)
            or not re.fullmatch(r'[0-9a-f]{64}', receipt['scorer_input_digest'])):
        raise ValueError('PRM scoring input digest is missing')
    return values


def score_response(frozen, query, response_ids, tokenizer, *, transport=http_json):
    ends, steps = split_steps(response_ids, tokenizer)
    payload = request_payload(frozen['identity'], query, steps)
    result = transport(frozen['endpoint'] + '/score', payload)
    receipt = {**result, 'response_tokens_digest': digest(response_ids), 'step_ends': ends, 'steps': steps,
               'token_returns': token_returns(ends, result.get('probabilities', []), len(response_ids))}
    validate_receipt(frozen, receipt, query, response_ids, tokenizer)
    return receipt
