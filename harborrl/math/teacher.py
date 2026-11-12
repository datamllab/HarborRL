"""Frozen, local-checkpoint SGLang teacher with exact student-token scoring."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

from harborrl.consistency.sglang import http_json, score
from harborrl.trajectories.native import digest, read_json

TOKENIZER_FILES = (
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'vocab.json', 'merges.txt',
)


def validate_config(config):
    if not isinstance(config, dict) or set(config) != {'endpoint', 'checkpoint', 'weight_version'}:
        raise ValueError('teacher requires endpoint, checkpoint and weight_version')
    if any(not isinstance(value, str) or not value.strip() for value in config.values()):
        raise ValueError('teacher fields must be nonempty strings')
    url = urlsplit(config['endpoint'])
    if (url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password
            or url.query or url.fragment or url.path not in ('', '/')):
        raise ValueError('teacher.endpoint must be a direct SGLang server HTTP(S) origin')


def _sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def _tokenizer_files(root):
    if any(not (root / name).is_file() for name in TOKENIZER_FILES[:2]):
        raise ValueError('OPD requires tokenizer.json and tokenizer_config.json')
    return {name: _sha256(root / name) for name in TOKENIZER_FILES if (root / name).is_file()}


def freeze(config, student_checkpoint):
    """Hash exactly the teacher safetensors checkpoint and compatible tokenizer."""
    validate_config(config)
    root = Path(config['checkpoint']).resolve()
    tokenizer = _tokenizer_files(root)
    if tokenizer != _tokenizer_files(Path(student_checkpoint)):
        raise ValueError('OPD teacher and student tokenizer files must match exactly')
    index = root / 'model.safetensors.index.json'
    names = {'config.json'}
    if index.is_file():
        weight_map = read_json(index).get('weight_map', {})
        if not weight_map:
            raise ValueError('Empty teacher safetensors weight map')
        names.update(weight_map.values())
        names.add(index.name)
    else:
        names.add('model.safetensors')
    files = {}
    for name in sorted(names):
        if Path(name).name != name or not (root / name).is_file():
            raise ValueError('Teacher requires a complete local safetensors checkpoint')
        files[name] = _sha256(root / name)
    frozen = {**config, 'checkpoint': str(root), 'endpoint': config['endpoint'].rstrip('/'),
              'files': files, 'tokenizer_files': tokenizer}
    return {**frozen, 'identity': digest(frozen)}


def load(path):
    frozen = read_json(path)
    if frozen.get('identity') != digest({key: value for key, value in frozen.items() if key != 'identity'}):
        raise ValueError('OPD teacher manifest identity changed')
    return frozen


def check(frozen, *, transport=http_json):
    """Check the trusted server's model, tokenizer and explicit weight version."""
    info = transport(frozen['endpoint'] + '/model_info')
    checkpoint = Path(frozen['checkpoint']).resolve()
    if (Path(info.get('model_path', '')).resolve() != checkpoint
            or Path(info.get('tokenizer_path') or info.get('model_path', '')).resolve() != checkpoint
            or str(info.get('weight_version', '')) != frozen['weight_version']
            or info.get('is_generation') is not True):
        raise ValueError('OPD teacher model, tokenizer or weight version changed')
    return {'checkpoint': str(checkpoint), 'weight_version': frozen['weight_version'],
            'identity': frozen['identity']}


def score_response(frozen, prompt, response, *, transport=http_json):
    values = score(frozen['endpoint'], prompt, response, frozen['weight_version'],
                   transport=transport, label='OPD teacher')
    return {'identity': frozen['identity'], 'weight_version': frozen['weight_version'],
            'logprobs': values, 'tokens_digest': digest(prompt + response)}
