"""Non-secret Sandbox settings; no SDK imports in the training process."""
import asyncio
import copy
import importlib
import re
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import yaml

from harborrl.tasks.content import tomllib, tree_digest


def _check_options(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError('sandbox option names must be strings')
            name = key.lower().replace('-', '_')
            if name in {'token', 'credential', 'password', 'secret', 'authorization'} or name.endswith(('_api_key', '_token', '_secret')) or name == 'api_key':
                raise ValueError('sandbox options must reference credential files, not contain secrets')
            _check_options(item)
    elif isinstance(value, list):
        for item in value:
            _check_options(item)


def load_sandbox(path):
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    required = {'domain', 'credential_file', 'bindings'}
    defaults = {'protocol': 'https', 'use_server_proxy': True,
                'request_timeout_s': 60, 'create_timeout_s': 600, 'lifecycle_minutes': 120,
                'cleanup_timeout_s': 120, 'max_live_instances': 2, 'startup_concurrency': 1,
                'adapter': 'harborrl.rollout.harbor_job.environments.opensandbox:ManagedOpenSandboxEnvironment',
                'options': {}, 'installation': None}
    if not isinstance(cfg, dict) or not required <= set(cfg) or set(cfg) - required - defaults.keys():
        raise ValueError('invalid sandbox configuration fields')
    cfg = {**defaults, **copy.deepcopy(cfg)}
    url = urlsplit(cfg['domain'])
    if (url.scheme not in ('http', 'https')
            or not url.hostname or url.username or url.password or url.query or url.fragment
            or url.path.rstrip('/').endswith('/v1') or cfg['protocol'] != url.scheme):
        raise ValueError('invalid sandbox domain/protocol')
    for key in defaults.keys() - {'protocol', 'use_server_proxy', 'adapter', 'options', 'installation'}:
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f'sandbox {key} must be a positive integer')
    if type(cfg['use_server_proxy']) is not bool or not isinstance(cfg['options'], dict):
        raise ValueError('invalid sandbox proxy/options')
    _check_options(cfg['options'])
    if not isinstance(cfg['adapter'], str) or not re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', cfg['adapter']):
        raise ValueError('sandbox adapter must be module:Class')
    if not isinstance(cfg['credential_file'], str) or not cfg['credential_file']:
        raise ValueError('sandbox credential_file required')
    cfg['credential_file'] = str((path.parent / cfg['credential_file']).resolve())
    bindings = cfg['bindings']
    if not isinstance(bindings, list) or not bindings:
        raise ValueError('sandbox image bindings required')
    images = set()
    task_digests = set()
    for binding in bindings:
        if not isinstance(binding, dict) or set(binding) - {'image', 'max_live_instances', 'options', 'task_digests'} or not {'image', 'max_live_instances'} <= set(binding):
            raise ValueError('invalid sandbox binding fields')
        binding.setdefault('options', {})
        _check_options(binding['options'])
        if (not isinstance(binding['image'], str) or not binding['image']
                or binding['image'] in images or not isinstance(binding['options'], dict)):
            raise ValueError('invalid or duplicate sandbox image binding')
        if type(binding['max_live_instances']) is not int or binding['max_live_instances'] <= 0:
            raise ValueError('binding max_live_instances must be positive')
        if 'task_digests' in binding:
            digests = binding['task_digests']
            if (not isinstance(digests, list) or not digests or any(
                    not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                    for value in digests)):
                raise ValueError('binding task_digests must be a nonempty list of SHA256 digests')
            if len(set(digests)) != len(digests) or task_digests.intersection(digests):
                raise ValueError('duplicate sandbox task digest binding')
            task_digests.update(digests)
        images.add(binding['image'])
    install = cfg['installation']
    if install is None:
        return cfg
    if not isinstance(install, dict) or set(install) != {'mode', 'version', 'binary', 'sha256'}:
        raise ValueError('installation requires mode, version, binary, sha256')
    if (install['mode'] != 'offline_binary' or not isinstance(install['version'], str)
            or not re.fullmatch(r'\d+\.\d+\.\d+', install['version'])
            or not isinstance(install['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', install['sha256'])
            or not isinstance(install['binary'], str) or not install['binary']):
        raise ValueError('invalid pinned offline binary installation')
    install['binary'] = str((path.parent / install['binary']).resolve())
    return cfg


def task_binding(cfg, task_path):
    if any(binding.get('task_digests') for binding in cfg['bindings']):
        task_digest = tree_digest(Path(task_path))
        for binding in cfg['bindings']:
            if task_digest in binding.get('task_digests', ()):
                return binding
    spec = tomllib.loads((Path(task_path) / 'task.toml').read_text())
    image = spec.get('environment', {}).get('docker_image')
    for binding in cfg['bindings']:
        if image == binding['image'] and not binding.get('task_digests'):
            return binding
    raise ValueError('sandbox task requires a prebuilt image with an exact binding')


def environment_class(cfg):
    """Import trusted deployment code only in the runner, never in GPU training."""
    module, name = cfg['adapter'].split(':')
    cls = getattr(importlib.import_module(module), name)
    from harborrl.rollout.harbor_job.environments.opensandbox import (
        ManagedOpenSandboxEnvironment,
    )
    if not isinstance(cls, type) or not issubclass(cls, ManagedOpenSandboxEnvironment):
        raise TypeError('sandbox adapter must implement the audited OpenSandbox lifecycle')
    cls.validate_config(cfg)
    return cls


def read_credential(cfg):
    path = Path(cfg['credential_file'])
    if path.stat().st_mode & 0o077:
        raise ValueError('sandbox credential file must have mode 600 or stricter')
    value = path.read_text().strip()
    if not value or '\n' in value:
        raise ValueError('sandbox credential file must contain one nonempty token')
    return value


class SandboxAdmission:
    """One runtime owns capacity; unresolved cleanup blocks further admission."""
    def __init__(self, cfg):
        self.total = asyncio.Semaphore(cfg['max_live_instances'])
        self.bindings = {b['image']: asyncio.Semaphore(b['max_live_instances']) for b in cfg['bindings']}
        self.blocked = False

    @asynccontextmanager
    async def lease(self, binding, root):
        # Wait for image-specific capacity before reserving a global live slot.
        async with self.bindings[binding['image']], self.total:
            if self.blocked:
                raise RuntimeError('sandbox admission blocked by unresolved cleanup')
            try:
                yield
            finally:
                if (Path(root) / 'cleanup-required.json').exists():
                    self.blocked = True
