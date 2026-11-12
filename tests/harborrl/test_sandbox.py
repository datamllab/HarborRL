"""Sandbox routing and fail-closed capacity contracts, without live resources."""
import asyncio
import json
from pathlib import Path

import pytest
import yaml
from test_native_training import native_config

from harborrl.config.native import load_config
from harborrl.rollout.harbor_job.sandbox_config import (
    SandboxAdmission,
    load_sandbox,
    task_binding,
)
from harborrl.tasks.content import tree_digest


def settings(tmp_path):
    return {'domain': 'https://sandbox.example', 'credential_file': 'key',
            'bindings': [{'image': 'registry.example/task@sha256:' + 'a' * 64,
                          'max_live_instances': 1}],
            'installation': {'mode': 'offline_binary', 'version': '2.1.0', 'binary': 'claude', 'sha256': 'b' * 64}}


def test_local_runner_opt_in_and_unknown_fields(tmp_path):
    path, _ = native_config(tmp_path)
    old = load_config(path)
    assert old['harbor']['workers'] == ['worker-a', 'worker-b']
    local = load_config(path, ['harbor.runner_transport=local', 'harbor.workers=[]',
                              'harbor.environment={runtime: sandbox, config: sandbox.yaml}'])
    assert local['harbor']['environment']['config'] == str(tmp_path / 'sandbox.yaml')
    with pytest.raises(ValueError, match='local runner'):
        load_config(path, ['harbor.runner_transport=local'])
    with pytest.raises(ValueError, match='fields'):
        load_config(path, ['harbor.environment={runtime: sandbox, config: x, api_key: secret}'])


def test_sandbox_binding_and_secret_rejection(tmp_path):
    cfg = settings(tmp_path)
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump(cfg))
    result = load_sandbox(path)
    assert result['credential_file'] == str(tmp_path / 'key')
    (tmp_path / 'task.toml').write_text('[environment]\ndocker_image = "' + cfg['bindings'][0]['image'] + '"\n')
    assert task_binding(result, tmp_path)['image'] == cfg['bindings'][0]['image']
    (tmp_path / 'task.toml').write_text('[environment]\ndocker_image = "wrong"\n')
    with pytest.raises(ValueError, match='exact binding'):
        task_binding(result, tmp_path)
    cfg['api_key'] = 'must-not-be-serialized'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='fields'):
        load_sandbox(path)


def test_task_digest_binding_preserves_original_files_and_rejects_changes(tmp_path):
    task = tmp_path / 'task'
    task.mkdir()
    (task / 'task.toml').write_text('[environment]\n')
    (task / 'instruction.md').write_text('Original task.')
    original = tree_digest(task)
    cfg = settings(tmp_path)
    cfg['bindings'][0]['task_digests'] = [original]
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump(cfg))
    config = load_sandbox(path)
    assert task_binding(config, task)['image'] == cfg['bindings'][0]['image']
    assert tree_digest(task) == original
    (task / 'instruction.md').write_text('Changed task.')
    with pytest.raises(ValueError, match='exact binding'):
        task_binding(config, task)
    (task / 'task.toml').write_text('[environment]\ndocker_image = "' + cfg['bindings'][0]['image'] + '"\n')
    with pytest.raises(ValueError, match='exact binding'):
        task_binding(config, task)


@pytest.mark.parametrize('digests', [[], 'a' * 64, [1], ['invalid'], ['a' * 64, 'a' * 64]])
def test_invalid_task_digest_bindings(tmp_path, digests):
    cfg = settings(tmp_path)
    cfg['bindings'][0]['task_digests'] = digests
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='digest'):
        load_sandbox(path)


def test_duplicate_task_digest_across_images(tmp_path):
    cfg = settings(tmp_path)
    cfg['bindings'][0]['task_digests'] = ['c' * 64]
    cfg['bindings'].append({'image': 'other', 'max_live_instances': 1, 'task_digests': ['c' * 64]})
    path = tmp_path / 'sandbox.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='duplicate sandbox task digest'):
        load_sandbox(path)


def test_admission_across_groups_and_cleanup_debt(tmp_path):
    async def run():
        cfg = {'max_live_instances': 2, 'bindings': [{'image': 'image', 'max_live_instances': 1}]}
        admission = SandboxAdmission(cfg)
        active, peak = 0, 0

        async def trial(index):
            nonlocal active, peak
            async with admission.lease(cfg['bindings'][0], tmp_path / str(index)):
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.01)
                active -= 1
        await asyncio.gather(*(trial(i) for i in range(6)))
        assert peak == 1
        root = tmp_path / 'failed'
        root.mkdir()
        async with admission.lease(cfg['bindings'][0], root):
            (root / 'cleanup-required.json').write_text('{}')
        with pytest.raises(RuntimeError, match='blocked'):
            async with admission.lease(cfg['bindings'][0], tmp_path):
                pytest.fail('must not allocate')
    asyncio.run(run())


def test_sandbox_launch_freezes_nonsecret_runtime(tmp_path):
    from harborrl.backends.slime_v032.config import launch_plan
    from harborrl.data.harbor.native_inspector import inspect_native
    path, task = native_config(tmp_path)
    cfg = settings(tmp_path)
    (tmp_path / 'sandbox.yaml').write_text(yaml.safe_dump(cfg))
    (task / 'task.toml').write_text('[environment]\ndocker_image = "' + cfg['bindings'][0]['image'] + '"\n')
    catalog = tmp_path / 'catalog.json'
    entries = json.loads(catalog.read_text())
    entries[0]['task_digest'] = inspect_native(task)['task_digest']
    catalog.write_text(json.dumps(entries))
    config = load_config(path, ['harbor.runner_transport=local', 'harbor.workers=[]',
                               'harbor.environment={runtime: sandbox, config: sandbox.yaml}'])
    plan = launch_plan(config)
    runtime = json.loads(plan['environment']['HARBORRL_NATIVE_SANDBOX'])
    assert runtime['installation']['version'] == '2.1.0'
    assert Path(runtime['installation']['binary']).is_absolute()
    assert plan['environment']['HARBORRL_NATIVE_RUNNER_WORKERS'] == '[]'
