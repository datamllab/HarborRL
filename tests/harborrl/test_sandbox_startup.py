"""Exercise Runner queue ownership with the pinned Harbor timer implementation."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip('harbor.environments.opensandbox')

from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from test_native_contracts import identity

from harborrl.rollout.harbor_job import runner
from harborrl.rollout.harbor_job.environments.opensandbox import (
    ManagedOpenSandboxEnvironment,
    startup_slot,
)
from harborrl.trajectories.native import digest, read_json


@pytest.mark.parametrize('mode', ['ready', 'start_failed', 'cancel_waiting'])
def test_runner_queue_does_not_consume_harbor_environment_timer(tmp_path, monkeypatch, mode):
    from harborrl.rollout.harbor_job import sandbox_config

    task = tmp_path / 'task'
    task.mkdir()
    profile = {'cli_version': '2.1.141', 'model': 'policy', 'max_turns': 2,
               'setup_timeout_sec': 1, 'agent_timeout_sec': 1, 'verifier_timeout_sec': 1}
    sandbox = {'adapter': 'test:Environment', 'installation': None, 'bindings': []}
    output = tmp_path / 'output'
    monkeypatch.setattr(runner, 'probe', dict)
    monkeypatch.setattr(runner.NATIVE_TASK_PROVIDER.adapter, 'inspect', lambda *a, **k: {
        'status': 'NEEDS_PROBE', 'path': str(task), 'resources': {}})
    monkeypatch.setattr(sandbox_config, 'environment_class', lambda _: ManagedOpenSandboxEnvironment)
    monkeypatch.setattr(sandbox_config, 'task_binding', lambda *a: {})
    env = object.__new__(ManagedOpenSandboxEnvironment)
    env.cfg = {'startup_concurrency': 1}
    env.evidence = output
    env.startup_dir = tmp_path / 'locks'
    env._startup_reservation = None
    env.record = {'sandbox_id': None, 'state': 'pending', 'events': []}
    starts = []

    async def start(force_build):
        assert env._startup_reservation is not None
        starts.append(force_build)
        try:
            await asyncio.sleep(.001)
            if mode == 'start_failed':
                raise RuntimeError('synthetic startup failure')
        finally:
            await env.release_startup()

    env.start = start

    class SyntheticTrial:
        id = 'synthetic-trial'
        _environment_build_timeout_sec = .05
        _now = staticmethod(lambda: datetime.now(timezone.utc))
        _start_agent_environment = Trial._start_agent_environment

        def __init__(self, config):
            self.config = config
            self.agent_environment = env
            self.result = SimpleNamespace(exception_info=None)
            self.paths = SimpleNamespace(result_path=output / 'harbor/result.json', trial_dir=output / 'harbor')
            self.hooks = {}

        def add_hook(self, event, hook):
            self.hooks[event] = hook

        async def _emit(self, event):
            await self.hooks[event](SimpleNamespace(
                event=event, trial_id=self.id, timestamp=self._now()))

        async def run(self):
            try:
                await Trial._setup_agent_environment(self)
            except RuntimeError as exc:
                self.result.exception_info = SimpleNamespace(exception_type=type(exc).__name__)
            return self.result

    async def create(config):
        assert isinstance(config, TrialConfig)
        return SyntheticTrial(config)

    monkeypatch.setattr(Trial, 'create', create)
    request = {'identity': identity(harness_digest=digest(profile)).to_dict(),
               'task_path': str(task), 'profile': profile, 'output': str(output),
               'sandbox': sandbox, 'runtime_digest': digest(sandbox),
               'gateway_url': 'http://gateway', 'credential': 'synthetic', 'max_output_tokens': 16}

    async def exercise():
        events = []
        run_sent = False
        cancel = asyncio.Event()

        async def read():
            nonlocal run_sent
            if not run_sent:
                run_sent = True
                return {'command': 'run'}
            await cancel.wait()
            return {'command': 'cancel'}

        async with startup_slot(env.startup_dir, 1):
            running = asyncio.create_task(runner.execute(request, events.append, read))
            async with asyncio.timeout(2):
                while env.record['state'] != 'waiting':
                    await asyncio.sleep(.001)
            await asyncio.sleep(.08)
            assert not running.done() and not starts
            if mode == 'cancel_waiting':
                cancel.set()
                await asyncio.wait_for(running, 1)
                assert events[-1]['type'] == 'cancelled'
        if mode != 'cancel_waiting':
            await asyncio.wait_for(running, 2)
            assert len(starts) == 1
            assert events[-1]['exception'] == ('RuntimeError' if mode == 'start_failed' else None)
        assert env._startup_reservation is None
        async with asyncio.timeout(1), startup_slot(env.startup_dir, 1):
            pass
        lifecycle = read_json(output / 'sandbox-lifecycle.json')
        assert lifecycle['sandbox_id'] is None
        assert not (output / 'cleanup-required.json').exists()
        assert [row['state'] for row in lifecycle['events']] == (
            ['waiting'] if mode == 'cancel_waiting' else ['waiting', 'reserved'])

    asyncio.run(exercise())


@pytest.mark.parametrize('mode', ['ready', 'create_failed', 'readiness_failed', 'binding_failed'])
def test_sdk_startup_reuses_reservation_and_releases_on_failure(tmp_path, mode):
    env = object.__new__(ManagedOpenSandboxEnvironment)
    env.cfg = {'startup_concurrency': 1, 'create_timeout_s': 1, 'request_timeout_s': 1,
               'lifecycle_minutes': 1}
    env.evidence, env.startup_dir = tmp_path, tmp_path / 'locks'
    env._startup_reservation = None
    env._sandbox = None
    env._startup_transport = None
    env._persistent_env = {}
    env.binding = {'image': 'synthetic-image'}
    env.record = {'sandbox_id': None, 'state': 'pending', 'events': [], 'runtime_digest': 'test'}
    env._build_connection_config = lambda _: SimpleNamespace(model_copy=lambda **k: None)
    env._build_resource = dict
    env._build_network_policy = lambda _: None

    class Transport:
        async def aclose(self):
            pass

    env.startup_transport = lambda _: Transport()

    async def validate():
        if mode == 'binding_failed':
            raise RuntimeError('synthetic binding failure')

    async def ready(_):
        if mode == 'readiness_failed':
            raise RuntimeError('synthetic readiness failure')

    async def stop(delete):
        env.record_state('deleted')

    class Sandbox:
        @staticmethod
        async def create(*a, **k):
            assert env._startup_reservation is not None
            if mode == 'create_failed':
                raise RuntimeError('synthetic create failure')
            return SimpleNamespace(id='synthetic-sandbox')

    env.validate_binding, env._wait_until_ready, env.stop = validate, ready, stop

    async def exercise():
        await env.reserve_startup()
        with pytest.raises(RuntimeError, match='already reserved'):
            await env.reserve_startup()
        if mode == 'ready':
            assert (await env._create_sandbox({'Sandbox': Sandbox})).id == 'synthetic-sandbox'
        else:
            with pytest.raises(RuntimeError):
                await env._create_sandbox({'Sandbox': Sandbox})
        assert env._startup_reservation is None
        async with asyncio.timeout(1), startup_slot(env.startup_dir, 1):
            pass
        states = [row['state'] for row in env.record['events']]
        assert states.count('reserved') == 1
        if mode == 'create_failed':
            assert json.loads((tmp_path / 'cleanup-required.json').read_text())['reason'].startswith('Creation outcome unknown')

    asyncio.run(exercise())
