"""Audited lifecycle around Harbor's OpenSandbox environment."""
import asyncio
import fcntl
import json
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

import httpx
from harbor.environments.opensandbox import OpenSandboxEnvironment

from harborrl.rollout.harbor_job.sandbox_config import read_credential
from harborrl.trajectories.native import digest, publish


@asynccontextmanager
async def startup_slot(directory, count):
    """Process-shared creation slots; kernel releases locks on runner death."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    held = None
    try:
        while held is None:
            for index in range(count):
                handle = (directory / f'{index}.lock').open('a')
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held = handle
                    break
                except BlockingIOError:
                    handle.close()
            if held is None:
                await asyncio.sleep(0.2)
        yield
    finally:
        if held is not None:
            held.close()


class ManagedOpenSandboxEnvironment(OpenSandboxEnvironment):
    def _validate_definition(self):
        # The deployment binding selects the built image without rewriting task files.
        self.task_env_config = self.task_env_config.model_copy(update={'docker_image': self.binding['image']})
        super()._validate_definition()

    @classmethod
    def validate_config(cls, cfg):
        if cfg['options'] or any(b['options'] for b in cfg['bindings']):
            raise ValueError('standard OpenSandbox adapter does not accept platform options')

    @classmethod
    def agent_gateway(cls, cfg, gateway_url):
        return gateway_url

    @classmethod
    async def network_check(cls, cfg):
        return {'status': 'not_probed', 'reason': 'Use the explicit sandbox smoke test for allocation checks'}

    async def validate_binding(self):
        pass

    def extensions(self):
        return {}

    def api_params(self):
        return {}

    def startup_transport(self, allocated):
        from harborrl.rollout.harbor_job.sandbox_transport import (
            SandboxStartupTransport,
        )
        return SandboxStartupTransport(self.cfg['domain'].rstrip('/') + '/v1', allocated,
                                       on_request=self.record_request)

    def __init__(self, *args, sandbox_config, binding, evidence_dir, startup_dir, gateway_url=None, **kwargs):
        self.validate_config(sandbox_config)
        self.cfg, self.binding = sandbox_config, binding
        self.evidence = Path(evidence_dir)
        self.startup_dir = startup_dir
        self.gateway_url = gateway_url
        self._startup_transport = None
        self._startup_reservation = None
        self._startup_requests = 0
        self.record = {'runtime_digest': digest(sandbox_config),
                       'image': binding['image'], 'sandbox_id': None, 'state': 'pending', 'events': []}
        super().__init__(*args, domain=self.cfg['domain'], api_key=read_credential(self.cfg),
                         protocol=self.cfg['protocol'], use_server_proxy=True,
                         ready_timeout_sec=self.cfg['create_timeout_s'],
                         sandbox_timeout_sec=self.cfg['lifecycle_minutes'] * 60,
                         request_timeout_sec=self.cfg['request_timeout_s'], **kwargs)

    def record_state(self, state):
        self.record['state'] = state
        self.record['events'].append({'state': state, 'time': time.time()})
        publish(self.evidence / 'sandbox-events' / f'{len(self.record["events"]):04d}.json', self.record)
        with tempfile.NamedTemporaryFile(mode='w', dir=self.evidence, delete=False) as stream:
            json.dump(self.record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(stream.name, self.evidence / 'sandbox-lifecycle.json')

    def cleanup_required(self, reason):
        publish(self.evidence / 'cleanup-required.json', {
            'sandbox_id': self.record['sandbox_id'], 'image': self.binding['image'],
            'reason': reason, 'runtime_digest': self.record['runtime_digest']})

    def record_request(self, record):
        self._startup_requests += 1
        publish(self.evidence / 'sandbox-requests' / f'{self._startup_requests:04d}.json', record)

    async def reserve_startup(self):
        """The Runner reserves before Harbor starts its environment timeout."""
        if self._startup_reservation is not None:
            raise RuntimeError('Sandbox startup slot already reserved')
        self.record_state('waiting')
        reservation = startup_slot(self.startup_dir, self.cfg['startup_concurrency'])
        await reservation.__aenter__()
        self._startup_reservation = reservation
        try:
            self.record_state('reserved')
        except BaseException:
            await self.release_startup()
            raise

    async def release_startup(self):
        reservation, self._startup_reservation = self._startup_reservation, None
        if reservation is not None:
            await reservation.__aexit__(None, None, None)

    def startup_failed(self, stage, exc):
        """Keep exception classes/status only; SDK messages may contain credentials."""
        causes, seen = [], set()
        while exc is not None and id(exc) not in seen:
            seen.add(id(exc))
            row = {'error_type': type(exc).__name__}
            status = getattr(exc, 'status_code', None)
            if type(status) is int:
                row['http_status'] = status
            causes.append(row)
            exc = exc.__cause__ or exc.__context__
        publish(self.evidence / 'sandbox-startup-failed.json', {
            'stage': stage, 'sandbox_id': self.record['sandbox_id'], 'causes': causes,
            'create_timeout_s': self.cfg['create_timeout_s'],
            'request_timeout_s': self.cfg['request_timeout_s']})

    async def _create_sandbox(self, sdk):
        # Direct environment callers retain the same process-shared limit.
        if self._startup_reservation is None:
            await self.reserve_startup()
        try:
            await self.validate_binding()
            self.record_state('creating')
            def allocated(sandbox_id):
                self.record['sandbox_id'] = sandbox_id
                self.record_state('allocated')

            self._startup_transport = self.startup_transport(allocated)
            connection = self._build_connection_config(sdk).model_copy(update={'transport': self._startup_transport})
            try:
                sandbox = await asyncio.wait_for(sdk['Sandbox'].create(
                    self.binding['image'], connection_config=connection,
                    timeout=timedelta(minutes=self.cfg['lifecycle_minutes']),
                    ready_timeout=timedelta(seconds=self.cfg['create_timeout_s']),
                    extensions=self.extensions(),
                    resource=self._build_resource(), env=dict(self._persistent_env),
                    network_policy=self._build_network_policy(sdk), skip_health_check=True),
                    self.cfg['create_timeout_s'])
            except BaseException as exc:  # noqa: BLE001 - persist unknown allocation on cancellation too
                self.startup_failed('create', exc)
                if self.record['sandbox_id']:
                    await self._cleanup_allocated()
                else:
                    self.record_state('creation_unknown')
                    self.cleanup_required('Creation outcome unknown; reconcile platform inventory before retry')
                await self._startup_transport.aclose()
                raise RuntimeError('Sandbox creation outcome unknown') from None
            self._sandbox = sandbox
            self.record['sandbox_id'] = sandbox.id
            self.record_state('created')
            try:
                await asyncio.wait_for(self._wait_until_ready(sandbox), self.cfg['create_timeout_s'])
                self.record_state('ready')
                return sandbox
            except BaseException as exc:  # noqa: BLE001 - cleanup before redacting SDK errors
                self.startup_failed('readiness', exc)
                await self.stop(delete=True)
                raise RuntimeError('Sandbox readiness failed') from None
        finally:
            await self.release_startup()

    async def _cleanup_allocated(self):
        """The SDK may already have killed an instance whose endpoints failed."""
        self.record_state('deleting')
        url = self.cfg['domain'].rstrip('/') + '/v1/sandboxes/' + self.record['sandbox_id']
        try:
            async with asyncio.timeout(self.cfg['cleanup_timeout_s']):
                async with httpx.AsyncClient(timeout=30, headers={'OPEN-SANDBOX-API-KEY': self._api_key},
                                             params=self.api_params()) as client:
                    status = await client.get(url)
                    if status.status_code != 404:
                        (await client.delete(url)).raise_for_status()
                        status = await client.get(url)
                    if status.status_code != 404:
                        raise RuntimeError('allocation deletion unconfirmed')
            self.record_state('deleted')
        except BaseException:  # noqa: BLE001 - retain debt if cleanup is interrupted
            self.record_state('delete_failed')
            self.cleanup_required('Allocated Sandbox deletion was not confirmed')

    async def start(self, force_build):
        try:
            await super().start(force_build)
        except BaseException:
            if self._sandbox is not None:
                await self.stop(delete=True)
            raise
        finally:
            await self.release_startup()

    async def stop(self, delete=True):
        if self._sandbox is None:
            if self._startup_transport is not None:
                await self._startup_transport.aclose()
            return
        sandbox = self._sandbox
        self.record_state('deleting')
        try:
            await asyncio.wait_for(sandbox.kill(), self.cfg['cleanup_timeout_s'])
            self.record_state('deleted')
            (self.evidence / 'cleanup-required.json').unlink(missing_ok=True)
            self._sandbox = None
        except BaseException:  # noqa: BLE001 - preserve cleanup debt on cancellation
            self.record_state('delete_failed')
            self.cleanup_required('Sandbox kill was not acknowledged')
            raise RuntimeError('Sandbox deletion failed') from None
        finally:
            try:
                await asyncio.wait_for(sandbox.close(), 10)
            except Exception:  # noqa: BLE001 - close errors must not expose SAT headers
                self.logger.warning('Sandbox SDK connection close failed')
            if self._startup_transport is not None:
                await self._startup_transport.aclose()
