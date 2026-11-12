"""Brainbox-specific settings and relay on the shared audited lifecycle."""
import asyncio
import re
from contextlib import suppress

import httpx

from harborrl.rollout.harbor_job.environments.opensandbox import (
    ManagedOpenSandboxEnvironment,
)
from harborrl.rollout.harbor_job.sandbox_config import read_credential
from harborrl.trajectories.native import publish
from harborrl_h_cluster.transport import SandboxStartupTransport


async def validate_environment(cfg, binding, credential):
    options, target = cfg['options'], binding['options']
    async with httpx.AsyncClient(timeout=cfg['request_timeout_s']) as client:
        response = await client.get(
            cfg['domain'].rstrip('/') + '/v1/sandbox-environments/' + target['environment_id'],
            params={'project': options['project']}, headers={'OPEN-SANDBOX-API-KEY': credential})
    if response.status_code != 200:
        raise RuntimeError(f'Environment query failed: HTTP {response.status_code}')
    environment = response.json()
    if environment.get('image', {}).get('uri') != binding['image']:
        raise ValueError('Environment image does not match task image')
    required_ports = {44772}
    if options['gateway_transport'] == 'platform_pull':
        required_ports.add(options['relay_port'])
    if not required_ports <= {p.get('containerPort') for p in environment.get('ports', [])}:
        raise ValueError('Environment command or relay port missing')
    capacity = environment.get('instanceCapacity')
    if type(capacity) is not int or capacity < binding['max_live_instances']:
        raise ValueError('Environment capacity below configured admission limit')
    mounts = {v.get('mountPath') for v in environment.get('volumes', [])}
    if set(target['required_mount_paths']) - mounts:
        raise ValueError('Environment required mounts missing')
    return {'environment_id': target['environment_id'], 'image': binding['image'], 'capacity': capacity}


class BrainboxEnvironment(ManagedOpenSandboxEnvironment):
    @classmethod
    def validate_config(cls, cfg):
        options = cfg['options']
        if set(options) != {'project', 'gateway_transport', 'relay_port', 'relay_timeout_s'}:
            raise ValueError('H-cluster options require project, gateway_transport, relay_port, relay_timeout_s')
        if not isinstance(options['project'], str) or not options['project'].strip():
            raise ValueError('H-cluster project required')
        if options['gateway_transport'] not in ('direct', 'platform_pull') or cfg['use_server_proxy'] is not True:
            raise ValueError('H-cluster requires server proxy and a supported gateway transport')
        if (type(options['relay_port']) is not int or not 0 < options['relay_port'] < 65536
                or options['relay_port'] == 44772 or type(options['relay_timeout_s']) is not int
                or options['relay_timeout_s'] < 30):
            raise ValueError('invalid H-cluster relay port/timeout')
        seen = set()
        for binding in cfg['bindings']:
            target = binding['options']
            if set(target) != {'environment_id', 'required_mount_paths'}:
                raise ValueError('H-cluster binding requires environment_id and required_mount_paths')
            eid = target['environment_id']
            if not isinstance(eid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', eid) or eid in seen:
                raise ValueError('invalid or duplicate H-cluster Environment ID')
            seen.add(eid)
            if not isinstance(target['required_mount_paths'], list) or any(
                    not isinstance(p, str) or not p.startswith('/') for p in target['required_mount_paths']):
                raise ValueError('invalid required_mount_paths')

    @classmethod
    def agent_gateway(cls, cfg, gateway_url):
        if cfg['options']['gateway_transport'] == 'platform_pull':
            return f"http://127.0.0.1:{cfg['options']['relay_port']}"
        return gateway_url

    @classmethod
    async def network_check(cls, cfg):
        credential = read_credential(cfg)
        if ':' not in credential:
            raise ValueError('H-cluster credential must contain AK:SK')
        return [await validate_environment(cfg, binding, credential) for binding in cfg['bindings']]

    def __init__(self, *args, **kwargs):
        self.relay_task = None
        super().__init__(*args, **kwargs)
        self.record['environment_id'] = self.binding['options']['environment_id']

    async def validate_binding(self):
        if ':' not in self._api_key:
            raise ValueError('H-cluster credential must contain AK:SK')
        await validate_environment(self.cfg, self.binding, self._api_key)

    def extensions(self):
        return {'environmentId': self.binding['options']['environment_id']}

    def api_params(self):
        return {'project': self.cfg['options']['project']}

    def startup_transport(self, allocated):
        return SandboxStartupTransport(self.cfg['domain'].rstrip('/') + '/v1', allocated,
                                       timeout=self.cfg['create_timeout_s'], on_request=self.record_request)

    def _build_resource(self):
        resource = super()._build_resource()
        memory = re.fullmatch(r'(\d+)Mi', resource.get('memory', ''))
        if memory and int(memory[1]) % 1024 == 0:
            resource['memory'] = f'{int(memory[1]) // 1024}Gi'
        return resource

    async def start(self, force_build):
        try:
            await super().start(force_build)
            if self.cfg['options']['gateway_transport'] == 'platform_pull':
                from harborrl_h_cluster.relay import start_relay
                if not self.gateway_url:
                    raise ValueError('platform relay requires manager Gateway URL')
                self.relay_task = await start_relay(self, self.gateway_url)
        except BaseException:
            if self._sandbox is not None:
                await self.stop(delete=True)
            raise

    async def exec(self, *args, **kwargs):
        if self.relay_task is None:
            return await super().exec(*args, **kwargs)
        command = asyncio.create_task(super().exec(*args, **kwargs))
        try:
            done, _ = await asyncio.wait((command, self.relay_task), return_when=asyncio.FIRST_COMPLETED)
            if self.relay_task in done:
                publish(self.evidence / 'relay-failed.json', {'reason': 'manager relay pump stopped'})
                raise RuntimeError('manager relay pump stopped; attempt invalid')
            return await command
        finally:
            if not command.done():
                command.cancel()
                with suppress(asyncio.CancelledError):
                    await command

    async def stop(self, delete=True):
        if self.relay_task is not None:
            task, self.relay_task = self.relay_task, None
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        await super().stop(delete=delete)
