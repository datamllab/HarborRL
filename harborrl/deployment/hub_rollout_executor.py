"""Rollout task executor for hub-mode workers.

The hub server deliberately holds no Gateway and no RolloutRuntime, so a
pulled rollout task is executed here on the worker:

  1. start a local Gateway + TraceRegistry for this task,
  2. register the attempt to obtain its credential,
  3. hand the runner a request whose gateway_url points at that gateway,
  4. speak the runner's stdin/stdout protocol (prepared -> run -> terminal).

The gateway must be reachable from inside the sandbox, so the advertised
origin is composed from --gateway-host (defaults to the local IP) and the
ephemeral port the server actually bound.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import socket
import threading
from pathlib import Path

from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry
from harborrl.trajectories.native import Identity, RewardProfile, digest, publish


def local_ip():
    """Best-effort address the sandbox can dial back to."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("10.255.255.255", 1))
            return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def sampling_digest_for(max_output_tokens, stop=None):
    """Digest the same sampling structure the gateway's convert() produces.

    The gateway compares this digest against the converted request, so the
    recipe here must match messages.convert() exactly: temperature=1,
    top_p=1, top_k=-1 are enforced for the initial profile, max_new_tokens
    is the profile output budget, and stop is the caller's sequence list
    (empty when unspecified).
    """
    return digest({
        # floats on purpose: messages.convert() defaults temperature/top_p to
        # 1.0, and digest() hashes JSON text, so 1 and 1.0 hash differently.
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": max_output_tokens,
        "stop": list(stop or []),
    })


class RolloutExecutor:
    """Run one rollout group request against a locally hosted gateway."""

    def __init__(self, *, run_root, task_path, profile, runner_python, source_dir,
                 gateway_host=None, tracing=True, log=print):
        self.run_root = Path(run_root)
        self.task_path = Path(task_path)
        self.profile = profile
        self.runner_python = runner_python
        self.source_dir = str(source_dir)
        self.gateway_host = gateway_host or local_ip()
        self.tracing = tracing
        self.log = log

    def execute(self, group_request):
        """Execute one group request; returns a result dict for the hub."""
        task_id = group_request.get('task_id') or 'rollout_' + secrets.token_hex(8)
        run_id = group_request.get('run_id', 'adhoc')
        group_id = group_request.get('group_id', task_id)
        slot_id = str(group_request.get('slot_id', '0'))

        attempt_id = f'{task_id}-{slot_id}'
        attempt_root = self.run_root / attempt_id
        attempt_root.mkdir(parents=True, exist_ok=True)

        trace_root = attempt_root / 'traces'
        registry = TraceRegistry(trace_root)

        model_tag = group_request.get('model_tag', 'qwen3-8B')
        from harborrl.math_registry import MATH_REGISTRY
        model = MATH_REGISTRY.model_for_preset(model_tag)

        backend = group_request.get('backend')
        if backend is None and group_request.get('synthetic'):
            # Transport-only verification: generate without a serving pool.
            from .hub_synthetic_backend import SyntheticServingBackend
            backend = SyntheticServingBackend(
                context=group_request.get('max_context', 32768))
        if backend is None:
            # A live rollout needs a serving pool; the hub does not carry one,
            # so the caller must supply an endpoint for real generation.
            endpoint = group_request.get('serving_endpoint')
            if not endpoint:
                raise ValueError(
                    'rollout execution requires serving_endpoint (or an injected backend)')
            # The tokenizer cannot cross JSON, so load it locally from the
            # checkpoint and verify it still matches the run's lock digests.
            from transformers import AutoTokenizer
            ckpt = group_request.get('hf_ckpt') or self.hf_ckpt
            if not ckpt:
                raise ValueError('live rollout requires hf_ckpt to load the tokenizer')
            tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
            from harborrl.rollout.runtime import tokenizer_fingerprint
            tok_digest, tpl_digest = tokenizer_fingerprint(tokenizer)
            expect_tok = group_request.get('tokenizer_digest')
            expect_tpl = group_request.get('template_digest')
            if expect_tok and tok_digest != expect_tok:
                raise ValueError(f'tokenizer digest mismatch: {tok_digest} != {expect_tok}')
            if expect_tpl and tpl_digest != expect_tpl:
                raise ValueError(f'template digest mismatch: {tpl_digest} != {expect_tpl}')
            from harborrl.gateway.sglang_backend import SGLangBackend
            backend = SGLangBackend(
                endpoint, tokenizer, model.adapter,
                tokenizer_digest=tok_digest, template_digest=tpl_digest,
                max_context=group_request.get('max_context', 32768))

        gateway = Gateway(registry, backend, model=self.profile.get('model', 'policy'),
                          max_output_tokens=group_request.get('max_output_tokens', 8192))
        server = make_server(gateway, host='0.0.0.0', port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True,
                                  name=f'hub-gateway-{task_id}')
        thread.start()

        advertised = f'http://{self.gateway_host}:{port}'
        self.log(f'[ROLLOUT] gateway up at {advertised} (attempt {attempt_id})')

        try:
            identity = Identity(
                run_id=run_id,
                batch_id=group_request.get('batch_id', 'batch'),
                group_id=group_id,
                slot_id=slot_id,
                attempt_id=attempt_id,
                trajectory_id=group_request.get('trajectory_id', f'traj-{slot_id}'),
                task_digest=group_request.get('task_digest', ''),
                policy_version=group_request.get('policy_version', '1'),
                harness_digest=digest(self.profile),
                reward_digest=RewardProfile().digest,
                sampling_digest=group_request.get('sampling_digest')
                or sampling_digest_for(group_request.get('max_output_tokens', 8192),
                                       group_request.get('stop')),
            )
            # register() mints the attempt credential; the runner must hand
            # exactly this token to the sandbox or the gateway rejects it.
            attempt_credential = registry.register(identity, task_id)

            request = {
                'identity': identity.to_dict(),
                'task_path': str(self.task_path),
                'profile': self.profile,
                'gateway_url': advertised,
                'credential': attempt_credential,
                'output': str(attempt_root / 'runner'),
                'max_output_tokens': group_request.get('max_output_tokens', 8192),
            }

            terminal = asyncio.run(
                self._drive_runner(request, attempt_root))
            publish(attempt_root / 'terminal.json', terminal)
            return {'success': terminal.get('type') == 'terminal', 'terminal': terminal,
                    'attempt_root': str(attempt_root)}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    async def _drive_runner(self, request, attempt_root):
        """Speak the runner stdin/stdout protocol to completion."""
        env = {**__import__('os').environ}
        env['PYTHONPATH'] = self.source_dir
        process = await asyncio.create_subprocess_exec(
            self.runner_python, '-m', 'harborrl.rollout.harbor_job.runner',
            cwd=self.source_dir, env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)

        process.stdin.write(json.dumps(request).encode() + b'\n')
        await process.stdin.drain()

        terminal = None
        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    self.log(f'[ROLLOUT] non-JSON line: {line[:200]!r}')
                    continue
                kind = event.get('type')
                self.log(f'[ROLLOUT] runner event: {kind}')
                if kind == 'prepared':
                    process.stdin.write(b'{"command":"run"}\n')
                    await process.stdin.drain()
                elif kind in ('terminal', 'cancelled'):
                    terminal = event
                    break
        finally:
            with __import__('contextlib').suppress(Exception):
                await asyncio.wait_for(process.wait(), timeout=60)
            stderr = b''
            if process.stderr is not None:
                with __import__('contextlib').suppress(Exception):
                    stderr = await process.stderr.read()
            if stderr:
                publish(attempt_root / 'runner.stderr.txt',
                        stderr.decode('utf-8', 'replace')[-8000:])

        if terminal is None:
            raise RuntimeError('runner exited before emitting a terminal record')
        return terminal
