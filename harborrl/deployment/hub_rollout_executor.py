"""Rollout task executor for hub-mode workers.

The hub server deliberately holds no Gateway and no RolloutRuntime, so a
pulled rollout task is executed here on the worker:

  1. start a local Gateway + TraceRegistry for this attempt,
  2. delegate the full trial to harbor_job.coordinator.run_attempt (which
     registers the attempt credential, drives the runner protocol, audits
     the session and assembles the trajectory IR),
  3. publish the IR under the attempt root (shared storage) and report
     {success, ir_path, ir_sha256, reward} back to the hub.

The gateway must be reachable from inside the sandbox, so the advertised
origin is composed from --gateway-host (defaults to the local IP) and the
ephemeral port the server actually bound.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import threading
from pathlib import Path

from harborrl.gateway.server import Gateway, make_server
from harborrl.gateway.trace import TraceRegistry


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
    from harborrl.trajectories.native import digest
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
    """Run one rollout attempt request against a locally hosted gateway."""

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
        """Execute one attempt request; returns a result dict for the hub."""
        import asyncio

        shared_root = os.environ.get('HARBORRL_HUB_ATTEMPT_MIRROR', '').strip()
        if not shared_root or not Path(shared_root).is_absolute():
            raise ValueError(
                'HARBORRL_HUB_ATTEMPT_MIRROR must be an absolute shared directory '
                'visible at the same path to workers and trainer')

        task_id = group_request.get('task_id') or 'rollout_' + __import__('secrets').token_hex(8)
        run_id = group_request.get('run_id', 'adhoc')
        group_id = group_request.get('group_id', task_id)
        slot_id = str(group_request.get('slot_id', '0'))

        attempt_id = f'{task_id}-{slot_id}'
        # run_attempt creates the attempt root itself (exist_ok=False) and
        # must stay its only owner; pre-creating it here made every retry
        # fail with FileExistsError.
        attempt_root = self.run_root / attempt_id

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
            ckpt = group_request.get('hf_ckpt')
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
            audited = group_request.get('audited_raw_logprobs', False)
            if type(audited) is not bool:
                raise ValueError('audited_raw_logprobs must be boolean')
            backend = SGLangBackend(
                endpoint, tokenizer, model.adapter,
                tokenizer_digest=tok_digest, template_digest=tpl_digest,
                max_context=group_request.get('max_context', 32768),
                audited_raw_logprobs=audited)

        max_output_tokens = group_request.get('max_output_tokens', 8192)
        gateway = Gateway(registry, backend, model=self.profile.get('model', 'policy'),
                          max_output_tokens=max_output_tokens)
        server = make_server(gateway, host='0.0.0.0', port=0)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True,
                                  name=f'hub-gateway-{task_id}')
        thread.start()

        advertised = f'http://{self.gateway_host}:{port}'
        self.log(f'[ROLLOUT] gateway up at {advertised} (attempt {attempt_id})')

        try:
            from harborrl.rollout.harbor_job.coordinator import run_attempt
            from harborrl.trajectories.native import Identity, RewardProfile, digest

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
                or sampling_digest_for(max_output_tokens, group_request.get('stop')),
            )

            sandbox = group_request.get('sandbox') or None
            reward_profile = RewardProfile(
                **group_request['reward_profile']) if group_request.get('reward_profile') else RewardProfile()

            ir = asyncio.run(run_attempt(
                identity,
                task_path=str(self.task_path),
                profile=self.profile,
                gateway_url=advertised,
                runner_python=self.runner_python,
                root=attempt_root,
                registry=registry,
                reward_profile=reward_profile,
                max_output_tokens=max_output_tokens,
                sandbox=sandbox,
            ))
            ir_path = attempt_root / 'trajectory.v2.json'
            ir_sha = digest(ir)
            reward = ir.get('evaluation', {}).get('training_reward')
            # The attempt runs on the worker's local disk (docker cannot bind
            # mount through the sshfs share); mirror the IR to shared storage
            # so the trainer can load and verify it, and report that path.
            mirrored = Path(shared_root) / attempt_id
            try:
                mirrored.mkdir(parents=True, exist_ok=True)
                for name in ('trajectory.v2.json', 'evaluation.json', 'terminal.json'):
                    src = attempt_root / name
                    if src.is_file():
                        shutil.copy2(src, mirrored / name)
                shared_ir = mirrored / 'trajectory.v2.json'
                if digest(json.loads(shared_ir.read_text(encoding='utf-8'))) != ir_sha:
                    raise ValueError('mirrored IR digest mismatch')
                ir_path = shared_ir
            except OSError as exc:
                raise RuntimeError(f'failed to mirror IR to shared storage: {exc}')
            return {'success': True,
                    'ir_path': str(ir_path), 'ir_sha256': ir_sha,
                    'reward': reward, 'readiness': ir.get('readiness'),
                    'integrity_errors': ir.get('integrity_errors'),
                    'attempt_root': str(attempt_root)}
        finally:
            server.shutdown()
            server.close() if hasattr(server, 'close') else server.server_close()
            thread.join(5)
