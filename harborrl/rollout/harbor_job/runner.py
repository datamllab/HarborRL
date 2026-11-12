"""Isolated Python >=3.12 worker using pinned Harbor public Trial API.

stdin: submit request, then {"command":"run"}, optionally {"command":"cancel"}.
stdout: JSON prepared/events/terminal only. No credentials in emitted records.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import importlib.metadata
import json
import signal
import sys
from pathlib import Path

from harborrl.math_registry import CLAUDE_CODE_HARNESS, NATIVE_TASK_PROVIDER
from harborrl.rollout.harbor_job.bindings import environment_timeouts
from harborrl.trajectories.native import Identity, digest, publish


def probe():
    if sys.version_info < (3, 12):
        raise RuntimeError("native Harbor Runner requires Python >=3.12")
    version = importlib.metadata.version("harbor")
    if version != "0.23.0":
        raise RuntimeError("native Runner requires Harbor 0.23.0")
    from harbor.agents.installed.claude_code import ClaudeCodeOptions
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.trial import Trial
    return {"harbor_version": version, "python": sys.version.split()[0],
            "claude_options": sorted(ClaudeCodeOptions.model_fields),
            "trial_factory": callable(Trial.create), "trial_config_fields": sorted(TrialConfig.model_fields)}


async def execute(request, emit, read_command):
    probe()
    from harbor.models.trial.config import TrialConfig
    from harbor.trial.hooks import TrialEvent
    from harbor.trial.trial import Trial

    identity = Identity(**request["identity"])
    if digest(request["profile"]) != identity.harness_digest:
        raise ValueError("harness profile differs from attempt manifest")
    inspection = NATIVE_TASK_PROVIDER.adapter.inspect(
        request["task_path"], expected_digest=identity.task_digest
    )
    if inspection["status"] != "NEEDS_PROBE":
        raise ValueError("native task preflight failed: " + ",".join(inspection["reasons"]))
    root = Path(request["output"]).resolve()
    # Never let a restarted process overwrite another attempt's Harbor result.
    root.mkdir(parents=True, exist_ok=False)
    publish(root / "manifest.json", {"identity": identity.to_dict(), "task": inspection,
                                     "profile": request["profile"], "gateway_url": request["gateway_url"]})
    sandbox = request.get('sandbox')
    agent_gateway = request['gateway_url']
    if sandbox is not None:
        from harborrl.rollout.harbor_job.sandbox_config import environment_class
        agent_gateway = environment_class(sandbox).agent_gateway(sandbox, agent_gateway)
    agent = CLAUDE_CODE_HARNESS.adapter.agent_config(
        request['profile'], agent_gateway, request['credential'], request['max_output_tokens'])
    environment = {'type': 'docker', 'delete': True}
    if sandbox is not None:
        from harborrl.rollout.harbor_job.sandbox_config import task_binding
        if digest(sandbox) != request.get('runtime_digest'):
            raise ValueError('sandbox runtime lock mismatch')
        if sandbox['installation'] and sandbox['installation']['version'] != request['profile']['cli_version']:
            raise ValueError('offline harness version mismatch')
        publish(root / 'runtime.json', {'config': sandbox, 'digest': digest(sandbox)})
        environment = {'type': 'opensandbox', 'delete': True,
                       'import_path': sandbox['adapter'],
                       'kwargs': {'sandbox_config': sandbox, 'binding': task_binding(sandbox, request['task_path']),
                                  'evidence_dir': str(root),
                                  'gateway_url': request['gateway_url'],
                                  'startup_dir': str(root.parents[2] / '.sandbox-startup')}}
        if sandbox['installation']:
            agent.pop('name')
            agent['import_path'] = 'harborrl.harnesses.offline_claude:OfflineClaudeCode'
            agent['kwargs']['installation'] = sandbox['installation']
    config = TrialConfig.model_validate({
        "task": {"path": inspection["path"]},
        "trial_name": digest(identity.to_dict()), "trials_dir": str(root / "harbor"),
        "agent": agent,
        "environment": environment,
        **environment_timeouts(request['profile'], inspection['resources']),
        "verifier": {"disable": False, "override_timeout_sec": request["profile"]["verifier_timeout_sec"]}})
    trial = await Trial.create(config)
    trial_id = str(trial.id)
    publish(root / "binding.json", {"identity": identity.to_dict(), "harbor_trial_id": trial_id})
    emit({"type": "prepared", "trial_id": trial_id})
    if (await read_command()).get("command") != "run":
        raise ValueError("expected run handshake before starting trial")

    async def event_hook(event):
        if sandbox is not None and event.event == TrialEvent.ENVIRONMENT_START:
            # Harbor awaits this public hook before starting its environment timer.
            await trial.agent_environment.reserve_startup()
        record = {"type": "event", "event": event.event.value, "trial_id": str(event.trial_id),
                  "timestamp": event.timestamp.isoformat()}
        # Each event is an independent atomic record; never serialize config credentials.
        publish(root / "events" / (event.event.value + ".json"), record)
        emit(record)

    for event in TrialEvent:
        trial.add_hook(event, event_hook)

    async def run():
        try:
            return await trial.run()
        finally:
            if sandbox is not None:
                await trial.agent_environment.release_startup()

    running = asyncio.create_task(run())
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, running.cancel)

    async def monitor():
        while not running.done():
            command = await read_command()
            if command.get("command") in ("cancel", "eof"):
                running.cancel()
                return
            running.cancel()
            return

    monitoring = asyncio.create_task(monitor())
    try:
        try:
            result = await running
            state = "terminal"
        except asyncio.CancelledError:
            result = trial.result
            state = "cancelled"
        record = {"type": state, "trial_id": trial_id,
                  "result_path": str(trial.paths.result_path), "trial_dir": str(trial.paths.trial_dir),
                  "exception": result.exception_info.exception_type if result.exception_info else None}
        publish(root / "terminal.json", record)
        emit(record)
    finally:
        monitoring.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitoring
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)


async def worker():
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    async def read():
        line = await reader.readline()
        return json.loads(line) if line else {"command": "eof"}
    def emit(value):
        sys.__stdout__.write(json.dumps(value) + "\n")
        sys.__stdout__.flush()
    first = await read()
    with contextlib.redirect_stdout(sys.stderr):
        await execute(first, emit, read)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(probe()))
    else:
        asyncio.run(worker())


if __name__ == "__main__":
    main()
