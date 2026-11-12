"""Hub client pullers for GPU training nodes and rollout worker nodes.

GPU Puller: runs on GPU nodes, pulls training tasks from hub manager.
Rollout Puller: runs on worker nodes, pulls rollout tasks from hub manager.

Both implement exponential backoff retry. The contract below mirrors
harborrl/deployment/hub_server.py exactly:

    GET  /pull/gpu_task        header X-GPU-ID      -> 200 {task_id, run_id, plan, deployment} | 204
    GET  /pull/rollout_task    header X-Worker-ID   -> 200 {task_id, run_id, group_id, request} | 204
    POST /report/gpu_status    {gpu_id, task_id, status: {success, result|error}}
    POST /report/rollout_result {worker_id, task_id, result}

Authentication is a single header: X-Api-Key (never combined with Authorization).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

# Exponential backoff parameters
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 60.0
BACKOFF_MULTIPLIER = 2.0
POLL_INTERVAL = 5.0

# Timeout settings
TASK_PULL_TIMEOUT = 30
STATUS_REPORT_TIMEOUT = 15


@dataclass
class PullerConfig:
    """Common configuration for hub pullers."""
    hub_url: str
    node_id: str
    node_type: str  # 'gpu' or 'rollout'
    token: str
    working_dir: str = None
    log_dir: str = None
    # rollout pullers need somewhere to send model traffic; the hub holds no
    # gateway of its own (see hub_server docstring), so the worker brings it.
    gateway_url: str = None
    runner_python: str = None
    source_dir: str = None
    profile_path: str = None
    task_path: str = None
    gateway_host: str = None


class HubClientError(Exception):
    """Base exception for hub client errors."""


class TaskExecutionError(HubClientError):
    """Raised when task execution fails."""


def _http_request(url, data=None, *, token=None, timeout=30, method=None, headers=None):
    """Make an HTTP request carrying exactly one authentication header."""
    hdrs = {'Content-Type': 'application/json'}
    if headers:
        hdrs.update(headers)
    if token:
        # Server rejects requests presenting both x-api-key and Authorization.
        hdrs['X-Api-Key'] = token

    body = json.dumps(data).encode('utf-8') if data is not None else None
    req = Request(url, data=body, headers=hdrs, method=method)

    try:
        with urlopen(req, timeout=timeout) as response:
            content = response.read().decode('utf-8')
            return json.loads(content) if content else {}
    except HTTPError as exc:
        if exc.code == 204:
            return {}
        error_body = exc.read().decode('utf-8', errors='replace')
        try:
            error_data = json.loads(error_body)
            message = error_data.get('error', error_body)
        except json.JSONDecodeError:
            message = error_body
        raise HubClientError(f"HTTP {exc.code}: {message}") from exc
    except URLError as exc:
        raise HubClientError(f"Network error: {exc.reason}") from exc


class BasePuller:
    """Base class for hub client pullers with common retry logic."""

    def __init__(self, config: PullerConfig):
        self.config = config
        self.running = False
        self.current_task = None
        self.task_lock = threading.RLock()
        self._backoff = INITIAL_BACKOFF

    def start(self):
        """Run the task polling loop until stopped."""
        self.running = True
        label = self.config.node_type.upper()
        print(f"[{label}] Puller started: {self.config.node_id}")
        print(f"[{label}] Hub URL: {self.config.hub_url}")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print(f"\n[{label}] Shutting down...")
        finally:
            self.running = False

    def _main_loop(self):
        """Main task polling loop with exponential backoff."""
        label = self.config.node_type.upper()
        while self.running:
            try:
                task = self._pull_task()
                if task:
                    self._backoff = INITIAL_BACKOFF  # Reset backoff on success
                    self._execute_task(task)
                else:
                    # No task available, back off
                    time.sleep(self._backoff)
                    self._backoff = min(self._backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 - loop must survive task errors
                print(f"[{label}] Error in main loop: {exc}", flush=True)
                time.sleep(self._backoff)
                self._backoff = min(self._backoff * BACKOFF_MULTIPLIER, MAX_BACKOFF)

    def _pull_task(self):
        """Pull next available task from the hub. Implemented per node type."""
        raise NotImplementedError

    def _execute_task(self, task):
        """Execute a task. Implemented per node type."""
        raise NotImplementedError

    def _stream_process(self, cmd, *, cwd, env, task_id, label):
        """Run cmd, stream its output, and return the exit code."""
        log_file = None
        if self.config.log_dir:
            log_path = Path(self.config.log_dir) / f'{task_id}.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open('w')

        process = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(f"[{label}:{task_id}] {line}", end='', flush=True)
                if log_file:
                    log_file.write(line)
                    log_file.flush()
            return process.wait()
        finally:
            if log_file:
                log_file.close()


class GpuPuller(BasePuller):
    """GPU training node puller for executing training jobs."""

    def __init__(self, config: PullerConfig):
        super().__init__(config)
        if config.node_type != 'gpu':
            raise ValueError("GpuPuller requires node_type='gpu'")

    def _pull_task(self):
        """GET /pull/gpu_task with the X-GPU-ID header."""
        url = urljoin(self.config.hub_url, '/pull/gpu_task')
        try:
            response = _http_request(
                url, token=self.config.token, timeout=TASK_PULL_TIMEOUT,
                method='GET', headers={'X-GPU-ID': self.config.node_id})
        except HubClientError as exc:
            if '404' in str(exc) or '503' in str(exc):
                return None
            raise
        if response and response.get('task_id'):
            return response
        return None

    def _report(self, task_id, *, success, result=None, error=None):
        """POST /report/gpu_status."""
        status = {'success': bool(success)}
        if success:
            status['result'] = result
        else:
            status['error'] = error or 'unknown error'
        url = urljoin(self.config.hub_url, '/report/gpu_status')
        payload = {
            'gpu_id': self.config.node_id,
            'task_id': task_id,
            'status': status,
        }
        try:
            _http_request(url, payload, token=self.config.token,
                          timeout=STATUS_REPORT_TIMEOUT, method='POST')
        except HubClientError as exc:
            print(f"[GPU] Status report failed: {exc}", flush=True)

    def _execute_task(self, task):
        """Execute one training task on this GPU node."""
        task_id = task['task_id']
        run_id = task.get('run_id')
        print(f"[GPU] Starting training task: {task_id} (run: {run_id})", flush=True)

        with self.task_lock:
            self.current_task = task_id

        try:
            deployment = task.get('deployment') or {}
            plan = task.get('plan') or {}
            gpu_target = deployment.get('gpu_target') or {}

            working_dir = Path(self.config.working_dir or '/tmp') / f'task-{task_id}'
            working_dir.mkdir(parents=True, exist_ok=True)
            deployment_file = working_dir / 'deployment.json'
            plan_file = working_dir / 'plan.json'
            deployment_file.write_text(json.dumps(deployment, indent=2))
            plan_file.write_text(json.dumps(plan, indent=2))

            env = os.environ.copy()
            if task.get('environment'):
                env.update(task['environment'])
            env['HARBORRL_DEPLOYMENT_FILE'] = str(deployment_file)
            env['HARBORRL_PLAN_FILE'] = str(plan_file)

            train_python = gpu_target.get('train_python') or sys.executable
            source_dir = gpu_target.get('source_dir') or self.config.source_dir or os.getcwd()
            train_module = task.get('train_module', 'harborrl.platform.native_train')

            cmd = [train_python, '-m', train_module, str(plan_file)]
            print(f"[GPU] Executing: {' '.join(cmd)}", flush=True)
            print(f"[GPU] Working directory: {source_dir}", flush=True)

            returncode = self._stream_process(
                cmd, cwd=source_dir, env=env, task_id=task_id, label='GPU')

            if returncode == 0:
                print(f"[GPU] Task completed successfully: {task_id}", flush=True)
                self._report(task_id, success=True, result={'exit_code': 0})
            else:
                print(f"[GPU] Task failed with exit code {returncode}: {task_id}", flush=True)
                self._report(task_id, success=False,
                             error=f'Training exited with code {returncode}')
        except Exception as exc:  # noqa: BLE001 - report every failure to the hub
            print(f"[GPU] Task execution error: {exc}", flush=True)
            self._report(task_id, success=False, error=str(exc))
            raise TaskExecutionError(f"Training task failed: {exc}") from exc
        finally:
            with self.task_lock:
                self.current_task = None


class RolloutPuller(BasePuller):
    """Rollout worker node puller for executing Harbor trials.

    The hub server carries no gateway and no RolloutRuntime, so execution
    happens here: the worker spawns the existing runner module and speaks its
    stdin/stdout JSON protocol, using the gateway this node was pointed at.
    """

    def __init__(self, config: PullerConfig):
        super().__init__(config)
        if config.node_type != 'rollout':
            raise ValueError("RolloutPuller requires node_type='rollout'")

    def _pull_task(self):
        """GET /pull/rollout_task with the X-Worker-ID header."""
        url = urljoin(self.config.hub_url, '/pull/rollout_task')
        try:
            response = _http_request(
                url, token=self.config.token, timeout=TASK_PULL_TIMEOUT,
                method='GET', headers={'X-Worker-ID': self.config.node_id})
        except HubClientError as exc:
            if '404' in str(exc) or '503' in str(exc):
                return None
            raise
        if response and response.get('task_id'):
            return response
        return None

    def _report(self, task_id, *, success, result=None, error=None):
        """POST /report/rollout_result."""
        payload_result = {'success': bool(success)}
        if success:
            payload_result['result'] = result
        else:
            payload_result['error'] = error or 'unknown error'
        url = urljoin(self.config.hub_url, '/report/rollout_result')
        payload = {
            'worker_id': self.config.node_id,
            'task_id': task_id,
            'result': payload_result,
        }
        try:
            _http_request(url, payload, token=self.config.token,
                          timeout=STATUS_REPORT_TIMEOUT, method='POST')
        except HubClientError as exc:
            print(f"[ROLLOUT] Result report failed: {exc}", flush=True)

    def _execute_task(self, task):
        """Execute one rollout task by hosting a gateway and driving the runner."""
        from harborrl.deployment.hub_rollout_executor import RolloutExecutor

        task_id = task['task_id']
        group_request = dict(task.get('request') or {})
        group_request.setdefault('task_id', task_id)
        group_request.setdefault('run_id', task.get('run_id'))
        group_request.setdefault('group_id', task.get('group_id'))
        print(f"[ROLLOUT] Starting rollout task: {task_id}", flush=True)

        with self.task_lock:
            self.current_task = task_id

        try:
            task_root = Path(self.config.working_dir or '/tmp') / f'task-{task_id}'
            task_root.mkdir(parents=True, exist_ok=True)

            profile_path = self.config.profile_path
            if not profile_path:
                raise TaskExecutionError('rollout puller requires --profile')
            profile = json.loads(Path(profile_path).read_text())

            task_path = group_request.get('task_path') or self.config.task_path
            if not task_path:
                raise TaskExecutionError('rollout task carries no task_path')

            executor = RolloutExecutor(
                run_root=task_root,
                task_path=task_path,
                profile=profile,
                runner_python=self.config.runner_python or sys.executable,
                source_dir=self.config.source_dir or os.getcwd(),
                gateway_host=self.config.gateway_host,
                log=lambda m: print(m, flush=True))
            result = executor.execute(group_request)

            self._report(task_id, success=bool(result.get('success')),
                         result=result,
                         error=None if result.get('success') else 'runner reported a failed terminal')
        except Exception as exc:  # noqa: BLE001 - report every failure to the hub
            print(f"[ROLLOUT] Task execution error: {exc}", flush=True)
            self._report(task_id, success=False, error=str(exc))
            raise TaskExecutionError(f"Rollout task failed: {exc}") from exc
        finally:
            with self.task_lock:
                self.current_task = None


def main():
    """CLI entry point for pullers."""
    import argparse

    parser = argparse.ArgumentParser(description='HarborRL Hub Client Puller')
    parser.add_argument('--hub-url', required=True,
                        help='Hub manager URL (e.g., http://manager:49300)')
    parser.add_argument('--node-id', required=True,
                        help='Unique node identifier')
    parser.add_argument('--node-type', required=True, choices=['gpu', 'rollout'],
                        help='Node type: gpu or rollout')
    parser.add_argument('--token-file', required=True,
                        help='Path to hub authentication token file')
    parser.add_argument('--working-dir',
                        help='Working directory for task execution')
    parser.add_argument('--log-dir',
                        help='Directory for task execution logs')
    parser.add_argument('--gateway-url',
                        help='Gateway origin for rollout tasks (required for rollout node)')
    parser.add_argument('--runner-python',
                        help='Interpreter that runs harborrl.rollout.harbor_job.runner')
    parser.add_argument('--source-dir',
                        help='Checkout that contains the harborrl package')
    parser.add_argument('--profile',
                        help='Claude-code profile JSON (required for rollout node)')
    parser.add_argument('--task-path',
                        help='Default Harbor task path used when the task omits one')
    parser.add_argument('--gateway-host',
                        help='Address the sandbox uses to reach this worker gateway')

    args = parser.parse_args()

    token_path = Path(args.token_file)
    if not token_path.is_file():
        print(f"Error: Token file not found: {args.token_file}", file=sys.stderr)
        sys.exit(1)

    token = token_path.read_text().strip()
    if not token:
        print(f"Error: Token file is empty: {args.token_file}", file=sys.stderr)
        sys.exit(1)

    config = PullerConfig(
        hub_url=args.hub_url,
        node_id=args.node_id,
        node_type=args.node_type,
        token=token,
        working_dir=args.working_dir,
        log_dir=args.log_dir,
        gateway_url=args.gateway_url,
        runner_python=args.runner_python,
        source_dir=args.source_dir,
        profile_path=args.profile,
        task_path=args.task_path,
        gateway_host=args.gateway_host,
    )

    if args.node_type == 'gpu':
        puller = GpuPuller(config)
    else:
        puller = RolloutPuller(config)

    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        puller.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        puller.start()
    except Exception as exc:  # noqa: BLE001 - surface fatal startup errors
        print(f"Fatal error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
