"""Hub Server: Centralized manager for multi-run GPU and rollout task scheduling.

This implements the hub mode control plane, where a single long-running manager
service accepts and queues multiple training runs, managing shared GPU resources
and rollout workers. Unlike service mode (1 manager → 1 run), hub mode supports
multi-tenancy (1 manager → N runs).

Architecture:
- Pull-based task distribution for GPU training and rollout workers
- Shared GPU pool with FIFO/priority scheduling
- Run queue with configurable concurrency limits
- HTTP API for task submission, status queries, and worker pulls
"""
from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

HUB_SCHEMA = "harborrl-hub-service-v1"


class HubError(Exception):
    """Maps to HTTP status codes with machine-readable error codes."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class GpuResource:
    """Represents a single GPU node in the resource pool."""
    id: str
    ssh: dict
    source_dir: str
    train_python: str
    expected_gpus: int
    status: str = "idle"  # idle | busy | reserved
    current_run: Optional[str] = None
    current_task: Optional[str] = None
    assigned_at: Optional[float] = None


@dataclass
class RolloutTask:
    """A rollout task ready to be pulled by workers."""
    task_id: str
    run_id: str
    group_id: str
    request: dict
    status: str = "queued"  # queued | assigned | running | completed | failed
    worker_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class GpuTask:
    """A GPU training task ready to be pulled by GPU workers."""
    task_id: str
    run_id: str
    plan: dict
    deployment: dict
    status: str = "queued"  # queued | assigned | running | completed | failed
    gpu_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class RunInfo:
    """Tracking information for a submitted training run."""
    run_id: str
    plan: dict
    deployment: dict
    status: str = "pending"  # pending | scheduled | running | completed | failed
    gpu_task_id: Optional[str] = None
    rollout_tasks: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    scheduled_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None


class HubManager:
    """Core hub manager coordinating GPU pool and task queues."""

    def __init__(self, *, hub_config: dict, run_root: str, max_concurrent_runs: int = 4,
                 run_queue_size: int = 20, token_file: Optional[str] = None):
        self.hub_config = hub_config
        self.run_root = Path(run_root).resolve()
        self.max_concurrent_runs = max_concurrent_runs
        self.run_queue_size = run_queue_size

        # Authentication
        self.hub_token = None
        if token_file is not None:
            token = Path(token_file).read_text().strip()
            if not token:
                raise HubError(500, "invalid_token", "hub token file is empty")
            self.hub_token = token

        # State tracking
        self.runs: Dict[str, RunInfo] = {}
        self.gpu_pool: Dict[str, GpuResource] = {}
        self.gpu_tasks: Dict[str, GpuTask] = {}
        self.rollout_tasks: Dict[str, RolloutTask] = {}
        self.lock = threading.RLock()

        # Initialize GPU pool from config
        if "gpu_pool" in hub_config:
            for gpu_spec in hub_config["gpu_pool"]:
                gpu = GpuResource(
                    id=gpu_spec["id"],
                    ssh=gpu_spec.get("ssh", {}),
                    source_dir=gpu_spec["source_dir"],
                    train_python=gpu_spec["train_python"],
                    expected_gpus=gpu_spec.get("expected_gpus", 1)
                )
                self.gpu_pool[gpu.id] = gpu

        # Start scheduler thread
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(
            target=self._scheduler_loop, name="hub-scheduler", daemon=True
        )
        self.scheduler_thread.start()

    def close(self):
        """Shutdown the hub manager."""
        self.scheduler_running = False
        self.scheduler_thread.join(timeout=5)

    def _authenticate(self, token: str):
        """Validate hub access token."""
        if self.hub_token is not None and token != self.hub_token:
            raise HubError(401, "unauthorized", "invalid hub token")

    def _scheduler_loop(self):
        """Background scheduler for GPU resource allocation."""
        while self.scheduler_running:
            try:
                self._schedule_pending_runs()
                time.sleep(1.0)
            except Exception as e:
                # Log but don't crash the scheduler
                print(f"Scheduler error: {e}")

    def _schedule_pending_runs(self):
        """Allocate GPU resources to pending runs if available."""
        with self.lock:
            # Count currently running
            running = sum(1 for r in self.runs.values() if r.status == "running")
            if running >= self.max_concurrent_runs:
                return

            # Find pending runs (FIFO order)
            pending = [
                (r.created_at, run_id, r)
                for run_id, r in self.runs.items()
                if r.status == "pending"
            ]
            pending.sort()

            # Try to schedule each pending run
            for _, run_id, run_info in pending:
                if running >= self.max_concurrent_runs:
                    break

                # Find idle GPU
                idle_gpu = next(
                    (g for g in self.gpu_pool.values() if g.status == "idle"),
                    None
                )

                if idle_gpu is None:
                    continue  # No GPU available

                # Create GPU task
                task_id = f"gpu_{secrets.token_hex(8)}"
                gpu_task = GpuTask(
                    task_id=task_id,
                    run_id=run_id,
                    plan=run_info.plan,
                    deployment=run_info.deployment
                )
                self.gpu_tasks[task_id] = gpu_task

                # Reserve GPU and update run
                idle_gpu.status = "reserved"
                idle_gpu.current_run = run_id
                idle_gpu.current_task = task_id
                run_info.status = "scheduled"
                run_info.gpu_task_id = task_id
                run_info.scheduled_at = time.time()
                running += 1

    def submit_run(self, plan: dict, deployment: dict, *, token: str) -> dict:
        """Submit a new training run to the hub."""
        self._authenticate(token)

        with self.lock:
            # Check queue capacity
            if len(self.runs) >= self.run_queue_size:
                raise HubError(503, "queue_full", "hub run queue is at capacity")

            # Generate run ID
            run_id = f"run_{secrets.token_hex(12)}"

            # Create run info
            run_info = RunInfo(
                run_id=run_id,
                plan=plan,
                deployment=deployment
            )
            self.runs[run_id] = run_info

            return {
                "schema_version": HUB_SCHEMA,
                "run_id": run_id,
                "status": "submitted",
                "queue_position": sum(
                    1 for r in self.runs.values() if r.status == "pending"
                )
            }

    def query_run(self, run_id: str, *, token: str) -> dict:
        """Query the status of a submitted run."""
        self._authenticate(token)

        with self.lock:
            run_info = self.runs.get(run_id)
            if run_info is None:
                raise HubError(404, "not_found", f"unknown run: {run_id}")

            result = {
                "schema_version": HUB_SCHEMA,
                "run_id": run_id,
                "status": run_info.status,
                "created_at": run_info.created_at
            }

            if run_info.scheduled_at:
                result["scheduled_at"] = run_info.scheduled_at
            if run_info.completed_at:
                result["completed_at"] = run_info.completed_at
            if run_info.error:
                result["error"] = run_info.error
            if run_info.gpu_task_id:
                result["gpu_task_id"] = run_info.gpu_task_id

            return result

    def cancel_run(self, run_id: str, *, token: str) -> dict:
        """Cancel a pending or running run."""
        self._authenticate(token)

        with self.lock:
            run_info = self.runs.get(run_id)
            if run_info is None:
                raise HubError(404, "not_found", f"unknown run: {run_id}")

            if run_info.status in ("completed", "failed"):
                raise HubError(400, "invalid_state", "cannot cancel completed run")

            # Release GPU if allocated
            if run_info.gpu_task_id:
                gpu_task = self.gpu_tasks.get(run_info.gpu_task_id)
                if gpu_task and gpu_task.gpu_id:
                    gpu = self.gpu_pool.get(gpu_task.gpu_id)
                    if gpu:
                        gpu.status = "idle"
                        gpu.current_run = None
                        gpu.current_task = None
                gpu_task.status = "failed"
                gpu_task.error = "cancelled"

            run_info.status = "failed"
            run_info.error = "cancelled by user"
            run_info.completed_at = time.time()

            return {"run_id": run_id, "status": "cancelled"}

    def hub_status(self, *, token: str) -> dict:
        """Query overall hub status and resource utilization."""
        self._authenticate(token)

        with self.lock:
            gpu_status = {
                "total": len(self.gpu_pool),
                "idle": sum(1 for g in self.gpu_pool.values() if g.status == "idle"),
                "reserved": sum(1 for g in self.gpu_pool.values() if g.status == "reserved"),
                "busy": sum(1 for g in self.gpu_pool.values() if g.status == "busy")
            }

            run_status = {
                "total": len(self.runs),
                "pending": sum(1 for r in self.runs.values() if r.status == "pending"),
                "scheduled": sum(1 for r in self.runs.values() if r.status == "scheduled"),
                "running": sum(1 for r in self.runs.values() if r.status == "running"),
                "completed": sum(1 for r in self.runs.values() if r.status == "completed"),
                "failed": sum(1 for r in self.runs.values() if r.status == "failed")
            }

            return {
                "schema_version": HUB_SCHEMA,
                "gpu_pool": gpu_status,
                "runs": run_status,
                "max_concurrent_runs": self.max_concurrent_runs,
                "queue_capacity": self.run_queue_size
            }

    def pull_gpu_task(self, gpu_id: str) -> Optional[dict]:
        """GPU worker pulls next available task."""
        with self.lock:
            # Find reserved task for this GPU
            for task_id, task in self.gpu_tasks.items():
                if task.status == "queued":
                    gpu = self.gpu_pool.get(gpu_id)
                    if gpu and gpu.current_task == task_id:
                        # Assign task
                        task.status = "assigned"
                        task.gpu_id = gpu_id
                        task.assigned_at = time.time()
                        gpu.status = "busy"

                        # Update run status
                        run_info = self.runs.get(task.run_id)
                        if run_info:
                            run_info.status = "running"

                        return {
                            "task_id": task_id,
                            "run_id": task.run_id,
                            "plan": task.plan,
                            "deployment": task.deployment
                        }

            return None

    def report_gpu_status(self, gpu_id: str, task_id: str, status: dict) -> dict:
        """GPU worker reports task completion or failure."""
        with self.lock:
            task = self.gpu_tasks.get(task_id)
            if task is None or task.gpu_id != gpu_id:
                raise HubError(404, "not_found", "unknown or mismatched task")

            # Update task
            if status.get("success"):
                task.status = "completed"
                task.result = status.get("result")
            else:
                task.status = "failed"
                task.error = status.get("error", "unknown error")
            task.completed_at = time.time()

            # Release GPU
            gpu = self.gpu_pool.get(gpu_id)
            if gpu:
                gpu.status = "idle"
                gpu.current_run = None
                gpu.current_task = None
                gpu.assigned_at = None

            # Update run
            run_info = self.runs.get(task.run_id)
            if run_info:
                if task.status == "completed":
                    run_info.status = "completed"
                else:
                    run_info.status = "failed"
                    run_info.error = task.error
                run_info.completed_at = time.time()

            return {"task_id": task_id, "acknowledged": True}

    def submit_rollout_task(self, run_id: str, group_request: dict) -> dict:
        """Submit a rollout group task for worker processing."""
        with self.lock:
            task_id = f"rollout_{secrets.token_hex(8)}"
            task = RolloutTask(
                task_id=task_id,
                run_id=run_id,
                group_id=group_request.get("group_id", task_id),
                request=group_request
            )
            self.rollout_tasks[task_id] = task

            # Link to run
            run_info = self.runs.get(run_id)
            if run_info:
                run_info.rollout_tasks.append(task_id)

            return {"task_id": task_id, "status": "queued"}

    def pull_rollout_task(self, worker_id: str) -> Optional[dict]:
        """Rollout worker pulls next available task."""
        with self.lock:
            # Find oldest queued task
            queued = [
                (task.created_at, task_id, task)
                for task_id, task in self.rollout_tasks.items()
                if task.status == "queued"
            ]

            if not queued:
                return None

            queued.sort()
            _, task_id, task = queued[0]

            # Assign to worker
            task.status = "assigned"
            task.worker_id = worker_id
            task.assigned_at = time.time()

            return {
                "task_id": task_id,
                "run_id": task.run_id,
                "group_id": task.group_id,
                "request": task.request
            }

    def report_rollout_result(self, worker_id: str, task_id: str, result: dict) -> dict:
        """Rollout worker reports task completion."""
        with self.lock:
            task = self.rollout_tasks.get(task_id)
            if task is None or task.worker_id != worker_id:
                raise HubError(404, "not_found", "unknown or mismatched task")

            # Update task
            if result.get("success"):
                task.status = "completed"
                task.result = result
            else:
                task.status = "failed"
                task.error = result.get("error", "unknown error")
            task.completed_at = time.time()

            return {"task_id": task_id, "acknowledged": True}


def make_hub_server(hub_manager: HubManager, host: str = "0.0.0.0", port: int = 0,
                    *, max_body_bytes: int = 16 * 1024 * 1024):
    """Create HTTP server for hub control plane."""

    class HubHandler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def log_message(self, fmt, *args):
            # Keep one line per request: without it the hub log carried only
            # the startup banner and scheduling was untraceable.
            print("HUB %s %s %s" % (self.command, self.path, fmt % args if args else ""),
                  flush=True)

        def _parse_token(self):
            """Extract authentication token from headers."""
            # Accept either spelling; the clients only ever send one, but a
            # proxy that adds Authorization must not turn a good request into
            # a 401. X-Api-Key wins when both appear.
            token = self.headers.get("x-api-key", "") or self.headers.get("X-Api-Key", "")
            if not token:
                auth = self.headers.get("Authorization", "")
                if auth.startswith("Bearer "):
                    token = auth[7:]
            return token

        def _read_json_body(self):
            """Read and parse JSON request body."""
            if self.headers.get("Transfer-Encoding"):
                raise HubError(400, "invalid_request", "chunked bodies not supported")
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > max_body_bytes:
                raise HubError(400, "invalid_request", "invalid body size")
            return json.loads(self.rfile.read(size))

        def reply(self, status: int, payload: dict):
            """Send JSON response."""
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            try:
                path = urlsplit(self.path).path
                token = self._parse_token()

                if path == "/hub/status":
                    result = hub_manager.hub_status(token=token)
                    return self.reply(200, result)

                elif path.startswith("/hub/runs/"):
                    run_id = path.split("/")[-1]
                    result = hub_manager.query_run(run_id, token=token)
                    return self.reply(200, result)

                elif path == "/pull/gpu_task":
                    gpu_id = self.headers.get("X-GPU-ID", "")
                    task = hub_manager.pull_gpu_task(gpu_id)
                    if task is None:
                        return self.reply(204, {})  # No content
                    return self.reply(200, task)

                elif path == "/pull/rollout_task":
                    worker_id = self.headers.get("X-Worker-ID", "")
                    task = hub_manager.pull_rollout_task(worker_id)
                    if task is None:
                        return self.reply(204, {})
                    return self.reply(200, task)

                else:
                    return self.reply(404, {
                        "error": {"type": "not_found", "message": "endpoint not found"}
                    })

            except HubError as e:
                return self.reply(e.status, {
                    "error": {"type": e.code, "message": str(e)}
                })
            except Exception as e:
                return self.reply(500, {
                    "error": {"type": "internal_error", "message": str(e)}
                })

        def do_POST(self):
            try:
                path = urlsplit(self.path).path
                token = self._parse_token()

                if path == "/hub/runs":
                    body = self._read_json_body()
                    result = hub_manager.submit_run(
                        body.get("plan", {}),
                        body.get("deployment", {}),
                        token=token
                    )
                    return self.reply(201, result)

                elif path == "/report/gpu_status":
                    body = self._read_json_body()
                    result = hub_manager.report_gpu_status(
                        body["gpu_id"],
                        body["task_id"],
                        body["status"]
                    )
                    return self.reply(200, result)

                elif path == "/report/rollout_result":
                    body = self._read_json_body()
                    result = hub_manager.report_rollout_result(
                        body["worker_id"],
                        body["task_id"],
                        body["result"]
                    )
                    return self.reply(200, result)

                elif path == "/hub/rollout_tasks":
                    body = self._read_json_body()
                    result = hub_manager.submit_rollout_task(
                        body["run_id"],
                        body["request"]
                    )
                    return self.reply(201, result)

                else:
                    return self.reply(404, {
                        "error": {"type": "not_found", "message": "endpoint not found"}
                    })

            except HubError as e:
                return self.reply(e.status, {
                    "error": {"type": e.code, "message": str(e)}
                })
            except Exception as e:
                return self.reply(500, {
                    "error": {"type": "internal_error", "message": str(e)}
                })

        def do_DELETE(self):
            try:
                path = urlsplit(self.path).path
                token = self._parse_token()

                if path.startswith("/hub/runs/"):
                    run_id = path.split("/")[-1]
                    result = hub_manager.cancel_run(run_id, token=token)
                    return self.reply(200, result)

                else:
                    return self.reply(404, {
                        "error": {"type": "not_found", "message": "endpoint not found"}
                    })

            except HubError as e:
                return self.reply(e.status, {
                    "error": {"type": e.code, "message": str(e)}
                })
            except Exception as e:
                return self.reply(500, {
                    "error": {"type": "internal_error", "message": str(e)}
                })

    class HubServer(ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 128

    server = HubServer((host, port), HubHandler)
    return server


if __name__ == "__main__":
    import sys
    import signal

    # Simple CLI for testing
    if len(sys.argv) < 2:
        print("Usage: python -m harborrl.deployment.hub_server <hub_config.yaml>")
        sys.exit(1)

    import yaml
    config_path = Path(sys.argv[1])
    with config_path.open() as f:
        config = yaml.safe_load(f)

    manager_config = config.get("manager", {})

    hub = HubManager(
        hub_config=manager_config,
        run_root=config.get("storage", {}).get("run_root", "/tmp/harborrl"),
        max_concurrent_runs=manager_config.get("max_concurrent_runs", 4),
        run_queue_size=manager_config.get("run_queue_size", 20),
        token_file=manager_config.get("run_token_file")
    )

    server = make_hub_server(
        hub,
        host=manager_config.get("bind", "0.0.0.0"),
        port=manager_config.get("hub_port", 49300)
    )

    print(f"Hub server listening on {server.server_address}")

    def shutdown_handler(signum, frame):
        print("\nShutting down hub server...")
        hub.close()
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    server.serve_forever()

