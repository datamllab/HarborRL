"""The remote gateway shard process: one registry, one gateway, one admin plane.

Usually launched on another machine that shares the run directory:

    RUN_ID=... RUN_DIR=/mnt/shared-storage-user/.../runs/Run-... \\
    HARBORRL_SHARD_NAME=gw1 \\
    HARBORRL_SHARD_GATEWAY_HOST=0.0.0.0 HARBORRL_SHARD_GATEWAY_PORT=49300 \\
    HARBORRL_SHARD_GATEWAY_URL=http://10.98.250.180:49300 \\
    HARBORRL_SHARD_ADMIN_HOST=0.0.0.0 HARBORRL_SHARD_ADMIN_PORT=49301 \\
    HARBORRL_SHARD_TOKEN_FILE=/mnt/.../run-token \\
    HARBORRL_SHARD_ENDPOINT=http://router:30000 \\
    HARBORRL_NATIVE_PROFILE=/mnt/.../profile.json HF_CKPT=/mnt/.../ckpt \\
    python -m harborrl.gateway.shard_service

A fixed gateway port requires its declared URL (the spelling must exist
before the process starts); an ephemeral bind — port 0, no URL — instead
reports the resolved gateway and admin ports on stdout as one ready line,
which is how an auto-growing manager launches this service over ssh.

Traces land in RUN_DIR/native-traces/shard-{name} on the shared storage, so
the GPU side digests them exactly like local-shard traces. The admin plane
carries only manager-side registry operations, run-token authenticated;
generation traffic hits the gateway port directly from runners.
"""
from __future__ import annotations

import hmac
import json
import signal
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from harborrl.trajectories.native import Identity

SHARD_SERVICE = "harborrl-gateway-shard-v1"
ADMIN_MAX_BODY_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ShardServiceConfig:
    """Deployment-independent description of one shard process."""

    run_id: str
    run_root: str
    shard_name: str
    gateway_host: str
    gateway_port: int
    gateway_url: str
    admin_host: str
    admin_port: int
    token_file: str
    endpoint: str
    profile_path: str
    model_tag: str
    max_output_tokens: int
    max_context: int
    expected_tokenizer_digest: str
    expected_template_digest: str
    audited_raw_logprobs: bool

    @property
    def trace_dir(self):
        # The same spelling the manager-side runtime derives from RUN_DIR.
        return Path(self.run_root) / "native-traces" / f"shard-{self.shard_name}"

    @staticmethod
    def _required(env, name):
        value = env.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"gateway shard service requires {name}")
        return value

    @classmethod
    def from_environment(cls, env):
        gateway_port = int(cls._required(env, "HARBORRL_SHARD_GATEWAY_PORT"))
        if gateway_port == 0:
            # An ephemeral gateway bind reports its port in the ready line;
            # whoever launched this process composes and advertises the URL.
            gateway_url = ""
        else:
            gateway_url = cls._required(env, "HARBORRL_SHARD_GATEWAY_URL")
        return cls(
            run_id=cls._required(env, "RUN_ID"),
            run_root=cls._required(env, "RUN_DIR"),
            shard_name=cls._required(env, "HARBORRL_SHARD_NAME"),
            gateway_host=env.get("HARBORRL_SHARD_GATEWAY_HOST", "0.0.0.0"),
            gateway_port=gateway_port,
            gateway_url=gateway_url,
            admin_host=env.get("HARBORRL_SHARD_ADMIN_HOST", "0.0.0.0"),
            admin_port=int(cls._required(env, "HARBORRL_SHARD_ADMIN_PORT")),
            token_file=cls._required(env, "HARBORRL_SHARD_TOKEN_FILE"),
            endpoint=cls._required(env, "HARBORRL_SHARD_ENDPOINT").rstrip("/"),
            profile_path=cls._required(env, "HARBORRL_NATIVE_PROFILE"),
            model_tag=env.get("MODEL_TAG", env.get("MODEL_ARGS_FILE", "qwen3-8B")),
            max_output_tokens=int(env.get("ROLLOUT_MAX_RESPONSE_LEN", "8192")),
            max_context=int(env.get("ROLLOUT_MAX_CONTEXT_LEN", "32768")),
            expected_tokenizer_digest=cls._required(env, "HARBORRL_NATIVE_TOKENIZER_DIGEST"),
            expected_template_digest=cls._required(env, "HARBORRL_NATIVE_TEMPLATE_DIGEST"),
            audited_raw_logprobs=env.get("HARBORRL_NATIVE_AUDITED_LOGPROBS") == "1",
        )


def _seal(registry, body):
    turns, seal = registry.seal(body["attempt_id"])
    return {"turns": turns, "seal": seal}


def _registry_actions():
    """The manager-side registry surface this process serves."""
    return {
        "register": lambda registry, body: {"credential": registry.register(
            Identity(**body["identity"]), body["trial_id"],
            credential=body.get("credential"))},
        "close": lambda registry, body: registry.close(body["attempt_id"]),
        "wait_delivered": lambda registry, body: registry.wait_delivered(
            body["attempt_id"], float(body["timeout"])),
        "unconsumed_responses": lambda registry, body: registry.unconsumed_responses(
            body["attempt_id"]),
        "consume": lambda registry, body: registry.consume(
            body["attempt_id"], body["response_id"], body["content"],
            body["evidence_ref"], cli_version=body.get("cli_version")),
        "audit_tool_rejections": lambda registry, body: registry.audit_tool_rejections(
            body["attempt_id"], body["results"], body["evidence_ref"]),
        "seal": _seal,
        "close_admission": lambda registry, body: registry.close_admission(),
        "drain": lambda registry, body: registry.drain(float(body["timeout"])),
        "publish_version": lambda registry, body: registry.publish_version(
            body["states"], body["expected"]),
    }


def make_admin_server(registry, *, shard_name, token, host="127.0.0.1", port=0,
                      on_shutdown=None):
    """Run-token-authenticated admin plane for one shard process.

    Every registry failure crosses the channel as data ({"ok": false,
    "error": {type, message}}) so the manager re-raises the exact exception
    the local registry would have raised; only transport failures stay
    transport errors.
    """
    actions = _registry_actions()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(135)

        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorized(self):
            given = self.headers.get("x-api-key", "")
            return bool(given) and hmac.compare_digest(given, token)

        def do_GET(self):
            if self.path == "/healthz":
                return self.reply(200, {"ok": True, "service": SHARD_SERVICE,
                                        "shard": shard_name})
            if self.path == "/status":
                if not self.authorized():
                    return self.reply(401, {"error": "invalid shard token"})
                return self.reply(200, {"shard": shard_name, "version": registry.version,
                                        "accepting": registry.accepting})
            self.reply(404, {"error": "unsupported endpoint"})

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "invalid shard token"})
            if self.path == "/shutdown":
                self.reply(200, {"ok": True, "shard": shard_name})
                if on_shutdown is not None:
                    threading.Thread(target=on_shutdown, daemon=True).start()
                return
            if not self.path.startswith("/registry/"):
                return self.reply(404, {"error": "unsupported endpoint"})
            action = self.path[len("/registry/"):]
            handler = actions.get(action)
            if handler is None:
                return self.reply(200, {"ok": False, "error": {
                    "type": "ValueError",
                    "message": f"unknown registry action {action!r}"}})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 0 or size > ADMIN_MAX_BODY_BYTES:
                    raise ValueError("invalid request body size")
                # Payload-free actions (close_admission) post an empty body.
                body = json.loads(self.rfile.read(size)) if size else {}
                result = handler(registry, body)
            except Exception as exc:  # noqa: BLE001 — registry failures cross as data
                return self.reply(200, {"ok": False, "error": {
                    "type": type(exc).__name__, "message": str(exc)}})
            self.reply(200, {"ok": True, "result": result})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def _ready_record(config, *, gateway_port, admin_port):
    """The one stdout line announcing this shard's resolved endpoints.

    An ephemeral gateway bind (port 0) has no declared URL, so the record
    carries the bound port for the launcher to compose the real URL from;
    a fixed port repeats the declared URL and port for a single spelling
    everywhere.
    """
    return {"shard": config.shard_name, "service": SHARD_SERVICE,
            "gateway": config.gateway_url or f"http://localhost:{gateway_port}",
            "gateway_port": gateway_port,
            "admin_port": admin_port,
            "trace_dir": str(config.trace_dir)}


def serve_shard(config, tokenizer, *, backend=None):
    """Run one shard process until SIGTERM, SIGINT or admin shutdown."""
    from harborrl.gateway.server import Gateway, make_server
    from harborrl.gateway.sglang_backend import SGLangBackend
    from harborrl.gateway.trace import TraceRegistry
    from harborrl.math_registry import MATH_REGISTRY
    from harborrl.rollout.runtime import tokenizer_fingerprint
    from harborrl.trajectories.native import read_json

    tokenizer_digest, template_digest = tokenizer_fingerprint(tokenizer)
    if (tokenizer_digest != config.expected_tokenizer_digest
            or template_digest != config.expected_template_digest):
        raise ValueError("live tokenizer/template digest differs from the native lock")
    profile = read_json(config.profile_path)
    registry = TraceRegistry(config.trace_dir)
    model = MATH_REGISTRY.model_for_preset(config.model_tag)
    resolved = backend or SGLangBackend(
        config.endpoint, tokenizer, model.adapter,
        tokenizer_digest=tokenizer_digest, template_digest=template_digest,
        max_context=config.max_context,
        audited_raw_logprobs=config.audited_raw_logprobs)
    gateway = Gateway(registry, resolved, model=profile["model"],
                      max_output_tokens=config.max_output_tokens)
    token = Path(config.token_file).read_text().strip()
    if not token:
        raise ValueError(f"shard token file is empty: {config.token_file}")
    gateway_server = make_server(gateway, host=config.gateway_host,
                                 port=config.gateway_port)
    gateway_thread = threading.Thread(
        target=gateway_server.serve_forever,
        name=f"harborrl-gateway-{config.shard_name}", daemon=True)
    admin = None
    stopped = []

    def on_shutdown():
        if stopped:
            return
        stopped.append(True)
        gateway_server.shutdown()
        if admin is not None:
            admin.shutdown()

    admin = make_admin_server(registry, shard_name=config.shard_name, token=token,
                              host=config.admin_host, port=config.admin_port,
                              on_shutdown=on_shutdown)

    def handle_signal(signum, frame):
        # shutdown() must never run on the serve_forever thread.
        threading.Thread(target=on_shutdown, daemon=True).start()

    previous = {sig: signal.signal(sig, handle_signal)
                for sig in (signal.SIGTERM, signal.SIGINT)}
    gateway_thread.start()
    admin_port = admin.server_address[1]
    print(json.dumps(_ready_record(config,
                                   gateway_port=gateway_server.server_address[1],
                                   admin_port=admin_port)), flush=True)
    try:
        admin.serve_forever()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        gateway_thread.join(5)
        gateway_server.server_close()
        admin.server_close()
        print(json.dumps({"shard": config.shard_name, "stopped": True}), flush=True)


def main():
    import os

    config = ShardServiceConfig.from_environment(dict(os.environ))

    def load_tokenizer():
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(os.environ["HF_CKPT"],
                                             trust_remote_code=True)

    serve_shard(config, load_tokenizer())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
