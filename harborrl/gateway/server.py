"""Bounded stdlib HTTP transport. All policy requests require attempt credentials."""
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from harborrl.trajectories.native import digest, encode

from .messages import ProtocolError, convert, sse_events


class Gateway:
    def __init__(self, registry, backend, *, model, max_output_tokens=8192):
        self.registry, self.backend = registry, backend
        self.model, self.max_output_tokens = model, max_output_tokens

    def generate(self, token, payload):
        aid, rid, identity, trial = self.registry.begin(token)
        try:
            converted = convert(payload, model=self.model, max_output_tokens=self.max_output_tokens)
            if digest(converted["sampling"]) != identity.sampling_digest:
                raise ProtocolError("sampling parameters differ from attempt manifest")
            response_id = "msg_" + secrets.token_hex(16)
            generated = self.backend.generate(converted, identity, response_id)
            if generated.get("policy_version") != identity.policy_version:
                raise ValueError("generation policy version mismatch")
            message = {"id": response_id, "type": "message", "role": "assistant", "model": self.model,
                       "content": generated.pop("content"), "stop_reason": generated["finish_reason"],
                       "stop_sequence": generated.pop("stop_sequence", None),
                       "usage": {"input_tokens": len(generated["input_ids"]), "output_tokens": len(generated["output_ids"])}}
            turn = {**generated, "request_id": rid, "response_id": response_id,
                    "identity": identity.to_dict(), "harbor_trial_id": trial, "role": "policy",
                    "client_request": payload, "response": message, "delivery": "generated"}
            for key in ("client_request", "serving_input", "response"):
                turn[key + "_digest"] = digest(turn[key])
            self.registry.finish(aid, rid, turn)
            return aid, message
        except BaseException as exc:
            # Deliberately retain failed generation in the seal, even if CLI retries.
            self.registry.finish(aid, rid, error=type(exc).__name__)
            raise

    def count_tokens(self, token, payload):
        self.registry.authenticate(token)
        converted = convert(payload, model=self.model, max_output_tokens=self.max_output_tokens, count_only=True)
        return {"input_tokens": self.backend.count_tokens(converted)}


def make_server(gateway, host="127.0.0.1", port=0, *, max_body_bytes=8 * 1024 * 1024,
                max_inflight=None):
    """Serve one gateway. ``max_inflight`` bounds concurrent policy requests.

    Beyond the bound the server answers 503 before any attempt state is
    touched, so an overloaded gateway never poisons a seal: no generation
    began and a client retry is safe. ``None`` keeps the historic unbounded
    behavior for tests and embedded single-attempt use.
    """
    inflight = threading.BoundedSemaphore(max_inflight) if max_inflight else None

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def log_message(self, *args):
            pass  # default access logs can expose caller-supplied URLs

        def do_GET(self):
            if urlsplit(self.path).path != "/readyz":
                return self.reply(404, {"type": "error", "error": {"type": "not_found_error", "message": "unsupported endpoint"}})
            return self.reply(200, {"ok": True, "model": gateway.model})

        def reply(self, status, payload):
            body = encode(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if inflight is not None and not inflight.acquire(blocking=False):
                return self.reply(503, {"type": "error", "error": {"type": "overloaded_error",
                                      "message": "gateway in-flight request limit reached"}})
            try:
                return self._handle_post()
            finally:
                if inflight is not None:
                    inflight.release()

        def _handle_post(self):
            aid = message = None
            try:
                path = urlsplit(self.path).path
                if path not in ("/v1/messages", "/v1/messages/count_tokens"):
                    return self.reply(404, {"type": "error", "error": {"type": "not_found_error", "message": "unsupported endpoint"}})
                if self.headers.get("Transfer-Encoding"):
                    raise ProtocolError("chunked request bodies are not supported")
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > max_body_bytes:
                    raise ProtocolError("invalid request body size")
                token = self.headers.get("x-api-key", "")
                auth = self.headers.get("Authorization", "")
                if auth:
                    if not auth.startswith("Bearer ") or token:
                        raise PermissionError("ambiguous authentication")
                    token = auth[7:]
                gateway.registry.authenticate(token)
                payload = json.loads(self.rfile.read(size), parse_constant=lambda v: (_ for _ in ()).throw(ProtocolError("nonfinite JSON")))
                if path.endswith("count_tokens"):
                    return self.reply(200, gateway.count_tokens(token, payload))
                aid, message = gateway.generate(token, payload)
                if payload.get("stream"):
                    # Entire response is generated/audited first; TCP disconnect still invalidates delivery.
                    body = b"".join(b"event: " + name.encode() + b"\ndata: " + encode(event) + b"\n\n" for name, event in sse_events(message))
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.reply(200, message)
                self.wfile.flush()
                gateway.registry.delivery(aid, message["id"], "sent")
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                if aid and message:
                    gateway.registry.delivery(aid, message["id"], "delivery_unknown")
            except (ValueError, TypeError, KeyError, PermissionError, OSError) as exc:
                if aid and message:
                    gateway.registry.delivery(aid, message["id"], "delivery_unknown")
                status = 401 if isinstance(exc, PermissionError) else 400 if isinstance(exc, (ProtocolError, json.JSONDecodeError)) else 502
                # Do not echo backend errors or request contents (credentials may appear there).
                self.reply(status, {"type": "error", "error": {"type": "authentication_error" if status == 401 else "invalid_request_error" if status == 400 else "api_error", "message": f"{type(exc).__name__}: {exc}"}})

    class Server(ThreadingHTTPServer):
        # One thread per connection; the kernel backlog bounds queued peers.
        daemon_threads = True
        request_queue_size = 128

    server = Server((host, port), Handler)
    return server
