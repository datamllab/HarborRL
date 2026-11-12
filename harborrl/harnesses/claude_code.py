"""Claude Code harness adapter with explicit credentials and session audit."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from harborrl.rollout.harbor_job.audit import audit_session, audit_termination
from harborrl.rollout.harbor_job.bindings import (
    claude_config,
    sanitized_runner_env,
)


class ClaudeCodeHarnessAdapter:
    harness_id = "claude_code"

    def validate_profile(self, profile, gateway_url):
        return claude_config(profile, gateway_url, "validation-only")

    def agent_config(self, profile, gateway_url, credential, max_output_tokens=8192):
        return claude_config(profile, gateway_url, credential, max_output_tokens)

    def runner_environment(self, environment):
        return sanitized_runner_env(environment)

    def audit_session(self, path, registry, attempt_id, *, cli_version=None):
        return audit_session(path, registry, attempt_id, cli_version=cli_version)

    def audit_termination(self, path, result_path, profile, audit, *, turns):
        return audit_termination(path, result_path, profile, audit, turns=turns)

    def audit_empty_final(self, path, registry, attempt_id, audit):
        """Claude 2.1.141 omits empty final messages from persisted sessions."""
        missing = registry.unconsumed_responses(attempt_id)
        if not missing:
            return
        if len(missing) != 1 or missing[0] != {
            "response_id": missing[0]["response_id"], "content": [],
            "finish_reason": "end_turn", "is_final": True,
        }:
            raise ValueError("only one empty final end_turn may use terminal evidence")
        path = Path(path)
        raw = path.read_bytes()
        events = [json.loads(line) for line in raw.splitlines() if line.strip()]
        results = [event for event in events if event.get("type") == "result"]
        if len(results) != 1 or events[-1] != results[0]:
            raise ValueError("expected one final Claude result")
        result = results[0]
        expected = {"subtype": "success", "is_error": False, "result": "",
                    "stop_reason": "end_turn", "session_id": audit["session_id"],
                    "terminal_reason": "completed"}
        if any(result.get(key) != value for key, value in expected.items()) or result.get("is_error") is not False:
            raise ValueError("empty final response lacks successful matching session result")
        evidence = f"{path}#sha256={hashlib.sha256(raw).hexdigest()}"
        registry.consume(attempt_id, missing[0]["response_id"], [], evidence)
