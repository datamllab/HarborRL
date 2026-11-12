"""Harbor 0.23 Claude Code configuration. No import of Harbor in the parent."""
import math
import re
from urllib.parse import urlsplit


def claude_config(profile, gateway_url, credential, max_output_tokens=8192):
    required = {"cli_version", "model", "max_turns", "agent_timeout_sec", "setup_timeout_sec", "verifier_timeout_sec"}
    optional = {"disallowed_tools", "append_system_prompt", "environment_timeout_sec"}
    allowed = required | optional
    if not required.issubset(profile) or set(profile) - allowed:
        raise ValueError(f"Claude profile requires {sorted(required)} and optionally {sorted(optional)}")
    if not isinstance(profile["cli_version"], str) or not re.fullmatch(r"\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?", profile["cli_version"]):
        raise ValueError("exact Claude CLI version required; latest is forbidden")
    if not isinstance(profile["model"], str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", profile["model"]):
        raise ValueError("explicit policy model alias required")
    for key in (required - {"model", "cli_version"}) | ({'environment_timeout_sec'} & profile.keys()):
        if type(profile[key]) is not int or profile[key] <= 0:
            raise ValueError(f"positive integer required: {key}")
    disallowed_tools = profile.get("disallowed_tools", "Agent,Task")
    if not isinstance(disallowed_tools, str) or not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*(?:,[A-Za-z][A-Za-z0-9_]*)*", disallowed_tools):
        raise ValueError("disallowed tools must be a comma-separated identifier list")
    kwargs = {"version": profile["cli_version"], "max_turns": profile["max_turns"],
              "max_thinking_tokens": 0, "disallowed_tools": disallowed_tools,
              "permission_mode": "bypassPermissions"}
    if "append_system_prompt" in profile:
        prompt = profile["append_system_prompt"]
        if not isinstance(prompt, str) or not prompt.strip() or "\x00" in prompt:
            raise ValueError("append_system_prompt must be a nonempty string without NUL")
        kwargs["append_system_prompt"] = prompt
    url = urlsplit(gateway_url)
    if url.scheme not in ("http", "https") or not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        raise ValueError("Anthropic base URL must be a plain origin (no /v1 suffix)")
    if not isinstance(credential, str) or not credential:
        raise ValueError("attempt credential is required")
    if type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise ValueError("positive Claude output budget required")
    # Some bundled runtimes do not interpret CIDR blocks in NO_PROXY.  The
    # policy gateway is the one mandatory direct route for every attempt.
    return {"name": "claude-code", "model_name": profile["model"],
            "override_timeout_sec": profile["agent_timeout_sec"],
            "override_setup_timeout_sec": profile["setup_timeout_sec"],
            "kwargs": kwargs,
            "env": {**client_controls(), "ANTHROPIC_API_KEY": credential,
                    "ANTHROPIC_MODEL": profile["model"],
                    "ANTHROPIC_BASE_URL": gateway_url.rstrip("/"),
                    "NO_PROXY": url.hostname, "no_proxy": url.hostname,
                    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max_output_tokens)}}


def environment_timeouts(profile, resources):
    """Use Harbor's public multiplier without changing the original task config."""
    if 'environment_timeout_sec' not in profile:
        return {}
    budget = profile['environment_timeout_sec']
    base = resources.get('build_timeout_sec', 600.0)  # Harbor 0.23 default.
    if type(budget) is not int or budget <= 0:
        raise ValueError('positive integer required: environment_timeout_sec')
    if type(base) not in (int, float) or not math.isfinite(base) or base <= 0:
        raise ValueError('positive finite task environment build timeout required')
    return {'environment_build_timeout_multiplier': budget / base}


def client_controls():
    """Pass controls into the Agent environment, not only the Runner host."""
    return {"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING": "1",
            "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
            "DISABLE_AUTOUPDATER": "1", "DISABLE_AUTO_COMPACT": "1"}


def sanitized_runner_env(environ):
    """No ambient provider keys, Claude settings or proxy credentials enter Runner."""
    allowed = {"PATH", "HOME", "USER", "LANG", "TMPDIR", "PYTHONPATH", "DOCKER_HOST", "DOCKER_TLS_VERIFY", "DOCKER_CERT_PATH"}
    env = {k: v for k, v in environ.items() if k in allowed}
    # These are profile controls, not proof of CLI behavior. Native session audit is mandatory.
    env.update(client_controls())
    return env
