# Harnesses

## Claude Code

`claude_code` is the only registered harness. It speaks Anthropic Messages v1 to the
HarborRL policy gateway and consumes Qwen tool calls through Harbor `0.23.0`.

The bundled profile shape is:

```json
{
  "agent_timeout_sec": 120,
  "cli_version": "2.1.141",
  "max_turns": 4,
  "model": "policy",
  "setup_timeout_sec": 300,
  "verifier_timeout_sec": 60
}
```

All six fields are required. `disallowed_tools` is the only optional field and must
be a comma-separated identifier list. HarborRL always adds `Agent` and `Task` to the
denylist.

## Runtime controls

For each attempt, the adapter:

- requires an exact CLI version rather than `latest`;
- injects a short-lived attempt credential and the plain gateway origin;
- disables prompt caching, cache warmup, compaction, auto-memory, auto-update, and
  nonessential traffic;
- sets `max_thinking_tokens` to zero and uses bypass-permissions mode inside the
  isolated task environment; and
- strips ambient provider keys, Claude settings, and proxy credentials from the
  runner environment.

The repository example pins Claude Code `2.1.141`. Other versions must be probed
against the native session contract before use.

## Consumption audit

HTTP delivery is not considered proof that the harness consumed a model response.
HarborRL parses the persisted native Claude session, requires one complete session,
merges the known split-message form, rejects auxiliary/sidechain generations and
summaries, and matches response IDs and content against gateway evidence.

The audit permits the locked CLI's special empty final `end_turn` form only when the
terminal result provides matching success evidence.

## Limits

- Other harnesses are not registered.
- Ambient credentials are intentionally rejected.
- The configured gateway URL must be a plain HTTP(S) origin with no `/v1` suffix,
  embedded credentials, query, or fragment.
