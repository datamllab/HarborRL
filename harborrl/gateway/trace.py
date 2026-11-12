"""Attempt-scoped credentials, request draining and immutable trace sealing."""
import hashlib
import re
import secrets
import threading
import time
from copy import deepcopy
from pathlib import Path

from harborrl.trajectories.native import digest, publish


def consumed_tool_defaults(turn, content):
    """Validate only omitted optional defaults declared by the request's tool schema."""
    original = turn['response']['content']
    if not isinstance(content, list) or len(content) != len(original):
        raise ValueError('consumed response differs or delivery is unknown')
    expected = deepcopy(original)
    changes = []
    tools = turn.get('client_request', {}).get('tools', [])
    for index, (served, consumed) in enumerate(zip(expected, content)):
        if digest(served) == digest(consumed):
            continue
        if (served.get('type') != 'tool_use' or not isinstance(consumed, dict)
                or not isinstance(served.get('input'), dict) or not isinstance(consumed.get('input'), dict)):
            raise ValueError('consumed response differs or delivery is unknown')
        matches = [tool for tool in tools if tool.get('name') == served.get('name')]
        if len(matches) != 1:
            raise ValueError('consumed tool defaults require a unique declared schema')
        schema = matches[0].get('input_schema', {})
        added = {}
        for key in consumed['input'].keys() - served['input'].keys():
            prop = schema.get('properties', {}).get(key, {})
            if ('default' not in prop or key in schema.get('required', [])
                    or digest(consumed['input'][key]) != digest(prop['default'])):
                raise ValueError('consumed tool parameter is not a declared optional default')
            served['input'][key] = deepcopy(prop['default'])
            added[key] = prop['default']
        if digest(served) != digest(consumed):
            raise ValueError('consumed response differs or delivery is unknown')
        changes.append({'block': index, 'tool': served['name'], 'defaults': added,
                        'schema_digest': digest(schema)})
    return {'kind': 'declared_tool_defaults_v1', 'changes': changes,
            'served_content_digest': digest(original), 'consumed_content_digest': digest(content)}


def consumed_cli_tool_inputs(turn, content, cli_version):
    """Prove pinned native trailing-whitespace removal, then schema defaults."""
    original = turn['response']['content']
    if cli_version != '2.1.141' or not isinstance(content, list) or len(content) != len(original):
        raise ValueError('consumed response differs or delivery is unknown')
    expected = deepcopy(turn)
    changes = []
    # ECMAScript \s, rather than Python's broader Unicode whitespace class.
    whitespace = '\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff'
    for index, (served, consumed) in enumerate(zip(expected['response']['content'], content)):
        if digest(served) == digest(consumed):
            continue
        if (served.get('type') != 'tool_use' or not isinstance(consumed, dict)
                or not isinstance(served.get('input'), dict) or not isinstance(consumed.get('input'), dict)):
            continue  # The strict defaults check below rejects unrelated changes.
        field = {'Edit': 'new_string', 'Write': 'content'}.get(served.get('name'))
        inputs = served['input']
        path = inputs.get('file_path')
        tools = [tool for tool in turn.get('client_request', {}).get('tools', [])
                 if tool.get('name') == served.get('name')]
        if (not field or not isinstance(path, str) or re.search(r'\.(md|mdx)$', path, re.IGNORECASE)
                or len(tools) != 1 or not isinstance(inputs.get(field), str)
                or tools[0].get('input_schema', {}).get('properties', {}).get(field, {}).get('type') != 'string'):
            continue
        pieces = re.split(r'(\r\n|\n|\r)', inputs[field])
        normalized = ''.join(piece.rstrip(whitespace) if i % 2 == 0 else piece
                             for i, piece in enumerate(pieces))
        if normalized != inputs[field] and consumed['input'].get(field) == normalized:
            changes.append({'block': index, 'tool': served['name'], 'field': field,
                            'rule': 'non_markdown_line_trailing_ecmascript_whitespace_v1',
                            'schema_digest': digest(tools[0]['input_schema']),
                            'served_value_digest': digest(inputs[field]),
                            'consumed_value_digest': digest(normalized)})
            inputs[field] = normalized
    if not changes:
        raise ValueError('consumed response differs or delivery is unknown')
    defaults = consumed_tool_defaults(expected, content)
    return {'kind': 'claude_code_2.1.141_tool_inputs_v1', 'changes': changes,
            'declared_defaults': defaults, 'served_content_digest': digest(original),
            'consumed_content_digest': digest(content)}


class TraceRegistry:
    def __init__(self, root):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.attempts = {}
        self.credentials = {}
        self.accepting = True
        self.version = None

    def register(self, identity, trial_id, *, credential=None):
        with self.lock:
            if not self.accepting or (self.version is not None and self.version != identity.policy_version):
                raise ValueError("policy admission closed or version mismatch")
            if identity.attempt_id in self.attempts:
                raise ValueError("attempt already registered; restart requires a new attempt")
            if not isinstance(trial_id, str) or not trial_id:
                raise ValueError("trial identity required")
            token = secrets.token_urlsafe(32) if credential is None else credential
            if not isinstance(token, str) or not token:
                raise ValueError("nonempty credential required")
            key = hashlib.sha256(token.encode()).hexdigest()
            if key in self.credentials or any(s["credential"] == key for s in self.attempts.values()):
                raise ValueError("attempt credential must be unique")
            root = self.root / digest(identity.to_dict())
            publish(root / "manifest.json", {"identity": identity.to_dict(), "harbor_trial_id": trial_id})
            # Never resurrect a persisted attempt after a process restart.
            claim = root / "writer.claim"
            with claim.open("x") as f:
                f.write("single writer\n")
            self.version = identity.policy_version
            self.attempts[identity.attempt_id] = {"identity": identity, "trial_id": trial_id, "root": root,
                "pending": set(), "turns": [], "errors": [], "closed": False, "sealed": False, "credential": key}
            self.credentials[key] = identity.attempt_id
            return token

    def authenticate(self, token):
        with self.lock:
            aid = self.credentials.get(hashlib.sha256(token.encode()).hexdigest())
            if aid is None:
                raise PermissionError("invalid attempt credential")
            state = self.attempts[aid]
            if not self.accepting or state["closed"]:
                raise PermissionError("attempt admission closed")
            return aid

    def begin(self, token):
        with self.lock:
            aid = self.authenticate(token)
            state = self.attempts[aid]
            if state["pending"]:
                state["errors"].append("concurrent_or_auxiliary_generation")
                raise ValueError("only one policy generation per attempt may be in flight")
            request_id = secrets.token_hex(16)
            publish(state["root"] / "requests" / f"{request_id}.json", {"request_id": request_id, "state": "registered"})
            state["pending"].add(request_id)
            return aid, request_id, state["identity"], state["trial_id"]

    def finish(self, aid, request_id, turn=None, error=None):
        with self.changed:
            state = self.attempts[aid]
            if request_id not in state["pending"]:
                raise ValueError("request not pending")
            if error:
                state["errors"].append(error)
            if turn is not None:
                state["turns"].append(deepcopy(turn))
            publish(state["root"] / "generated" / f"{request_id}.json", {"turn": turn, "error": error})
            if error:
                state["pending"].remove(request_id)
                self.changed.notify_all()

    def delivery(self, aid, response_id, status):
        with self.changed:
            state = self.attempts[aid]
            if state["sealed"]:
                raise ValueError("trace already sealed")
            matches = [t for t in state["turns"] if t["response_id"] == response_id]
            if len(matches) != 1 or status not in ("sent", "delivery_unknown"):
                raise ValueError("invalid delivery record")
            matches[0]["delivery"] = status
            publish(state["root"] / "delivery" / f"{response_id}.json", {"status": status})
            state["pending"].discard(matches[0]["request_id"])
            self.changed.notify_all()

    def wait_delivered(self, aid, timeout):
        deadline = time.monotonic() + timeout
        with self.changed:
            state = self.attempts[aid]
            while state["pending"]:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("policy delivery acknowledgement still pending")
                self.changed.wait(remaining)
            if any(turn.get("delivery") not in ("sent", "consumed_confirmed")
                   for turn in state["turns"]):
                raise ValueError("policy response delivery was not confirmed")

    def audit_tool_rejections(self, aid, results, evidence_ref):
        """Unavailable model actions are valid only after an exact native rejection."""
        from .messages import unavailable_tool_error

        with self.lock:
            state = self.attempts[aid]
            if state['sealed']:
                raise ValueError('trace already sealed')
            for turn in state['turns']:
                declared = {tool['name'] for tool in turn['client_request'].get('tools', [])}
                unavailable = [block for block in turn['response']['content']
                               if block['type'] == 'tool_use' and block['name'] not in declared]
                for block in unavailable:
                    matches = [result for result in results if result.get('tool_use_id') == block['id']]
                    if (len(matches) != 1 or matches[0].get('is_error') is not True
                            or matches[0].get('content') != unavailable_tool_error(block['name'])):
                        state['errors'].append('unconfirmed_tool_rejection')
                        raise ValueError('unavailable tool lacks its exact native rejection')
                if unavailable:
                    turn['tool_rejection_ref'] = evidence_ref

    def consume(self, aid, response_id, content, evidence_ref, *, cli_version=None):
        """Called only by trusted native-session audit, never by an HTTP policy client."""
        with self.lock:
            state = self.attempts[aid]
            if state["sealed"]:
                raise ValueError("trace already sealed")
            matches = [t for t in state["turns"] if t["response_id"] == response_id]
            if len(matches) != 1 or not evidence_ref:
                raise ValueError("unknown consumed response")
            turn = matches[0]
            if turn["delivery"] not in ("sent", "consumed_confirmed"):
                raise ValueError("consumed response differs or delivery is unknown")
            normalization = {}
            if digest(turn['response']['content']) != digest(content):
                try:
                    normalization = {'tool_defaults': consumed_tool_defaults(turn, content)}
                except ValueError:
                    normalization = {'tool_normalization': consumed_cli_tool_inputs(turn, content, cli_version)}
            publish(state["root"] / "consumption" / f"{response_id}.json", {
                "response_id": response_id, "content_digest": digest(content), "evidence_ref": evidence_ref,
                **normalization})
            if normalization:
                for kind, proof in normalization.items():
                    turn['consumption_' + kind] = proof
            turn.update(delivery="consumed_confirmed", consumption_ref=evidence_ref)

    def unconsumed_responses(self, aid):
        with self.lock:
            state = self.attempts[aid]
            return [{"response_id": turn["response_id"], "content": deepcopy(turn["response"]["content"]),
                     "finish_reason": turn.get("finish_reason"), "is_final": index == len(state["turns"]) - 1}
                    for index, turn in enumerate(state["turns"]) if turn.get("delivery") != "consumed_confirmed"]

    def close(self, aid):
        with self.lock:
            state = self.attempts[aid]
            state["closed"] = True
            self.credentials.pop(state["credential"], None)

    def seal(self, aid):
        with self.lock:
            state = self.attempts[aid]
            if not state["closed"] or state["pending"]:
                raise ValueError("close admission and drain requests before sealing")
            turns = deepcopy(state["turns"])
            errors = list(state["errors"])
            if any(t.get("delivery") != "consumed_confirmed" for t in turns):
                errors.append("unconfirmed_consumption")
            seal = {"identity": state["identity"].to_dict(), "harbor_trial_id": state["trial_id"],
                    "status": "sealed", "pending_requests": 0, "request_ids": [t["request_id"] for t in turns],
                    "turns_digest": digest(turns), "errors": sorted(set(errors))}
            publish(state["root"] / "trace.json", {"turns": turns, "seal": seal})
            state["sealed"] = True
            return turns, seal

    def close_admission(self):
        """Reject new attempts immediately; drain() still waits for pending ones.

        Split from drain so a cross-shard barrier can close admission on every
        shard before waiting on any of them.
        """
        with self.changed:
            self.accepting = False
            for aid in self.attempts:
                self.close(aid)

    def drain(self, timeout):
        deadline = time.monotonic() + timeout
        self.close_admission()
        with self.changed:
            while any(s["pending"] for s in self.attempts.values()):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("generation still in flight; weights must not change")
                self.changed.wait(remaining)

    def publish_version(self, states, expected):
        from harborrl.platform.policy_pool import validate_pool
        with self.lock:
            if self.accepting or any(not s["sealed"] for s in self.attempts.values()):
                raise ValueError("all attempts must be sealed behind a closed admission barrier")
            self.version = validate_pool(states, expected)
            self.accepting = True
