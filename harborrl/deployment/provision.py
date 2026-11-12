"""Generic provisioning: obtain and release the GPU machine a dispatch needs.

A deployment may declare ``gpu_target.provision`` instead of a fixed
``gpu_target.ssh``. The core owns the lifecycle — ``up`` runs before any
connectivity check, ``down`` runs in the dispatcher's ``finally`` so a failed,
cancelled or crashed run still releases the machine — while the commands and
the machine family they speak to stay site-specific: an optional practice
package implements them, the core never learns those names.

Both commands are opaque argv lists. ``up`` prints one JSON object naming the
machine it produced; ``down`` receives that same object as JSON on stdin.
Nothing secret belongs in either direction: the record carries an ssh
destination, an optional port and identity-file *path*, and an opaque handle
the site may need to release the machine. Commands inherit the dispatcher's
environment, so a site that needs credentials keeps them in its own
environment or a mode-600 file — never in the deployment config.
"""
from __future__ import annotations

import json
import os
import subprocess

from harborrl.trajectories.native import publish

# What one provision record may name. Anything else is a site's own business
# and must stay inside its opaque handle.
RECORD_FIELDS = {"host", "user", "port", "identity", "handle"}
# Error text is diagnostic, not evidence: keep it short and never echo a
# command's whole output into a record that outlives the run.
ERROR_TAIL = 400


class ProvisionError(RuntimeError):
    """The provisioning contract was not honoured."""


def _run(argv, *, stdin_json=None, timeout=None):
    return subprocess.run(
        list(argv),
        input=None if stdin_json is None else json.dumps(stdin_json),
        capture_output=True, text=True, errors="replace", timeout=timeout, check=False,
        env=dict(os.environ))


def _tail(text):
    text = (text or "").strip()
    return text[-ERROR_TAIL:]


def _valid_record(record, label):
    """Validate one candidate record; raises ProvisionError when unusable."""
    if not isinstance(record, dict):
        raise ProvisionError(f"{label} record must be a JSON object")
    unknown = sorted(set(record) - RECORD_FIELDS)
    if unknown:
        raise ProvisionError(
            f"{label} record names unsupported fields {unknown}; keep site data in 'handle'")
    host = record.get("host")
    if not isinstance(host, str) or not host.strip() or any(
            character.isspace() for character in host) or host.startswith("-"):
        raise ProvisionError(f"{label} record requires an ssh destination host")
    user = record.get("user")
    if user is not None and (not isinstance(user, str) or not user.strip()
                             or user.startswith("-")):
        raise ProvisionError(f"{label} record user must not look like a flag")
    port = record.get("port")
    if port is not None and (type(port) is not int or not 0 < port < 65536):
        raise ProvisionError(f"{label} record port must be an integer in 1..65535")
    identity = record.get("identity")
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ProvisionError(f"{label} record identity must be a path")
    handle = record.get("handle")
    if handle is not None and not isinstance(handle, str):
        raise ProvisionError(f"{label} record handle must be a string")
    return record


def _record(stdout, label):
    """Parse the JSON object a provision command reports.

    A site submission tool is chatty and keeps logging after it prints the
    machine, so the record is not necessarily the last line — a trailing log
    entry may itself be JSON with a ``host`` key, and taking it would dispatch
    onto the wrong machine. Scan backwards for the last line that validates as
    a complete record; a stream with no such line fails closed rather than
    guessing.
    """
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        raise ProvisionError(f"{label} reported no record on stdout")
    first_error = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        try:
            return _valid_record(candidate, label)
        except ProvisionError as exc:
            # Keep looking: a chatty trailing log object is not the record.
            first_error = first_error or exc
    raise first_error or ProvisionError(f"{label} did not report a JSON record")


def ssh_spec(record):
    """The normalized ssh spec a provision record licenses."""
    spec = {"host": record["host"]}
    for key in ("user", "port", "identity"):
        if record.get(key) is not None:
            spec[key] = record[key]
    return spec


def up(provision, *, evidence_path=None):
    """Run ``up`` and return the normalized machine record.

    The record is published to ``evidence_path`` (when given) before dispatch
    proceeds, so the machine that a run actually used — and the handle needed
    to release it — outlives the process that obtained it.

    A publish failure releases the machine before raising. Nothing downstream
    would ever learn the handle, so leaving it running would bill an
    allocation that no run directory can account for.
    """
    try:
        result = _run(provision["up"], timeout=provision["timeout_s"])
    except subprocess.TimeoutExpired as exc:
        raise ProvisionError(
            f"gpu_target.provision.up exceeded {provision['timeout_s']}s") from exc
    except OSError as exc:
        raise ProvisionError(f"gpu_target.provision.up could not start: {exc}") from exc
    if result.returncode != 0:
        raise ProvisionError(
            f"gpu_target.provision.up failed with {result.returncode}: "
            f"{_tail(result.stderr)}")
    record = _record(result.stdout, "gpu_target.provision.up")
    if evidence_path is not None:
        try:
            publish(evidence_path, {"action": "up", "record": record})
        except Exception as exc:  # noqa: BLE001 — any evidence failure leaks otherwise
            outcome, detail = down(provision, record)
            raise ProvisionError(
                f"cannot record the provisioned machine {record.get('handle')!r} "
                f"at {evidence_path}: {exc}; release reported {outcome} ({detail})"
            ) from exc
    return record


def down(provision, record, *, evidence_path=None):
    """Run ``down`` for a machine this dispatch obtained.

    Returns ``"released"``, ``"undeclared"`` (no down command: the machine
    lives out its own declared lifetime) or ``"failed"``; the release record
    is published either way so a leaked machine is visible, never silent.
    """
    command = provision.get("down")
    if command is None:
        outcome, detail = "undeclared", "no gpu_target.provision.down declared"
        result = None
    else:
        try:
            result = _run(command, stdin_json=record, timeout=provision["timeout_s"])
            outcome = "released" if result.returncode == 0 else "failed"
            detail = (f"exit {result.returncode}: {_tail(result.stderr)}"
                      if result.returncode else "released")
        except subprocess.TimeoutExpired:
            outcome, detail = "failed", f"exceeded {provision['timeout_s']}s"
        except OSError as exc:
            outcome, detail = "failed", f"could not start: {exc}"
    if evidence_path is not None:
        payload = {"action": "down", "outcome": outcome, "detail": detail,
                   "record": record}
        try:
            publish(evidence_path, payload)
        except ValueError:
            # Release evidence is advisory; the machine's fate is what the
            # outcome reports, not whether the record could be written.
            pass
    return outcome, detail
