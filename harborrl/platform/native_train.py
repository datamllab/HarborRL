"""Schema-2 native training dispatch and preflight.

The dispatcher freezes a run directory and materializes the original-task lock
into Slime prompt rows before handing lifecycle control to the existing Slime
launcher. It never rewrites task tests or rewards. A service-mode deployment
routes through the manager-side controller instead of a local child process.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from harborrl.deployment.training_target import local_target
from harborrl.platform.evidence import effective_config, new_run_root, run_environment
from harborrl.trajectories.native import publish, read_json


def _run_root(plan):
    return new_run_root(plan["config"]["output"]["root"])


def doctor(plan):
    """Compatibility entry point for deployment preflight."""
    from harborrl.platform.preflight import doctor as check
    return check(plan)


def _materialize(plan, root):
    from harborrl.platform.evidence import _materialize as prepare
    return prepare(plan, root)


def _source_audit(root):
    from harborrl.platform.evidence import _source_audit as audit
    return audit(root)


def dispatch(plan, *, doctor_only=False):
    deployment = (plan.get("deployment") or {}).get("config")
    if doctor_only:
        checks = doctor(plan)
        explicit = plan.get("deployment", {}).get("source", "embedded-local") != "embedded-local"
        print(json.dumps({"checks": checks,
                          "scope": ("deployment topology, " if explicit else "")
                          + ("math config, locked tasks, training " if
                             plan['config']['execution']['backend'] == 'text_math' else
                             "native config, original tasks, isolated Harbor, training ")
                          + "dependencies and GPU budget; protocol/token semantics "
                            "require the deferred live probe"},
                         indent=2))
        return int(any(not check["ok"] for check in checks))

    checks = doctor(plan)
    failed = [check for check in checks if not check["ok"]]
    if failed:
        print(json.dumps({"failed_checks": failed}, indent=2), file=sys.stderr)
        return 1

    if deployment is not None and deployment["manager"]["mode"] == "service":
        from harborrl.deployment.controller import dispatch_service
        return dispatch_service(plan, deployment, checks=checks)

    root = _run_root(plan)
    root.mkdir(parents=True)
    catalog_path, prompt_path = _materialize(plan, root)
    source = _source_audit(root)
    environment = run_environment(plan, root, catalog_path, prompt_path)
    launch = {"plan": plan, "run_dir": str(root), "source": source,
              "materialized_catalog": str(catalog_path), "prompt_data": str(prompt_path)}
    publish(root / "launch.json", launch)

    target = local_target(plan, environment=environment)
    publish(root / "effective-config.json", effective_config(plan, environment, target.env, source, checks))
    target.launch()
    return target.wait()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m harborrl.platform.native_train PLAN_JSON")
    raise SystemExit(dispatch(read_json(Path(sys.argv[1]))))
