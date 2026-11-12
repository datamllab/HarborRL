"""Public entry point for the native HarborRL training pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "hello":
        from harborrl.hello import main as hello_main

        return hello_main(argv[1:])
    if argv and argv[0] == "web":
        from harborrl.webui.cli import main as web_main

        return web_main(argv[1:])
    if argv and argv[0] == "distill":
        from harborrl.distill.cli import main as distill_main

        return distill_main(argv[1:])
    if argv and argv[0] in {"options", "validate", "init"}:
        from harborrl.options.cli import main as options_main

        return options_main(argv)
    if argv and argv[0] == "list":
        from harborrl.math_registry import list_components

        choices = {"models", "algorithms", "tasks", "harnesses"}
        if len(argv) != 2 or argv[1] not in choices:
            parser = argparse.ArgumentParser(prog="harborrl list")
            parser.error("choose one of: " + ", ".join(sorted(choices)))
        print(json.dumps(list_components()[argv[1]], indent=2))
        return 0
    if argv and argv[0] == "hub":
        from harborrl.tasks.cli import main as hub_main

        return hub_main(argv[1:])
    parser = argparse.ArgumentParser(
        prog="harborrl", description=__doc__, allow_abbrev=False,
        epilog="New here? harborrl hello. Configuration tools: harborrl options --help; harborrl validate --help; harborrl init --help; harborrl web --help."
    )
    parser.add_argument("-model", help="Slime model preset to select")
    parser.add_argument("-harness", help="agent harness to select")
    parser.add_argument("-task", help="task ID to select from the locked catalog")
    parser.add_argument("-alg", help="training algorithm to select")
    parser.add_argument("command", choices=["train", "doctor"])
    parser.add_argument("config_path", nargs="?", help="YAML configuration file")
    parser.add_argument("--config", dest="config_option", help="YAML configuration file")
    parser.add_argument("--deployment", help="deployment YAML describing the manager, GPU target and rollout workers")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    from harborrl.config import launch_plan, load_config
    from harborrl.deployment.schema import embedded_local, load_deployment
    from harborrl.deployment.schema import summary as deployment_summary

    if args.config_path and args.config_option:
        parser.error(
            "specify the configuration file either positionally or with --config, not both"
        )
    dimensions = (args.model, args.harness, args.task, args.alg)
    config_path = args.config_option or args.config_path
    if config_path is None and any(value is not None for value in dimensions):
        config_path = "config.yaml"
    if config_path is None:
        parser.error("a configuration file is required")

    overrides = list(args.set)
    if args.model is not None:
        overrides.append(f"model.args_file={args.model}")
    if args.harness is not None:
        overrides.append(f"harness.name={args.harness.replace('-', '_')}")
    if args.alg is not None:
        from harborrl.math_registry import MATH_REGISTRY

        matches = [
            spec for spec in MATH_REGISTRY.algorithms.items()
            if args.alg in (spec.id,)
        ]
        if not matches:
            parser.error(
                f"unsupported algorithm: {args.alg}; "
                f"supported algorithms: "
                f"{', '.join(spec.id for spec in MATH_REGISTRY.algorithms.items())}"
            )
        contract = next(iter(matches[0].training_backend_contracts))
        overrides.append(f"training.backend_contract={contract}")

    try:
        config = load_config(
            config_path,
            overrides,
            task_ids=None if args.task is None else [args.task],
        )
        deployment, deployment_source = load_deployment(args.deployment)
        if deployment_source == "embedded-local":
            deployment = embedded_local(
                run_root=str(Path(config["output"]["root"]).resolve()),
                workers=config["harbor"]["workers"],
                expected_gpus=config["deployment"]["num_gpus"],
            )
        plan = launch_plan(config)
        plan["deployment"] = deployment_summary(deployment, deployment_source)
    except (TypeError, ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from harborrl.platform.native_train import dispatch

    return dispatch(plan, doctor_only=args.command == "doctor")


if __name__ == "__main__":
    raise SystemExit(main())
