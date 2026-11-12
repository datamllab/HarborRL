"""Math-only data import, on-policy reference training and held-out evaluation."""
import argparse
import json
import os
import sys
from pathlib import Path

from harborrl.deployment.schema import embedded_local, load_deployment
from harborrl.deployment.training_target import LocalTrainingTarget, SshTrainingTarget
from harborrl.platform.evidence import _source_audit
from harborrl.trajectories.native import digest, publish

from .config import load
from .data import import_samples, load_tasks


def plan(config, *, allow_synthetic=False, resume=None):
    tasks = load_tasks(config["tasks"]["path"])
    if not any(task["split"] == "train" for task in tasks):
        raise ValueError("No training split")
    root = Path(config["output"]["directory"])
    deployment, source = (load_deployment(config["deployment"]["file"]) if config["deployment"]["file"]
                          else (embedded_local(), "embedded-local"))
    gpu = deployment["gpu_target"]
    if config["deployment"]["device"] == "cuda" and source != "embedded-local" and gpu["expected_gpus"] < config["deployment"]["world_size"]:
        raise ValueError("GPU target budget below requested world size")
    python = gpu["train_python"] if source != "embedded-local" else sys.executable
    command = [python]
    if config["deployment"]["world_size"] > 1:
        command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={config['deployment']['world_size']}", "--module", "harborrl.distill.runner"]
    else:
        command += ["-m", "harborrl.distill.runner"]
    command += ["--config", str(root / "config.json")]
    if allow_synthetic:
        command.append("--allow-synthetic")
    if resume:
        command += ["--resume", str(Path(resume).resolve())]
    return {"backend_contract": config["backend_contract"], "method": config["method"], "command": command,
            "config": config, "config_digest": digest(config), "tasks_digest": digest(tasks),
            "deployment_source": source, "gpu_target": gpu, "connectivity": deployment["connectivity"],
            "scope": "reference math pipeline; live dependencies, model compatibility and capacity remain unchecked"}


def launch(value):
    root = Path(value["config"]["output"]["directory"])
    if root.exists():
        raise ValueError("Run output must be a new directory; resume into a different output")
    root.mkdir(parents=True)
    publish(root / "config.json", value["config"])
    publish(root / "plan.json", value)
    _source_audit(root)
    gpu = value["gpu_target"]
    environment = {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "TOKENIZERS_PARALLELISM": "false"}
    if gpu["transport"] == "ssh":
        target = SshTrainingTarget(gpu["ssh"], source_dir=gpu["source_dir"], run_dir=root,
                                  environment_file=gpu.get("environment_file"),
                                  command_timeout=value["connectivity"]["command_timeout_s"] + 30,
                                  ssh_timeout=value["connectivity"]["ssh_timeout_s"])
        target.write_launcher(value["command"], {**environment, "PYTHONPATH": gpu["source_dir"]})
        import signal
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def cancel(_signum, _frame):
            target.cancel()
            raise KeyboardInterrupt
        try:
            for sig in previous:
                signal.signal(sig, cancel)
            target.launch()
            return target.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    from harborrl.config import ROOT
    env = {**os.environ, **environment}
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT), env.get("PYTHONPATH")]))
    target = LocalTrainingTarget(value["command"], cwd=ROOT, env=env)
    target.launch()
    try:
        return target.wait()
    finally:
        if target.child.poll() is None:
            target.cancel()
            target.child.wait(timeout=30)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="harborrl distill", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    imported = commands.add_parser("import", help="Import curated responses into immutable sample/verification records")
    imported.add_argument("--tasks", required=True)
    imported.add_argument("--samples", required=True)
    imported.add_argument("--out-dir", required=True)
    for name in ("train", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("config")
        command.add_argument("--allow-synthetic", action="store_true")
        if name == "train":
            command.add_argument("--dry-run", action="store_true")
            command.add_argument("--resume", help="Trusted checkpoint; use a new output directory")
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            print(json.dumps(import_samples(args.tasks, args.samples, args.out_dir), indent=2))
            return 0
        config = load(args.config, allow_synthetic=args.allow_synthetic)
        if args.command == "evaluate":
            from .runner import evaluate
            result = evaluate(config, args.checkpoint)
            publish(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
        value = plan(config, allow_synthetic=args.allow_synthetic, resume=args.resume)
        if args.dry_run:
            print(json.dumps(value, indent=2))
            return 0
        return launch(value)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
