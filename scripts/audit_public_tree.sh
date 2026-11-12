#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - "$@" <<'PY'
import argparse
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser(description="Audit public files and optional new-commit document history.")
parser.add_argument("--history-base", help="Check every commit after this ancestor, including deleted documents.")
args = parser.parse_args()

public_documents = {
    "CONTRIBUTING.md", "NOTICE.md", "README.md", "README_zh.md", "SECURITY.md",
    "configs/harbor_hub/public_compatibility_summary.md",
    "configs/harbor_hub/public_compatibility_summary_zh.md",
    "docs/agents/environment-check.md", "docs/configuration-wizard.md",
    "docs/deployment.md", "docs/math-distill.md", "docs/sandbox-rollout.md",
    "docs/train-inference-consistency.md", "docs/web-console.md",
    "practices/h_cluster/README.md", "harborrl/webui/static/lucide-LICENSE.txt",
}


def audit_documents(names, location):
    for name in names:
        path = Path(name)
        if path.suffix.lower() not in {".md", ".rst", ".adoc", ".txt"}:
            continue
        task_instruction = (
            path.parts[:3] == ("examples", "native", "tasks")
            and path.name == "instruction.md"
        )
        if name not in public_documents and not task_instruction:
            raise SystemExit(f"document is not approved for publication ({location}): {name}")


root = Path(".")
files = subprocess.check_output(
    ["git", "ls-files", "-z"], text=True
).split("\0")
files = [Path(name) for name in files if name]
if not files:
    raise SystemExit("tracked tree is empty")
audit_documents([path.as_posix() for path in files], "index")

if args.history_base:
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", args.history_base, "HEAD"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode:
        raise SystemExit("history base must resolve to an ancestor of HEAD")
    commits = subprocess.check_output(
        ["git", "rev-list", f"{args.history_base}..HEAD"], text=True
    ).splitlines()
    for commit in commits:
        names = subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", "-z", commit], text=True
        ).split("\0")
        audit_documents([name for name in names if name], commit[:12])
    print(f"public document history audit passed: {len(commits)} new commits")

forbidden_directories = {"backends", "benchmarks", "deploy", "runs", "tools"}
for path in files:
    if path.is_symlink():
        raise SystemExit(f"symlink is not publishable: {path}")
    if path.parts[0] in forbidden_directories:
        raise SystemExit(f"private path remains tracked: {path}")
    if path.stat().st_size > 2_000_000:
        raise SystemExit(f"file exceeds the 2 MB publication limit: {path}")

secret_patterns = {
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(
        rb"(?i)(api[_-]?key|token|password|secret)\s*=\s*['\"][^'\"{}$\n]{12,}"
    ),
}
internal_patterns = {
    "personal path": re.compile(rb"/(mnt/shared-storage-user|home/[A-Za-z0-9_.-]+|root)/"),
    "internal DNS": re.compile(rb"(?i)[a-z0-9.-]+\.(local|internal|svc)\b"),
    "private registry": re.compile(rb"(?i)(registry\.[a-z0-9.-]+/(?:[a-z0-9_.-]+/){2,})"),
    "cluster DNS": re.compile(rb"(?i)[a-z0-9.-]+\.(?:pjlab|p\x6a-lab)\.org\.cn\b"),
}
text_suffixes = {
    ".cfg", ".env", ".github", ".in", ".json", ".md", ".py", ".sh", ".toml",
    ".txt", ".yaml", ".yml", ".lock",
}
for path in files:
    if path.suffix.lower() not in text_suffixes and path.name != "Dockerfile":
        continue
    data = path.read_bytes()
    for label, pattern in secret_patterns.items():
        if pattern.search(data):
            raise SystemExit(f"{path} may contain a {label}")
    for label, pattern in internal_patterns.items():
        for match in pattern.finditer(data):
            value = match.group(0)
            if b"127.0.0.1" in value or b"localhost" in value:
                continue
            raise SystemExit(f"{path} contains {label}: {value.decode(errors='replace')}")

pyproject = Path("pyproject.toml").read_text()
if "tools*" in pyproject or "tools.evaluation" in pyproject:
    raise SystemExit("pyproject still packages the removed evaluation stack")
required = {
    ".github/workflows/ci.yml",
    "NOTICE.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "configs/harbor_hub/manifests.yaml",
    "examples/native/cpu_contract_smoke.sh",
    "examples/native/train_qwen_native.yaml",
    "examples/native/tasks/hello_world/task.toml",
    "scripts/audit_public_tree.sh",
    "scripts/bootstrap_backends.sh",
    "scripts/sglang_semantic_probe.py",
}
missing = sorted(name for name in required if not Path(name).is_file())
if missing:
    raise SystemExit("missing publication files: " + ", ".join(missing))
print(f"public tree audit passed: {len(files)} tracked files")
PY
