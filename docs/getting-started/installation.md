# Installation

## CPU development environment

Install Python 3.10 or newer and [uv](https://docs.astral.sh/uv/) 0.9 or newer.
From the repository root:

```bash
uv sync --extra dev
uv run harborrl --help
uv run pytest
```

`uv sync` uses the checked-in `uv.lock` and creates `.venv/`. The core environment
does not install CUDA kernels, Docker, model weights, Harbor, Slime, Megatron-LM, or
SGLang.

For an editable pip installation instead:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

## Optional Sandbox runner

The standard OpenSandbox integration requires Python 3.12 or newer in the runner
environment:

```bash
python -m pip install -e '.[sandbox]'
python scripts/sandbox_sdk_contract.py
```

The contract script checks installed APIs and a fake lifecycle; it does not connect
to a platform or prove that a live allocation works. Continue with the
[OpenSandbox guide](../sandbox-rollout.md).

## Native training backends

Native training uses a pinned external Slime/Megatron/SGLang stack. Install it in a
writable location outside the repository:

```bash
bash scripts/bootstrap_backends.sh "/absolute/path/to/harborrl-backends"
. "/absolute/path/to/harborrl-backends/harborrl-backend.env"
```

The generated environment file provides `SLIME_DIR`, `MEGATRON_DIR`, and
`SGLANG_IMAGE` to child processes. The bootstrap is network- and storage-intensive;
inspect the script and obtain site approval before running it on managed machines.

After preparing a valid training and deployment pair, source the generated
environment and resolve the explicit plan:

```bash
uv run harborrl train /absolute/path/to/config.yaml \
  --deployment /absolute/path/to/deployment.yaml \
  --dry-run
```

Remove `--dry-run` only after `harborrl doctor` passes and the operator has approved
resource consumption. `scripts/launch_native.sh` is a convenience wrapper for the
embedded/default-discovery path; it has no `--deployment` option and will bootstrap
the directory passed with `--backends` when its environment file is absent.

## Documentation environment

The documentation toolchain is separate from the Python 3.10 runtime support matrix
and is validated with Python 3.12:

```bash
python -m pip install -e . -r docs/requirements.txt
python -m mkdocs build --strict
python -m mkdocs serve
```

The strict build treats broken internal links, missing navigation entries, and other
MkDocs warnings as errors.
