---
hide:
  - toc
---

<div class="harbor-hero" markdown>

![HarborRL](assets/HarborRL-title.jpg)

# Auditable agentic reinforcement learning

HarborRL connects verifiable Harbor tasks, a tool-using agent harness, model serving,
trajectory evidence, and GRPO updates in one reproducible pipeline. Start on a CPU,
then carry the same contracts into a native GPU deployment.

[Run the CPU walkthrough](tutorials/hello-world.md){ .md-button .md-button--primary }
[Understand the architecture](getting-started/index.md){ .md-button }

</div>

## What is implemented today

HarborRL validates combinations before launch. The current registry is intentionally
small and explicit:

| MATH dimension | Registered implementation | Contract |
| --- | --- | --- |
| **Model** | Qwen (`qwen`) | Qwen tool calls served by SGLang |
| **Algorithm** | GRPO (`grpo`) | Slime `v0.3.2` native hooks |
| **Tasks** | Native Harbor JSON and JSONL catalogs | Content-digested, verifier-backed tasks |
| **Harness** | Claude Code (`claude_code`) | Harbor `0.23.0`, Anthropic Messages v1 |

!!! info "The registry is the support boundary"

    Names mentioned by upstream task collections or roadmap material are not supported
    merely because they exist elsewhere. Use `harborrl list ...` or
    `harborrl options ...` to query the running installation.

<div class="harbor-grid" markdown>

<div class="harbor-card" markdown>

### CPU-first onboarding

Run a real verifier and export a two-sample GRPO batch without model weights, CUDA,
Docker, Harbor, Slime, or SGLang.

[Hello World](tutorials/hello-world.md)

</div>

<div class="harbor-card" markdown>

### Strict configuration

Training schema 2 and deployment schema 1 reject unknown fields and inconsistent
MATH, topology, GPU, and protocol selections.

[Configuration](training/configuration.md)

</div>

<div class="harbor-card" markdown>

### Evidence, not assumptions

Task digests, policy versions, response consumption, verifier receipts, and token
evidence are checked before a trajectory reaches training.

[Run artifacts](reference/artifacts.md)

</div>

</div>

## First five minutes

```bash
git clone https://github.com/datamllab/HarborRL.git
cd HarborRL
uv sync --extra dev
uv run harborrl hello --out-dir ./harborrl-hello
```

The walkthrough creates a native task, executes one failing and one passing scripted
attempt, runs the actual verifier, and exports `training-batch.json`. Its token evidence
is a fixture: it does **not** perform inference or update model weights.

Continue with the [quick start](getting-started/quickstart.md), or inspect the exact
[CLI surface](reference/cli.md).

## Pipeline at a glance

```text
locked Harbor tasks
        │
        ▼
Claude Code harness ──► Harbor runner ──► verifier reward
        │                     │
        ▼                     ▼
policy gateway / SGLang   audited trajectory evidence
                              │
                              ▼
                    complete GRPO groups
                              │
                              ▼
                    Slime native training
```

`--dry-run`, offline validation, CPU tests, and live training establish different
levels of evidence. The deployment guides label those boundaries instead of treating
a parsed configuration as proof that a GPU run will succeed.
