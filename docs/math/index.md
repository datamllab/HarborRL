# MATH components

MATH means **Models × Algorithms × Tasks × Harnesses**. HarborRL represents each
dimension with an explicit registry entry and rejects incompatible combinations
before launch.

## Current registry

| Dimension | ID | Implementation | Key constraint |
| --- | --- | --- | --- |
| Model | `qwen` | Qwen adapter, SGLang serving | `model.args_file` starts with `qwen` |
| Algorithm | `grpo` | Slime GRPO | Complete groups of at least two samples |
| Task provider | `native_harbor` | JSON catalog | Native Harbor task locks |
| Task provider | `harbor_jsonl` | JSONL catalog | One complete task lock on every line; no blanks |
| Harness | `claude_code` | Claude Code | Harbor `0.23.0`, exact CLI version |

Query the installed package rather than relying on a static list:

```bash
harborrl list models
harborrl list algorithms
harborrl list tasks
harborrl list harnesses
harborrl options compat --json
```

## Compatibility checks

For a selected configuration, HarborRL verifies that:

- the model and algorithm both support the backend contract;
- the task provider supports `execution.backend`;
- the selected Harbor version is supported by the harness;
- model and harness tool protocols match;
- the model gateway exposes the harness request protocol;
- the sample group meets the algorithm's minimum size; and
- audited raw log probabilities are enabled when the algorithm requires them.

These checks establish interface compatibility, not model quality, memory fit, or a
successful live run.

## Explore each dimension

<div class="harbor-grid" markdown>

<div class="harbor-card" markdown>

### [Models](models.md)

Tokenizer/template identity, SGLang serving, and Qwen tool-call parsing.

</div>

<div class="harbor-card" markdown>

### [Algorithms](algorithms.md)

GRPO group requirements, advantage computation, and pinned backend hooks.

</div>

<div class="harbor-card" markdown>

### [Tasks](tasks.md)

Native Harbor format, immutable task locks, and verifier reward profiles.

</div>

<div class="harbor-card" markdown>

### [Harnesses](harnesses.md)

Claude Code profile validation, credentials, controls, and session audit.

</div>

</div>
