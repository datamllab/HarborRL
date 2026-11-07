# CLI reference

The installed entry point is `harborrl`. Top-level dispatch includes more commands
than the generated `harborrl --help` screen, which currently shows only `train` and
`doctor`.

## Command index

| Command | Purpose | Side effects |
| --- | --- | --- |
| `harborrl hello` | CPU walkthrough with real verifier and fixture tokens | Writes a new artifact directory |
| `harborrl options` | Show implemented components, fields, datasets, presets | None |
| `harborrl validate` | Offline training/deployment validation | None |
| `harborrl init` | Build a configuration bundle or save a draft | Writes a new bundle/draft |
| `harborrl list` | Emit registered MATH components as JSON | None |
| `harborrl hub inspect` | Static native-task inspection | Appends evidence in `--run-dir` |
| `harborrl web` | Start the local single-user console | Opens loopback server; writes workspace |
| `harborrl train` | Resolve and optionally launch native training | Live launch unless `--dry-run` |
| `harborrl doctor` | Run environment/deployment preflight | Live read/probe checks; no training |

## `train` and `doctor`

```text
harborrl [-model PRESET] [-harness ID] [-task ID] [-alg ID]
         {train,doctor} [CONFIG_PATH]
         [--config PATH] [--deployment PATH]
         [--set SECTION.FIELD=VALUE]... [--dry-run]
```

- Supply the config positionally or with `--config`, never both.
- A configuration is always required. Any MATH selector without a path selects
  `./config.yaml`.
- `--set` is repeatable and YAML-decodes each value.
- `--deployment` is optional; discovery falls back to `./deployment.yaml`, then an
  embedded-local topology.
- `--dry-run` prints the JSON launch plan and skips dispatch. It still validates the
  profile and task locks.
- `doctor` never starts training. `doctor --dry-run` only prints the plan because
  dry-run returns before doctor dispatch.

## `hello`

```text
harborrl hello [--out-dir NEW_DIRECTORY]
```

The explicit directory must not exist. Without it, HarborRL creates
`./harborrl-hello-TIMESTAMP`.

## `list`

```text
harborrl list {models,algorithms,tasks,harnesses}
```

Output is always JSON.

## `options`

```text
harborrl options
harborrl options [models|algorithms|tasks|harnesses|datasets|deployments|fields|compat]
                 [--json]
```

The selected JSON response retains the catalog and schema-version envelope.

## `validate`

```text
harborrl validate CONFIG [--deployment PATH] [--check-files] [--json]
```

Without `--check-files`, validation checks structure, semantics, and cross-file
consistency without opening referenced files. File checks are controller-local;
remote paths and live services remain deferred.

Exit status is 0 for `valid` or `valid_with_deferred_checks`, 1 for an invalid
configuration, and 2 for CLI usage errors.

## `init`

```text
harborrl init --out-dir NEW_DIRECTORY
  [--answers YAML | --resume DRAFT_JSON]
  [--draft-file PATH] [--save-draft]
  [--example local|sandbox] [--probe-local] [--prepare-model]
```

`--out-dir` must not exist. Noninteractive use requires `--answers`, `--resume`,
`--example`, or `--save-draft`. Interactive progress is revisioned in a draft after
each answer. See the [configuration CLI guide](../configuration-wizard.md).

## `hub inspect`

```text
harborrl hub inspect --tasks-dir PATH --dataset NAME \
  --revision REVISION --run-dir DIRECTORY
```

The command performs static inspection only. The nested parser may display
`harborrl inspect` in generated usage text; the public invocation includes `hub`.

## `web`

```text
harborrl web [--workspace DIRECTORY] [--port 0..65535] [--no-browser]
```

Defaults are `./harborrl-web` and an automatically selected port. The listener is
fixed to `127.0.0.1`, and one process owns a workspace at a time.
