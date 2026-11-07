# Compatibility and evidence

HarborRL uses three different sources of compatibility information. They answer
different questions.

## Component registry

The runtime registry is authoritative for selectable MATH components. Query it with:

```bash
harborrl options compat --json
harborrl options models --json
harborrl options algorithms --json
harborrl options tasks --json
harborrl options harnesses --json
```

Registry compatibility checks protocols, backend contracts, execution backends,
Harbor versions, group size, and raw-logprob requirements. A passing combination is
eligible for configuration; it is not live deployment evidence.

## Dataset metadata

`harborrl options datasets --json` reports each known dataset's status and whether
its task files are locally bundled. Planned datasets are not ready to train, and an
installed wheel does not contain repository examples.

HarborRL never downloads a dataset as a side effect of catalog browsing, validation,
or configuration generation.

## Harbor Hub compatibility summary

The repository's
[public compatibility summary](https://github.com/datamllab/HarborRL/blob/main/configs/harbor_hub/public_compatibility_summary.md)
records scoped evidence about candidate Harbor task collections. Read its evidence
level literally:

- **static** evidence covers format or metadata inspection;
- **dynamic** evidence covers the explicitly recorded execution scope; and
- neither automatically proves end-to-end RL training, learning improvement, or
  support for every task in a collection.

## Acceptance levels

Use the [environment verification protocol](../agents/environment-check.md) to report
one of `cpu_development`, `deployment_preflight`, or `training`. Do not promote a CPU
test or dry run to a live-training claim.
