# Train / inference consistency

Consistency checks compare log probabilities of the **same response tokens at
the same prefix**, before an optimizer update. This measures backend drift;
it does not measure model quality or full-distribution KL divergence.

The three policies are the observed behavior `q`, frozen pre-update trainer
`p0`, and current trainer `pθ`. Backend drift is `log p0 - log q`; optimization
drift is `log pθ - log p0`. The Native loss remains unchanged. No TIS, clipping
correction, or new training objective is introduced.

## Service sampling lock

The service client computes the sampling digest from the original group request,
normalizing equivalent JSON probabilities (`1` and `1.0`) the same way as the
executor and Gateway. Each slot descriptor and trajectory identity must match
that requested digest. Native readiness validation then binds every turn's
sampling evidence to the trajectory identity. Internally consistent manifests
from a different distribution are rejected before conversion to training samples.

## Configuration

Both Native schema 2 and the Transformers math configuration accept an optional
`consistency` mapping. Its default mode is `off`. Existing configurations remain
valid. `audit` records numerical violations; `enforce` rejects violations before
training or serving admission. Invalid evidence always rejects, including in
audit mode. Enforce requires an explicit `max_abs_delta`. Calibrate tolerances
for the actual model, precision, parallel layout and kernels.

```yaml
consistency:
  mode: audit
  max_abs_delta: null
  max_mean_abs_delta: null
  bad_token_delta: null
  max_bad_fraction: 0.0
  worst_tokens: 8
```

Native overrides use the existing `--set consistency.mode=audit` interface.
For enforce, configure a deployment-specific `max_abs_delta` in YAML before
loading the configuration. The configuration is preserved in launch evidence.

## Native Megatron / SGLang

The enabled launcher runs `harborrl.consistency.train`, which uses the pinned
Slime training loop and its supported actor factory. It never edits upstream
Slime. The existing pre-update forward supplies actual packed-batch `p0` values.
Immutable serving sidecars bind token IDs, float32-normalized behavior scores,
response masks, policy version, tokenizer and template identity. Missing or
mismatched sidecars reject the batch.

Before each weight synchronization, the actor evaluates two fixed token
fixtures using Megatron. Every synchronized SGLang engine then teacher-forces
those exact sequences. Generation-time and post-probe versions must both match.
Each engine must satisfy the numerical limits independently. Checks also cover
the final sync, even when no subsequent rollout occurs. Probe records are reused
only for the same pool, configuration and immutable reference. A fixed canary
is a smoke check; actual rollout-batch checks provide broader coverage.

Supported initial scope: dense models, synchronous full NCCL synchronization,
PP=CP=1, with TP and DP. Temperature and top-p must be 1; top-k must be disabled.
Critic, old-actor queues, MoE, release-train and routing replay are rejected.
Transformed sampling would require a separate definition of behavior scores.

`RUN_DIR/consistency` contains sync references, pool admissions, serving-turn
sidecars and `batch-*.json` reports. References include a bounded sampled
parameter fingerprint per rank; this is **not** a full weight checksum. Reports
contain valid-token-weighted absolute/signed errors, log-ratio range, bad-token
fraction, bounded worst positions and histogram percentile **upper bounds**.
They do not copy raw prompt or response text. TP duplicate counts are excluded.
Distributed validation collects rank errors before all ranks reject.

## Serving text and token evidence

The text passed to the model-output parser must equal the complete visible
decoding of the generation token IDs. Prefix agreement and whitespace
normalization do not prove that the responses match. The only accepted omission
is an exact terminal stop string declared by the request and explicitly reported
as matched by SGLang's stop finish reason. Extra text before or after that stop
is rejected. The original generation IDs and logprobs remain unchanged.

## Transformers mathematical reference

The optional gate teacher-forces the frozen batch under `no_grad` before any
backward call and uses the same metrics and report format. The pre-existing
`training.behavior_tolerance` check remains active; audit does not loosen it.
Fake teacher scores never stand in for observed student behavior.

## Validation and limitations

Run `pytest -q tests/test_consistency.py` for the dependency-free checks. With
Torch and Transformers installed, run `pytest -q tests/test_consistency_torch.py
tests/harborrl/test_distill.py` for real optimizer rejection, parameter changes,
two-rank failure propagation and the math pipeline tests.

GPU acceptance must additionally exercise the locked Megatron / SGLang versions
on the target hardware: initial sync, actual rollout tokens, optimizer update,
final sync and checkpoint resume. Record dtype, attention backend, TP/DP, model
revision and measured error distribution. CPU tests and mocked HTTP responses
cannot certify SGLang input-logprob behavior or CUDA numerical tolerances.
This implementation has not yet passed that live GPU acceptance gate.

H-cluster relay topology remains a deployment practice. No H-cluster hostnames,
private paths or machine-specific numerical thresholds belong in this module.
