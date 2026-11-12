# Math training reference pipeline

This optional pipeline adds math tasks, result verification, process rewards
and on-policy teacher scoring to the controller-sandbox source tree. Its
contract is `transformers-math-onpolicy-v1`. It uses the existing local/SSH
training targets and immutable record writer. It does not register a Slime
algorithm or change Native IR, GRPO, sandbox rollout or the controller gateway.
Math generation and scoring currently run in the trainer process.

## Install and run

Use a Linux source checkout with Python 3.10 or newer:

```sh
pip install -e '.[distill]'
harborrl distill train examples/distill/opd.yaml --allow-synthetic --dry-run
harborrl distill train examples/distill/opd.yaml --allow-synthetic
harborrl distill evaluate examples/distill/opd.yaml --allow-synthetic \
  --checkpoint runs/math-opd-smoke/checkpoint-000002.pt \
  --output runs/math-opd-evaluation.json
```

Training requires a Git source checkout for the source audit. Help, curated
import and dry-run do not import Torch, Transformers, Harbor or sandbox SDKs.
Optional dependencies are loaded only for execution. Dry-run validates the
configuration and tasks; it does not verify device capacity or model loading.

The bundled tiny Qwen and byte tokenizer are randomly initialized fixtures.
Fake teacher/PRM and tiny models require explicit `--allow-synthetic`. Student
generation, backward, AdamW updates and checkpointing are real even in this
mode. These fixtures cannot demonstrate improved mathematical ability.

Examples: `reinforce.yaml`, `prm.yaml`, `opd.yaml`, `prm-opd.yaml` and
`local-models.yaml`. All paths resolve relative to the configuration file.
Each run requires a new output directory. Unknown configuration fields fail.

Set `deployment.device: cuda` for a GPU. Set `deployment.world_size: 2` for
two data-parallel processes; global `training.batch_size` must divide evenly.
`training.microbatch_size` controls accumulation within each rank. Gradients
are averaged across ranks, clipped, and applied once per generated batch.
Metrics are per rank, so aggregate their losses by an equal-rank mean.

An explicit `deployment.file` can select the existing shared-filesystem SSH
training target. The same absolute source, data, model and run paths must be
accessible to the target; this command does not copy assets or translate
paths. No deployment file means local execution. Cluster addresses and
credentials belong in private deployment files, not the public examples.

## Data and scoring

Each task JSONL row has exactly `task_id`, `problem_id`, `prompt`, `reference`,
`verifier`, `split`, `source`, and `license`. The bundled 12 arithmetic tasks
are original MIT fixtures: eight train, two validation and two test.
`verifier` is `exact_rational_v1`. References are bounded integers, decimals
or fractions. Duplicate task IDs, cross-split problem IDs and identical
cross-split prompts are rejected. This is not semantic deduplication.

The verifier accepts a bare number, a final `####` answer or a final
`\boxed{...}` number. Correct/incorrect/unparseable/unsupported are distinct;
their rewards are +1/-1/0/0. It never executes generated text and does not
claim symbolic algebra support.

Curated responses can be imported separately:

```sh
harborrl distill import --tasks tasks.jsonl --samples responses.jsonl \
  --out-dir datasets/math-curated
```

Response rows require `task_id`, `response`, `source`, `license`. Import writes
immutable samples, verification sidecars and a manifest. Imported text is
declared evidence with no token probabilities. It is not eligible as an
on-policy batch; the trainer generates fresh student responses from tasks.

PRM adapters split responses at newline token boundaries. `fake` supplies
deterministic test rewards; `equation` verifies bounded binary rational
equations such as `1 + 1 = 2`. Unsupported steps receive zero. A boundary
token belongs to the preceding step. Each token receives the undiscounted
sum of rewards from its step onward. This is a process-reward integration
point, not a trained neural PRM or a general proof checker.

## Objectives and admission

For each valid response token the detached advantage is:

```text
opd:       opd_weight * (teacher_logprob - behavior_logprob)
prm:       prm.weight * step_reward_to_go
prm_opd:   sum of the two terms above
reinforce: outcome_weight * final_verifier_reward
```

`outcome_weight` may additionally contribute to the other methods. The loss
is the negative mean over trajectories of each trajectory's valid-token
mean of `advantage * current_logprob`. OPD implements a sampled-token,
fixed-prefix reverse-KL surrogate, not full-vocabulary KL or the full
sequence-distribution KL gradient including future-prefix costs.

Sampling uses unmodified raw student probabilities: temperature=1, top_p=1,
no top-k, no special-token filtering, no implicit truncation. Each generated
batch permits one optimizer update. The scorer must score those exact token
IDs at the same causal prefixes. Identity, teacher revision, policy version,
request/response digests, finite probabilities and response masks are checked
before backward. Recomputed actor log probabilities must match the frozen
behavior within the configured tolerance. Teacher parameters remain frozen.

`hf_local` loads only local, non-remote-code causal models and fast tokenizers.
Teacher/student tokenizer, vocabulary, special tokens and serialization
identity must match exactly. Model vocabulary must equal tokenizer length;
padded-vocabulary models are currently rejected. Prompts use literal text
with tokenizer special tokens, not automatic chat-template rendering.
Use task prompts appropriate to that serialization. A live teacher capability
probe checks first-token, EOS and causal-prefix scoring consistency.

## Resume and evidence

To resume, copy the configuration, select a new output directory and raise
`training.steps` to the total desired completed steps:

```sh
harborrl distill train resume.yaml --allow-synthetic \
  --resume runs/math-opd-smoke/checkpoint-000001.pt
```

Only load trusted checkpoints. Restore validates configuration, task digest,
tokenizer identity and parameter digest, and restores optimizer state.
Output/deployment, total steps, microbatch size and save interval may change;
objective and global batch settings must remain compatible.

Runs contain effective configuration, launch plan, source audit, runtime
versions, optional teacher probe, immutable token batches, per-rank metrics,
checkpoints and completion/failure records. Evaluation reads a compatible
checkpoint on validation/test data and reports verifier accuracy separately.
No training completion flag implies improved model capability.

## Boundaries

This is a small reference backend using eager float32 Transformers inference,
serial per-sample generation/scoring and full parameter hashing. It supports
CPU and replicated data parallel GPUs, not tensor/pipeline parallelism or
large-model efficiency claims. Start with small models and contexts.

Not included: API teacher collection, LLM judges, a trained PRM, SGLang scoring,
Slime/Megatron OPD integration, replay/off-policy training, SFT from imported
samples, multi-turn environments, automatic dataset licensing checks, or
production model-quality evaluation. Keeping these outside this first phase
makes the math contracts and genuine optimizer path independently testable.
