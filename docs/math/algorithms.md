# Algorithms

## GRPO

`grpo` is the only registered training algorithm. It uses the
`slime-v0.3.2-native-v1` backend contract and HarborRL's native advantage, conversion,
loss, and post-processing hooks.

```yaml
training:
  backend_contract: slime-v0.3.2-native-v1
  learning_rate: 0.000001
  num_rollout: 1
  save_interval: 1

sampling:
  group_size: 2
  groups_per_batch: 1
  max_attempts: 2
  max_context: 8192
  max_tokens: 1024
```

## Group contract

Every admitted group must:

- contain at least two completed trajectories;
- use the same locked policy version and immutable task identity;
- carry finite verifier rewards (equal-reward groups are valid and produce zero
  advantages);
- carry audited raw model log probabilities and valid loss masks; and
- pass trajectory, reward-receipt, and consumption-evidence validation.

Advantages are normalized within each complete group. The training reduction is
global-sum, and export validates the total token weight before a batch reaches the
backend.

`sampling.max_attempts` bounds retries for invalid attempts; it is not a retry budget
for low rewards.

## Backend hooks

HarborRL maps the contract to pinned Slime hook implementations for:

- rollout generation;
- sample-to-training-data conversion;
- advantage and return computation;
- clipped policy-gradient loss; and
- rollout-data post-processing.

The launcher validates the hook assignment and backend versions. It does not edit
the upstream Slime checkout.

## Limits

- DAPO and other algorithms are not registered.
- Only the Slime v0.3.2 native hook contract is accepted.
- CPU tests validate numerical and structural contracts, not live distributed GPU
  behavior.
