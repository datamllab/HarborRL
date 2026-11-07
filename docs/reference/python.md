# Python API

The command-line interface is HarborRL's primary user surface. `harborrl.config`
explicitly declares its public exports. The additional symbols below are curated,
CPU-testable integration contracts used by adapters and tests; they may evolve with
the native contract version.

## Configuration

::: harborrl.config
    options:
      members:
        - load_config
        - launch_plan

## MATH registry

::: harborrl.math_registry
    options:
      members:
        - ModelSpec
        - AlgorithmSpec
        - TaskProviderSpec
        - HarnessSpec
        - MATHSelection
        - select_math
        - validate_combination
        - list_components

## Task providers

::: harborrl.tasks.provider.NativeTaskProvider

::: harborrl.tasks.jsonl_provider.JsonlTaskProvider

## Native trajectory contracts

::: harborrl.trajectories.native
    options:
      members:
        - Identity
        - RewardProfile
        - digest
        - read_json
        - publish
        - validate_v2
        - assemble
        - require_ready

## Training export contracts

::: harborrl.export.contract
    options:
      members:
        - TokenSpan
        - TrajectoryTrainingUnit
        - TrainingGroup
        - TrainingBatch

::: harborrl.export.native
    options:
      members:
        - group_statistics
        - export_training_batch

::: harborrl.export.validate
    options:
      members:
        - validate_token_span
        - validate_trajectory
        - validate_group
        - validate_batch
        - batch_weight_sum

## Rollout runtime

::: harborrl.rollout.runtime
    options:
      members:
        - RuntimeLifecycleError
        - CleanupRequiredError
        - CleanupAdmission
        - RolloutRuntimeConfig
        - RolloutRuntime
        - tokenizer_fingerprint
        - runtime
        - run_embedded
        - publish_native_version
        - current_policy_version
        - close_runtime

## Deployment

::: harborrl.deployment.schema
    options:
      members:
        - DeploymentError
        - destination
        - validate
        - embedded_local
        - load_deployment
        - summary

::: harborrl.deployment.training_target
    options:
      members:
        - TrainingTargetError
        - LocalTrainingTarget
        - SshTrainingTarget
        - local_target

## Consistency configuration and metrics

::: harborrl.consistency.config
    options:
      members:
        - validate
        - require_raw_sampling
        - violations

::: harborrl.consistency.metrics
    options:
      members:
        - statistics
        - merge
