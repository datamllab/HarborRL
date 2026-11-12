"""Slime v0.3.2 rollout function hook for complete Native Harbor groups.

This module is the training-side adapter: it maps Slime args and Sample
objects onto the framework-free rollout core. When the manager runs the
RolloutService (P3), groups are requested over HTTP instead of executing in
this process; conversion to Slime samples is identical either way.
"""
from __future__ import annotations

import asyncio
import math
import os
from pathlib import Path

from harborrl.backends.slime_v032.versions import lock_policy_version
from harborrl.export.native import export_training_batch
from harborrl.rollout.runtime import (
    RolloutRuntimeConfig,
    cancel_and_wait,
    close_runtime,
    current_policy_version,
    digest,
    run_group,
    runtime,
    task_entry,
)
from harborrl.trajectories.native import canonical_sampling


def sampling_params(args):
    max_new_tokens = getattr(args, "rollout_max_response_len", None)
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("native rollout requires a positive rollout_max_response_len")
    return canonical_sampling({
        "temperature": getattr(args, "rollout_temperature", 1.0),
        "top_p": getattr(args, "rollout_top_p", 1.0),
        "top_k": getattr(args, "rollout_top_k", -1),
        "max_new_tokens": max_new_tokens,
        "stop": list(getattr(args, "rollout_stop", None) or []),
    })


def runtime_config_from_args(args):
    """Map Slime runtime args onto the extracted rollout core config."""
    endpoint = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    engine_count = int(args.rollout_num_gpus) // int(args.rollout_num_gpus_per_engine)
    limit = max(1, int(args.sglang_server_concurrency) * engine_count)
    return RolloutRuntimeConfig.from_environment(
        os.environ, endpoint=endpoint, concurrency_limit=limit,
        group_size=int(args.n_samples_per_prompt))


def slime_tokenizer_loader(args):
    """The single Slime dependency at the rollout boundary: the tokenizer."""
    from slime.utils.processing_utils import load_tokenizer

    return lambda: load_tokenizer(args.hf_checkpoint, trust_remote_code=True)


def group_index_of(group):
    indexes = {getattr(sample, "group_index", None) for sample in group}
    if len(indexes) != 1 or type(next(iter(indexes), None)) is not int:
        raise ValueError("Slime must provide one group identity before native generation")
    return indexes.pop()


def assert_locked_policy_versions(ir_groups, policy_version):
    """Fail closed if any generation escaped the locked weight version."""
    for group in ir_groups:
        for ir in group:
            if not isinstance(ir, dict):
                raise TypeError("native rollout did not return trajectory evidence")
            version = ir.get("identity", {}).get("policy_version")
            if policy_version is not None and version != policy_version:
                raise ValueError("native rollout observed a stale policy version")


async def generate_group(args, group, sampling_params_, rollout_id, evaluation=False):
    """Run one complete native group; evaluation is intentionally unsupported."""
    from harborrl.math_registry import NATIVE_TASK_PROVIDER

    if evaluation:
        raise ValueError("native training rollout cannot be reused as evaluation")
    if len(group) != args.n_samples_per_prompt:
        raise ValueError("native rollout group is incomplete before launch")
    entries = [task_entry(sample) for sample in group]
    if any(entry != entries[0] for entry in entries[1:]):
        raise ValueError("native rollout group mixes task locks")
    entry = entries[0]
    report = NATIVE_TASK_PROVIDER.adapter.inspect(
        entry["path"], expected_digest=entry["task_digest"]
    )
    if report["status"] != "NEEDS_PROBE":
        raise ValueError(f"native task preflight failed: {report['reasons']}")
    stop = list(sampling_params_.get("stop") or [])
    sampling_digest = digest(canonical_sampling({
        "temperature": sampling_params_.get("temperature"),
        "top_p": sampling_params_.get("top_p"),
        "top_k": sampling_params_.get("top_k"),
        "max_new_tokens": sampling_params_.get("max_new_tokens"),
        "stop": stop,
    }))
    config = runtime_config_from_args(args)
    runtime_ = await runtime(config, tokenizer_loader=slime_tokenizer_loader(args))
    policy_version = current_policy_version(runtime_)
    trajectories = await run_group(
        config, runtime_, entry=entry, group_index=group_index_of(group),
        rollout_id=rollout_id, sampling_digest=sampling_digest,
        policy_version=policy_version,
    )
    return bind_group_samples(group, trajectories)


def bind_group_samples(group, trajectories):
    """Attach trajectory evidence to Slime samples; order defines the slots."""
    output = []
    for sample, ir in zip(group, trajectories):
        sample.metadata["native_ir"] = ir
        sample.response = "native Harbor trial"
        sample.reward = {"score": ir["evaluation"]["training_reward"]}
        sample.status = sample.Status.COMPLETED
        try:
            sample.policy_version = int(ir["identity"]["policy_version"])
        except ValueError as exc:
            raise ValueError("native policy version must be an integer in Slime") from exc
        output.append(sample)
    return output


async def run_service_groups(args, groups, rollout_id):
    """Request complete groups from the manager-side RolloutService (P3)."""
    from harborrl.backends.slime_v032.rollout_service_client import (
        RolloutServiceClient,
        fetch_pool_states,
        load_group_result,
        service_group_request,
    )

    token_file = os.environ.get('HARBORRL_ROLLOUT_SERVICE_TOKEN_FILE')
    token = Path(token_file).read_text().strip() if token_file else os.environ.get('HARBORRL_ROLLOUT_SERVICE_TOKEN', '')
    client = RolloutServiceClient(
        os.environ["HARBORRL_ROLLOUT_SERVICE_URL"], os.environ["RUN_ID"],
        token)
    router = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    # Policy lifecycle (§7): the GPU job publishes its serving endpoint, then
    # the manager admits the version the complete pool actually serves. The
    # manager never opens a version some engine has not reached yet.
    await asyncio.to_thread(client.announce_serving, router)
    states = await asyncio.to_thread(fetch_pool_states, router)
    from harborrl.consistency.config import environment
    from harborrl.consistency.sglang import admit_pool
    if environment()['mode'] != 'off':
        await asyncio.to_thread(admit_pool, states, os.environ["RUN_DIR"])
    await asyncio.to_thread(client.activate_policy, states, states[0]["weight_version"])
    # The manager's activation publishes the locked policy-pool record into
    # the shared run directory; lock the same version before requesting.
    policy_version = lock_policy_version(args)
    params = sampling_params(args)
    async def request_group(group):
        request = service_group_request(group, params, rollout_id, policy_version)
        payload = await asyncio.to_thread(client.submit_group, request)
        # The HTTP layer returns a status summary; completed work carries the
        # GroupResult itself.
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            from harborrl.backends.slime_v032.rollout_service_client import (
                ServiceRolloutError,
            )
            raise ServiceRolloutError(
                f"group {request['group_id']} is {payload.get('status')!r}: "
                f"{payload.get('error', '')}")
        irs = load_group_result(client.run_id, request, result, allow_synthetic=False)
        bind_group_samples(group, irs)
        return irs

    pending = [asyncio.create_task(request_group(group)) for group in groups]
    try:
        results = await asyncio.gather(*pending, return_exceptions=True)
        # Retain independent groups' evidence, but never export a partial batch.
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results
    except BaseException:
        # HTTP workers keep running until the manager acknowledges cancellation.
        try:
            await asyncio.to_thread(client.cancel_run)
        finally:
            await asyncio.gather(*pending, return_exceptions=True)
        raise


async def run_native_groups(args, groups, policy_version, rollout_id):
    """Run every group concurrently and return only complete IR evidence."""
    params = sampling_params(args)
    if os.environ.get("HARBORRL_ROLLOUT_SERVICE_URL"):
        ir_groups = await run_service_groups(args, groups, rollout_id)
        assert_locked_policy_versions(ir_groups, policy_version)
        return ir_groups
    pending = [asyncio.create_task(
        generate_group(args, group, params, rollout_id=rollout_id)) for group in groups]
    batch = asyncio.gather(*pending)
    try:
        # Keep parent cancellation from being forwarded twice during cleanup.
        results = await asyncio.shield(batch)
    except BaseException:
        # Groups own their slot cleanup; finish it before shutting down the gateway.
        await cancel_and_wait(pending, batch)
        # Close the cached runtime here unless an embedded run owns the loop:
        # run_embedded's failure handler closes it after the loop unwinds, so
        # the original error is what propagates.
        from harborrl.rollout import runtime as _runtime
        loop = getattr(_runtime, "_EMBEDDED_LOOP", None)
        if not (loop is not None and loop.is_running()):
            close_runtime()
        raise
    ir_groups = []
    for group in results:
        trajectories = []
        for sample in group:
            metadata = getattr(sample, "metadata", None)
            ir = metadata.get("native_ir") if isinstance(metadata, dict) else None
            if not isinstance(ir, dict):
                raise TypeError("native rollout did not return trajectory evidence")
            trajectories.append(ir)
        ir_groups.append(trajectories)
    assert_locked_policy_versions(ir_groups, policy_version)
    return ir_groups


def to_slime_samples(batch, ir_groups=None, *, sample_factory=None):
    """Expand one Slime training sample per Native policy turn."""
    if sample_factory is None:
        from slime.utils.types import Sample

        def sample_factory(**fields):
            fields["status"] = Sample.Status.COMPLETED
            return Sample(**fields)

    output = []
    index = 0
    for group_index, (group, trajectories) in enumerate(zip(batch.groups, ir_groups or [None] * len(batch.groups), strict=True)):
        by_id = {
            ir["identity"]["trajectory_id"]: ir
            for ir in (trajectories or [])
        }
        group_samples = []
        for unit in group.trajectories:
            for span in unit.turns:
                group_samples.append(sample_factory(
                    group_index=group_index,
                    index=index,
                    rollout_id=unit.rollout_id,
                    tokens=list(span.tokens),
                    response_length=span.response_length,
                    response="native Harbor trial",
                    reward={"score": unit.reward},
                    loss_mask=list(span.loss_mask[span.prompt_length:]),
                    rollout_log_probs=list(span.old_logprobs),
                    weight_versions=[span.policy_version],
                    metadata={
                        "native_ir": by_id.get(unit.trajectory_id),
                        "native_turn": span.turn_index,
                    },
                ))
                index += 1
        output.append(group_samples)
    if any(sample.metadata.get("native_ir") is None for group in output for sample in group):
        raise ValueError("Slime turn samples require native IR metadata")
    return output


def collect_metrics(batch, *, policy_version=None):
    rewards = [
        unit.reward
        for group in batch.groups
        for unit in group.trajectories
    ]
    metrics = {
        "native_group_count": float(len(batch.groups)),
        "native_trajectory_count": float(len(rewards)),
        "native_turn_count": float(sum(
            len(unit.turns) for group in batch.groups for unit in group.trajectories
        )),
        "native_reward_mean": math.fsum(rewards) / len(rewards),
        "native_reward_std": math.sqrt(math.fsum(
            (reward - math.fsum(rewards) / len(rewards)) ** 2 for reward in rewards
        ) / len(rewards)),
    }
    if policy_version is not None:
        metrics["native_policy_version"] = policy_version
    return metrics


def generate_rollout(
    args,
    rollout_id,
    data_source,
    evaluation=False,
    *,
    runner=None,
    output_factory=None,
    sample_factory=None,
):
    """Slime ``--rollout-function-path`` hook; groups stay atomic."""
    if evaluation:
        raise NotImplementedError("Native evaluation rollout is not part of v1")
    if type(rollout_id) is not int or rollout_id < 0:
        raise ValueError("Slime must provide a nonnegative rollout id")
    batch_size = getattr(args, "rollout_batch_size", None)
    expected_size = getattr(args, "n_samples_per_prompt", None)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("native rollout requires a positive rollout batch size")
    if type(expected_size) is not int or expected_size < 2:
        raise ValueError("native rollout requires a complete group size")
    groups = [list(group) for group in data_source.get_samples(batch_size)]
    if len(groups) != batch_size:
        raise ValueError("native rollout batch is incomplete")
    if any(len(group) != expected_size for group in groups):
        raise ValueError("native rollout group is incomplete before launch")
    if runner is None:
        async def runner(args_, groups_, policy_version):
            return await run_native_groups(args_, groups_, policy_version, rollout_id)
    ir_groups = asyncio.run(runner(args, groups, None))
    policy_version = lock_policy_version(args)
    assert_locked_policy_versions(ir_groups, policy_version)
    batch = export_training_batch(
        ir_groups,
        group_size=expected_size,
        rollout_id_base=rollout_id * batch_size,
    )
    if output_factory is None:
        from slime.rollout.base_types import RolloutFnTrainOutput
        output_factory = RolloutFnTrainOutput
    return output_factory(
        samples=to_slime_samples(batch, ir_groups, sample_factory=sample_factory),
        metrics=collect_metrics(batch, policy_version=policy_version),
    )
