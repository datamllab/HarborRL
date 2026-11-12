"""Bind self-consistent group evidence to the sampling actually requested."""
from copy import deepcopy

import pytest
from test_rollout_service import FakeRuntime, group_request, service_ir

from harborrl.backends.slime_v032.rollout_service_client import (
    ServiceRolloutError,
    load_group_result,
)
from harborrl.rollout.service import build_group_result
from harborrl.trajectories.native import (
    Identity,
    assemble,
    canonical_sampling,
    digest,
    publish,
    require_ready,
)


def result_for(root, request, *, actual_sampling=None):
    sampling = canonical_sampling(actual_sampling or request["sampling"])
    runtime, trajectories = FakeRuntime(root), []
    for slot in request["slots"]:
        source = deepcopy(service_ir(root / "receipts", slot["slot_id"]))
        identity = Identity(**{**source["identity"], "sampling_digest": digest(sampling)})
        evaluation, seal, turns = source["evaluation"], source["trace_seal"], source["turns"]
        evaluation["identity"] = seal["identity"] = identity.to_dict()
        for turn in turns:
            turn["identity"] = identity.to_dict()
            turn["sampling"] = sampling
        seal["turns_digest"] = digest(turns)
        ir = assemble(identity, source["harbor_trial_id"], turns, seal, evaluation)
        require_ready(ir, allow_synthetic=True)  # internally valid, even for a different distribution
        publish(runtime.manifest_path(ir["identity"]), ir)
        trajectories.append(ir)
    return build_group_result("run-1", request, trajectories, runtime=runtime)


@pytest.mark.parametrize("key,value", [
    ("temperature", .5), ("top_p", .8), ("top_k", 10),
    ("max_new_tokens", 99), ("stop", ["END"]),
])
def test_rejects_internally_consistent_but_unrequested_sampling(tmp_path, key, value):
    request = group_request()
    result = result_for(tmp_path, request, actual_sampling={**request["sampling"], key: value})
    with pytest.raises(ServiceRolloutError, match="sampling"):
        load_group_result("run-1", request, result, allow_synthetic=True)


@pytest.mark.parametrize("probability", [1, 1.0])
def test_accepts_semantically_equivalent_probability_numbers(tmp_path, probability):
    request = group_request()
    request["sampling"].update(temperature=probability, top_p=probability)
    result = result_for(tmp_path, request)
    assert len(load_group_result("run-1", request, result, allow_synthetic=True)) == 2
    assert type(request["sampling"]["temperature"]) is type(probability)


@pytest.mark.parametrize("missing", [False, True])
def test_descriptor_sampling_cannot_be_changed_or_omitted(tmp_path, missing):
    request = group_request()
    result = result_for(tmp_path, request)
    if missing:
        del result["slots"][0]["sampling_digest"]
    else:
        result["slots"][0]["sampling_digest"] = "other-distribution"
    with pytest.raises(ServiceRolloutError, match="sampling digest"):
        load_group_result("run-1", request, result, allow_synthetic=True)


def test_requested_descriptor_does_not_mask_different_manifest_sampling(tmp_path):
    request = group_request()
    result = result_for(tmp_path, request, actual_sampling={**request["sampling"], "temperature": .5})
    for slot in result["slots"]:
        slot["sampling_digest"] = digest(canonical_sampling(request["sampling"]))
    with pytest.raises(ServiceRolloutError, match="manifest sampling"):
        load_group_result("run-1", request, result, allow_synthetic=True)
