"""Worker-published IR acceptance and trust-boundary regression checks."""
import json

import pytest

from harborrl.backends.slime_v032.hub_rollout_client import HubRolloutError, load_attempt_ir
from harborrl.trajectories.native import digest, publish


@pytest.mark.parametrize('mirror', [None, '', 'relative/attempts'])
def test_worker_requires_shared_mirror_before_starting_attempt(monkeypatch, tmp_path, mirror):
    from harborrl.deployment.hub_rollout_executor import RolloutExecutor

    if mirror is None:
        monkeypatch.delenv('HARBORRL_HUB_ATTEMPT_MIRROR', raising=False)
    else:
        monkeypatch.setenv('HARBORRL_HUB_ATTEMPT_MIRROR', mirror)
    run_root = tmp_path / 'attempts'
    executor = RolloutExecutor(
        run_root=run_root, task_path=tmp_path / 'task', profile={},
        runner_python='python', source_dir=tmp_path, gateway_host='127.0.0.1')
    with pytest.raises(ValueError, match='absolute shared directory'):
        executor.execute({})
    assert not run_root.exists()

def test_worker_publishes_ir_to_configured_shared_storage(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from harborrl.deployment import hub_rollout_executor
    from harborrl.rollout.harbor_job import coordinator

    mirror = tmp_path / 'shared'
    monkeypatch.setenv('HARBORRL_HUB_ATTEMPT_MIRROR', str(mirror))
    closed = []
    server = SimpleNamespace(
        server_address=('127.0.0.1', 12345), serve_forever=lambda: None,
        shutdown=lambda: closed.append('shutdown'),
        server_close=lambda: closed.append('close'))
    monkeypatch.setattr(hub_rollout_executor, 'make_server', lambda *a, **k: server)

    async def attempt(identity, *, root, **kwargs):
        root.mkdir(parents=True, exist_ok=True)
        ir = {'identity': {'policy_version': identity.policy_version},
              'readiness': 'RL_READY', 'integrity_errors': [],
              'evaluation': {'training_reward': 1.0}}
        publish(root / 'trajectory.v2.json', ir)
        publish(root / 'evaluation.json', ir['evaluation'])
        return ir

    monkeypatch.setattr(coordinator, 'run_attempt', attempt)
    executor = hub_rollout_executor.RolloutExecutor(
        run_root=tmp_path / 'worker', task_path=tmp_path / 'task',
        profile={'model': 'policy'}, runner_python='python', source_dir='.',
        gateway_host='127.0.0.1', log=lambda _: None)
    result = executor.execute({'task_id': 'task-1', 'task_digest': 'task-sha', 'synthetic': True})
    expected = mirror / 'task-1-0' / 'trajectory.v2.json'
    assert result['ir_path'] == str(expected)
    assert load_attempt_ir('task-1', result, expected_policy_version='1') == json.loads(
        expected.read_text(encoding='utf-8'))
    assert (expected.parent / 'evaluation.json').is_file()
    assert closed == ['shutdown', 'close']


def test_hub_requires_explicit_configuration(monkeypatch, tmp_path):
    from harborrl.backends.slime_v032 import rollout, hub_rollout_client

    monkeypatch.delenv('HARBORRL_HUB_URL', raising=False)
    monkeypatch.delenv('HARBORRL_HUB_ENV_FILE', raising=False)
    reads = []
    monkeypatch.setattr(rollout.Path, 'read_text', lambda *a, **k: reads.append(a) or '{}')
    assert rollout.hub_rollout_enabled() is False
    with pytest.raises(HubRolloutError, match='is not set'):
        hub_rollout_client.client_from_environment()
    assert reads == []


def test_explicit_hub_file_is_passed_to_training_and_ray(monkeypatch, tmp_path):
    from harborrl.deployment.training_target import local_target
    from harborrl.platform.hub_train import rollout_environment

    filename = str(tmp_path / 'hub.json')
    monkeypatch.setenv('HARBORRL_HUB_ENV_FILE', filename)
    monkeypatch.setenv('UNRELATED_SECRET', 'must-not-propagate')
    target = local_target({'training_command': ['true']}, environment={'RUN_ID': 'run-1'})
    forwarded = rollout_environment(target.env)
    assert forwarded['HARBORRL_HUB_ENV_FILE'] == filename
    assert forwarded['RUN_ID'] == 'run-1'
    assert 'UNRELATED_SECRET' not in forwarded


@pytest.fixture
def published_ir(tmp_path):
    ir = {"identity": {"policy_version": "3", "task_digest": "task-sha"},
          "readiness": "RL_READY", "integrity_errors": [],
          "evaluation": {"training_reward": 1.0}}
    path = tmp_path / "trajectory.v2.json"
    publish(path, ir)
    return ir, path, {"ir_path": str(path), "ir_sha256": digest(ir)}


def test_accepts_worker_publication_with_trailing_newline(published_ir):
    ir, path, report = published_ir
    assert path.read_bytes().endswith(b"\n")
    assert load_attempt_ir("task", report, expected_policy_version="3",
                           expected_task_digest="task-sha") == ir


def test_structural_digest_is_independent_of_json_whitespace(published_ir):
    ir, path, report = published_ir
    path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    assert load_attempt_ir("task", report) == ir


def test_accepts_worker_report_envelope(published_ir):
    ir, _, report = published_ir
    envelope = {"success": True, "result": {"success": True, **report}}
    assert load_attempt_ir("task", envelope, expected_policy_version="3") == ir


@pytest.mark.parametrize("envelope", [
    {"success": False, "result": {}},
    {"success": True, "result": None},
    {"success": True, "result": {"success": False}},
])
def test_rejects_failed_worker_envelope(envelope):
    with pytest.raises(HubRolloutError, match="unsuccessful|failed attempt"):
        load_attempt_ir("task", envelope)


def test_enveloped_synthetic_ir_is_still_rejected(published_ir):
    ir, path, report = published_ir
    ir["readiness"] = "CONTRACT_READY"
    path.write_text(json.dumps(ir), encoding="utf-8")
    report["ir_sha256"] = digest(ir)
    with pytest.raises(HubRolloutError, match="not RL_READY"):
        load_attempt_ir("task", {"success": True, "result": report})


def test_hub_refreshes_policy_lock_after_weight_update(monkeypatch, tmp_path):
    import asyncio
    from types import SimpleNamespace
    from harborrl.backends.slime_v032 import rollout, rollout_service_client
    from harborrl.backends.slime_v032 import hub_rollout_client

    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    monkeypatch.setenv("HARBORRL_HUB_ENV_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("HARBORRL_HUB_SERVING_ENDPOINT", "http://worker:30001")
    monkeypatch.setenv("HARBORRL_HUB_HF_CKPT", "checkpoint")
    monkeypatch.setenv("HARBORRL_CONSISTENCY_MODE", "off")
    monkeypatch.setattr(hub_rollout_client, "client_from_environment", lambda: object())
    states = [dict(endpoint="http://engine:15000", healthy=True, weight_version="1")]
    monkeypatch.setattr(rollout_service_client, "fetch_pool_states", lambda _: states)
    args = SimpleNamespace(sglang_router_ip="router", sglang_router_port=8000,
                           rollout_max_response_len=2048)
    for step in (0, 1):
        states[0]["weight_version"] = str(step + 1)
        assert asyncio.run(rollout.run_hub_groups(args, [], step)) == []
        assert json.loads((tmp_path / "policy-pool.json").read_text())["weight_version"] == str(step + 1)
    history = (tmp_path / "policy-pool-history.jsonl").read_text().splitlines()
    assert [json.loads(line)["weight_version"] for line in history] == ["1", "2"]
    states[0]["healthy"] = False
    with pytest.raises(RuntimeError, match="unhealthy"):
        asyncio.run(rollout.run_hub_groups(args, [], 2))


@pytest.mark.parametrize("audited", [False, True])
def test_hub_request_preserves_explicit_logprob_audit(audited):
    from harborrl.backends.slime_v032.hub_rollout_client import hub_attempt_request
    group = dict(task=dict(path="/task", task_digest="sha"), rollout_id=0,
                 group_id="group", policy_version="1")
    result = hub_attempt_request(group, dict(slot_id="0"), serving_endpoint="http://engine",
        hf_ckpt="model", tokenizer_digest="tok", template_digest="tpl", model_tag="qwen3-8B",
        max_output_tokens=2048, max_context=32768, task_path="/task",
        audited_raw_logprobs=audited)
    assert result["audited_raw_logprobs"] is audited


@pytest.mark.parametrize('generation', [0, 1])
def test_hub_dispatch_identity_satisfies_generation_journal(monkeypatch, generation):
    from types import SimpleNamespace
    from harborrl.backends.slime_v032 import rollout_service_client
    from harborrl.backends.slime_v032.hub_rollout_client import hub_attempt_request
    from harborrl.rollout.generations import ACTIVE_GENERATION, GenerationJournal

    task = dict(path='/task', task_digest='sha')
    monkeypatch.setattr(rollout_service_client, 'task_entry', lambda _: task)
    token = ACTIVE_GENERATION.set({'generation': generation})
    try:
        group = rollout_service_client.service_group_request(
            [SimpleNamespace(group_index=0), SimpleNamespace(group_index=0)], {}, 2, '3')
    finally:
        ACTIVE_GENERATION.reset(token)
    kwargs = dict(serving_endpoint='http://engine', hf_ckpt='model',
                  tokenizer_digest='tok', template_digest='tpl', model_tag='qwen3-8B',
                  max_output_tokens=512, max_context=32768, task_path='/task')
    identities = [hub_attempt_request(group, slot, **kwargs) for slot in group['slots']]
    inputs = {'groups': [{'group_index': 0, 'slots': ['0', '1'], 'tasks': [task, task]}]}
    assert GenerationJournal.validate_result(
        [[{'identity': identity} for identity in identities]], inputs, 2, generation, '3')
    with pytest.raises(ValueError, match='conflicts with rollout generation'):
        hub_attempt_request(group, group['slots'][0], batch_id='batch', **kwargs)


@pytest.mark.parametrize('sandbox', [None, {'adapter': 'example:Sandbox', 'credential_file': '/run/key'}])
def test_hub_dispatch_preserves_sandbox_runtime(monkeypatch, tmp_path, sandbox):
    import asyncio
    from types import SimpleNamespace
    from harborrl.backends.slime_v032 import rollout, rollout_service_client, hub_rollout_client
    from harborrl.platform import policy_pool

    class Submitted(Exception):
        pass

    captured = []
    class Client:
        def submit_attempt(self, request):
            captured.append(request)
            raise Submitted()

    monkeypatch.setenv('RUN_DIR', str(tmp_path))
    monkeypatch.setenv('HARBORRL_HUB_ENV_FILE', str(tmp_path / 'missing.json'))
    monkeypatch.setenv('HARBORRL_HUB_SERVING_ENDPOINT', 'http://engine:15000')
    monkeypatch.setenv('HARBORRL_HUB_HF_CKPT', 'model')
    monkeypatch.setenv('HARBORRL_NATIVE_SANDBOX', json.dumps(sandbox))
    monkeypatch.setattr(hub_rollout_client, 'client_from_environment', Client)
    monkeypatch.setattr(rollout_service_client, 'fetch_pool_states', lambda _: [])
    monkeypatch.setattr(policy_pool, 'publish_pool', lambda _: None)
    monkeypatch.setattr(rollout, 'lock_policy_version', lambda _: '1')
    group = dict(task=dict(path='/task', task_digest='sha'), rollout_id=0,
                 group_id='group', policy_version='1', slots=[dict(slot_id='0')])
    monkeypatch.setattr(rollout_service_client, 'service_group_request', lambda *a: group)
    args = SimpleNamespace(sglang_router_ip='router', sglang_router_port=8000,
                           rollout_max_response_len=512)
    with pytest.raises(Submitted):
        asyncio.run(rollout.run_hub_groups(args, [object()], 0))
    assert captured[0].get('sandbox') == sandbox
    assert ('sandbox' in captured[0]) is (sandbox is not None)


def test_rejects_changed_reward(published_ir):
    ir, path, report = published_ir
    ir["evaluation"]["training_reward"] = 0.0
    path.write_text(json.dumps(ir), encoding="utf-8")
    with pytest.raises(HubRolloutError, match="digest mismatch"):
        load_attempt_ir("task", report)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', 'broken'])
def test_rejects_invalid_or_ambiguous_json(published_ir, raw):
    _, path, report = published_ir
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(HubRolloutError, match="invalid IR"):
        load_attempt_ir("task", report)


@pytest.mark.parametrize("field,value,error", [
    ("readiness", "CONTRACT_READY", "not RL_READY"),
    ("integrity_errors", ["bad seal"], "integrity errors"),
])
def test_keeps_training_readiness_gate(published_ir, field, value, error):
    ir, path, report = published_ir
    ir[field] = value
    path.write_text(json.dumps(ir), encoding="utf-8")
    report["ir_sha256"] = digest(ir)
    with pytest.raises(HubRolloutError, match=error):
        load_attempt_ir("task", report)


@pytest.mark.parametrize("expected,error", [
    ({"expected_policy_version": "4"}, "policy version"),
    ({"expected_task_digest": "other"}, "task digest"),
])
def test_keeps_identity_locks(published_ir, expected, error):
    with pytest.raises(HubRolloutError, match=error):
        load_attempt_ir("task", published_ir[2], **expected)
