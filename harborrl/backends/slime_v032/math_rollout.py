"""Single-turn math evidence adapter for the shared Slime training loop."""
from __future__ import annotations

import asyncio
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from urllib import request

from harborrl.backends.slime_v032.converter import STANDARD_FIELDS
from harborrl.export.contract import (
    DAPO_REDUCTION,
    OPD_REDUCTION,
    REDUCTION,
    SCHEMA_VERSION,
    ProcessRewards,
    TokenSpan,
    TrainingBatch,
    TrainingGroup,
    TrajectoryTrainingUnit,
)
from harborrl.export.validate import validate_batch
from harborrl.math import dapo
from harborrl.math import prm as prm_api
from harborrl.math import teacher as teacher_api
from harborrl.math.data import verify
from harborrl.trajectories.native import digest, finite, publish


def _request(url, payload=None):
    opener = request.build_opener(request.ProxyHandler({}))
    req = request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                          headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=600) as response:
        return json.load(response)


def _pool(endpoint):
    from harborrl.backends.slime_v032.rollout_service_client import fetch_pool_states
    states = fetch_pool_states(endpoint)
    versions = {state['weight_version'] for state in states}
    if len(versions) != 1 or any(not state['healthy'] for state in states):
        raise ValueError('math rollout requires one healthy synchronized policy pool')
    return states, next(iter(versions))


def build_batch(args, samples):
    """Rebuild neutral groups from exact serving evidence, never Harbor IR."""
    from collections import defaultdict
    algorithm = os.environ.get('HARBORRL_MATH_ALGORITHM', 'grpo')
    if algorithm not in ('grpo', 'opd', 'dapo'):
        raise ValueError('Unknown math algorithm')
    teacher = teacher_api.load(os.environ['HARBORRL_OPD_TEACHER']) if algorithm == 'opd' else None
    process = prm_api.load(os.environ['HARBORRL_PROCESS_REWARD']) if os.environ.get('HARBORRL_PROCESS_REWARD') else None
    if process is not None:
        from slime.utils.processing_utils import load_tokenizer
        if algorithm != 'grpo':
            raise ValueError('PRM currently requires GRPO')
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    settings = dapo.environment(args.rollout_max_response_len) if algorithm == 'dapo' else None
    groups = defaultdict(list)
    for sample in samples:
        evidence = sample.metadata['math_evidence']
        if evidence.get('algorithm', 'grpo') != algorithm:
            raise ValueError('Math sample algorithm changed')
        if (sample.tokens != evidence['tokens'] or sample.rollout_log_probs != evidence['logprobs']
                or sample.response != evidence['response']
                or sample.response_length != evidence['response_length']
                or sample.loss_mask != [1] * evidence['response_length']):
            raise ValueError('math sample differs from serving evidence')
        task = sample.metadata['math_task']
        if (task['split'] != 'train' or digest(task) != sample.metadata['task_digest']
                or evidence['task_digest'] != sample.metadata['task_digest']):
            raise ValueError('math task identity or train split changed')
        result = verify(sample.response, task['reference'], task['verifier'])
        if result != evidence['verification']:
            raise ValueError('math verifier evidence changed')
        if settings is not None:
            expected = dapo.reward(result['reward'], sample.response_length,
                                   args.rollout_max_response_len, settings)
            if evidence.get('dapo') != expected or sample.rollout_id != evidence['rollout_id']:
                raise ValueError('DAPO reward or rollout identity changed')
        elif 'dapo' in evidence:
            raise ValueError('Only DAPO may consume shaped rewards')
        if teacher is not None:
            scores = evidence.get('teacher', {})
            if (scores.get('identity') != teacher['identity']
                    or scores.get('weight_version') != teacher['weight_version']
                    or scores.get('tokens_digest') != digest(sample.tokens)
                    or scores.get('logprobs') is None
                    or getattr(sample, 'teacher_log_probs', None) != scores['logprobs']):
                raise ValueError('OPD teacher identity or exact-token evidence changed')
        elif 'teacher' in evidence or getattr(sample, 'teacher_log_probs', None) is not None:
            raise ValueError('Only OPD may consume teacher evidence')
        if process is not None:
            if tokenizer.decode(sample.tokens[-sample.response_length:], skip_special_tokens=True) != sample.response:
                raise ValueError('PRM response text differs from exact student tokens')
            prm_api.validate_receipt(process, evidence.get('process_reward', {}), task['prompt'],
                                     sample.tokens[-sample.response_length:], tokenizer)
        elif 'process_reward' in evidence:
            raise ValueError('Process rewards require a frozen PRM manifest')
        groups[evidence['group_key']].append((sample, evidence, result))
    output, versions = [], set()
    for key, rows in groups.items():
        if len(rows) != args.n_samples_per_prompt:
            raise ValueError('incomplete math group')
        if len({row[1]['task_digest'] for row in rows}) != 1:
            raise ValueError('math group mixes tasks')
        if settings is not None and len({row[2]['reward'] for row in rows}) == 1:
            raise ValueError('DAPO must filter zero-variance outcome groups')
        rewards = [row[1]['dapo']['reward'] if settings is not None else row[2]['reward'] for row in rows]
        if process is not None:
            rewards = [row[1]['process_reward']['token_returns'][0] for row in rows]
        mean = math.fsum(rewards) / len(rows)
        std = math.sqrt(math.fsum((reward - mean) ** 2 for reward in rewards) / len(rows))
        units = []
        for (sample, evidence, result), reward in zip(rows, rewards, strict=True):
            response_length = evidence['response_length']
            prompt_length = len(sample.tokens) - response_length
            version = evidence['policy_version']
            versions.add(version)
            span = TokenSpan(0, sample.tokens, prompt_length, response_length,
                             [False] * prompt_length + [True] * response_length,
                             sample.rollout_log_probs, version,
                             sample.teacher_log_probs if teacher is not None else None,
                             ProcessRewards(process['identity'], evidence['process_reward']['step_ends'],
                                            evidence['process_reward']['probabilities'],
                                            evidence['process_reward']['token_returns']) if process else None)
            advantage = 0.0 if teacher is not None else reward - mean
            if settings is not None:
                advantage = dapo.advantage(reward, mean, std, len(rows))
            units.append(TrajectoryTrainingUnit(evidence['trajectory_id'], evidence['rollout_id'], key,
                         reward, advantage,
                         response_length, (span,)))
        output.append(TrainingGroup(key, len(rows), mean, std, tuple(units)))
    reduction = {'grpo': REDUCTION, 'opd': OPD_REDUCTION, 'dapo': DAPO_REDUCTION}[algorithm]
    batch = TrainingBatch(SCHEMA_VERSION, tuple(output), reduction,
                          frozenset(versions), algorithm, teacher['identity'] if teacher else None,
                          process['identity'] if process else None)
    validate_batch(batch)
    return batch


async def _generate(args, rollout_id, groups):
    from slime.utils.processing_utils import load_tokenizer
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    endpoint = f'http://{args.sglang_router_ip}:{args.sglang_router_port}'
    before, version = await asyncio.to_thread(_pool, endpoint)
    algorithm = os.environ.get('HARBORRL_MATH_ALGORITHM', 'grpo')
    settings = dapo.environment(args.rollout_max_response_len) if algorithm == 'dapo' else None
    teacher = teacher_api.load(os.environ['HARBORRL_OPD_TEACHER']) if algorithm == 'opd' else None
    process = prm_api.load(os.environ['HARBORRL_PROCESS_REWARD']) if os.environ.get('HARBORRL_PROCESS_REWARD') else None
    if process is not None:
        if algorithm != 'grpo' or process['endpoint'].rstrip('/') == endpoint:
            raise ValueError('PRM requires GRPO and a separate frozen scorer endpoint')
        process_before = await asyncio.to_thread(prm_api.check, process)
    if teacher is not None:
        if teacher['endpoint'].rstrip('/') == endpoint:
            raise ValueError('OPD teacher must use a separate frozen serving endpoint')
        teacher_before = await asyncio.to_thread(teacher_api.check, teacher)
    semaphore = asyncio.Semaphore(max(1, args.sglang_server_concurrency))

    async def one(sample, group_key, group_rollout_id):
        async with semaphore:
            task = sample.metadata['math_task']
            if task['split'] != 'train' or digest(task) != sample.metadata['task_digest']:
                raise ValueError('math generation requires an unchanged train task')
            prompt = tokenizer.encode(sample.prompt, add_special_tokens=False)
            if len(prompt) + args.rollout_max_response_len > args.rollout_max_context_len:
                raise ValueError('math prompt exceeds the context budget')
            payload = {'input_ids': prompt, 'return_logprob': True, 'stream': False,
                       'sampling_params': {'temperature': 1.0, 'top_p': 1.0, 'top_k': -1,
                                           'max_new_tokens': args.rollout_max_response_len,
                                           'skip_special_tokens': True, 'no_stop_trim': True}}
            if os.environ.get('HARBORRL_MATH_SEED'):
                payload['sampling_params']['sampling_seed'] = (args.rollout_seed + sample.index) % 2**31
            raw = await asyncio.to_thread(_request, endpoint + '/generate', payload)
            meta = raw['meta_info']
            if str(meta.get('weight_version', '')) != version:
                raise ValueError('math generation served a stale or missing policy version')
            rows = meta.get('output_token_logprobs')
            if not rows or any(len(row) < 2 or not finite(row[0]) or row[0] > 0
                               or type(row[1]) is not int or row[1] < 0 for row in rows):
                raise ValueError('math generation requires exact token logprobs')
            response_ids = [row[1] for row in rows]
            text = tokenizer.decode(response_ids, skip_special_tokens=True)
            if text != raw['text']:
                raise ValueError('math serving text differs from output tokens')
            finish = meta['finish_reason']['type']
            if finish not in ('stop', 'length'):
                raise ValueError('unsupported math finish reason')
            sample.tokens = prompt + response_ids
            sample.response = text
            sample.response_length = len(response_ids)
            sample.rollout_log_probs = [row[0] for row in rows]
            sample.loss_mask = [1] * len(response_ids)
            sample.rollout_id = group_rollout_id
            sample.status = sample.Status.TRUNCATED if finish == 'length' else sample.Status.COMPLETED
            sample.weight_versions = [version]
            result = verify(text, task['reference'], task['verifier'])
            sample.reward = {'score': result['reward']}
            evidence = {'tokens': sample.tokens, 'response': text, 'response_length': len(response_ids),
                        'logprobs': sample.rollout_log_probs, 'policy_version': version,
                        'group_key': group_key, 'trajectory_id': f'{group_key}:{sample.index}',
                        'rollout_id': group_rollout_id, 'task_digest': digest(task),
                        'verification': result, 'truncated': finish == 'length',
                        'serving_input': payload, 'serving_output_digest': digest(raw)}
            sample.metadata['math_evidence'] = evidence
            evidence['algorithm'] = algorithm
            if settings is not None:
                evidence['dapo'] = dapo.reward(result['reward'], sample.response_length,
                                             args.rollout_max_response_len, settings)
                sample.reward = {'score': evidence['dapo']['reward']}
            if teacher is not None:
                scores = await asyncio.to_thread(teacher_api.score_response, teacher, prompt, response_ids)
                sample.teacher_log_probs = scores['logprobs']
                evidence['teacher'] = scores
            if process is not None:
                scores = await asyncio.to_thread(prm_api.score_response, process, task['prompt'], response_ids, tokenizer)
                evidence['process_reward'] = scores
                sample.reward = {'score': scores['token_returns'][0]}
            return sample

    pending = []
    for group_index, group in enumerate(groups):
        if len({sample.metadata['task_digest'] for sample in group}) != 1:
            raise ValueError('math prompt group mixes tasks')
        if settings is not None:
            source_ids = {sample.group_index for sample in group}
            if len(source_ids) != 1 or type(group[0].group_index) is not int or group[0].group_index < 0:
                raise ValueError('DAPO requires persistent data-source group indices')
            group_index = group[0].group_index
        group_key = digest({'rollout': rollout_id, 'group': group_index,
                            'task': group[0].metadata['task_digest'], 'policy': version})
        group_rollout_id = group_index if settings is not None else rollout_id * args.rollout_batch_size + group_index
        pending.extend(one(sample, group_key, group_rollout_id) for sample in group)
    samples = await asyncio.gather(*pending)
    after, after_version = await asyncio.to_thread(_pool, endpoint)
    if before != after or after_version != version:
        raise ValueError('policy pool changed during synchronous math rollout')
    if teacher is not None and await asyncio.to_thread(teacher_api.check, teacher) != teacher_before:
        raise ValueError('OPD teacher changed during scoring')
    if process is not None and await asyncio.to_thread(prm_api.check, process) != process_before:
        raise ValueError('PRM changed during scoring')
    return samples


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    from slime.rollout.base_types import RolloutFnTrainOutput
    if evaluation:
        raise ValueError('math evaluation must use its independent locked split')
    if (args.rollout_temperature, args.rollout_top_p, args.rollout_top_k) != (1.0, 1.0, -1):
        raise ValueError('math training requires unmodified raw-model sampling')
    root = Path(os.environ['RUN_DIR']) / 'math' / f'{rollout_id:06d}'
    algorithm = os.environ.get('HARBORRL_MATH_ALGORITHM', 'grpo')
    settings = dapo.environment(args.rollout_max_response_len) if algorithm == 'dapo' else None
    samples, versions = [], set()
    selection = {'status': 'sampling', 'accepted': [], 'rejected': [], 'generation_batches': 0}
    limit = settings['max_refill_batches'] if settings is not None else 1
    for attempt in range(limit):
        remaining = args.rollout_batch_size - len(samples) // args.n_samples_per_prompt
        groups = [list(group) for group in data_source.get_samples(remaining)]
        if len(groups) != remaining or any(len(group) != args.n_samples_per_prompt for group in groups):
            raise ValueError('incomplete math rollout batch')
        generated = asyncio.run(_generate(args, rollout_id, groups))
        if len(generated) != remaining * args.n_samples_per_prompt:
            raise ValueError('incomplete math generation batch')
        versions.update(sample.metadata['math_evidence']['policy_version'] for sample in generated)
        if len(versions) != 1:
            raise ValueError('policy changed during math refill sampling')
        if settings is None:
            samples = generated
            break
        publish(root / f'sampling-evidence-{attempt:03d}.json', [sample.metadata for sample in generated])
        for start in range(0, len(generated), args.n_samples_per_prompt):
            group = generated[start:start + args.n_samples_per_prompt]
            keys = {sample.metadata['math_evidence']['group_key'] for sample in group}
            if len(keys) != 1:
                raise ValueError('DAPO generation mixed prompt groups')
            key = next(iter(keys))
            if key in selection['accepted'] or key in selection['rejected']:
                raise ValueError('DAPO generated a duplicate prompt group')
            outcomes = [sample.metadata['math_evidence']['verification']['reward'] for sample in group]
            if len(set(outcomes)) == 1:
                selection['rejected'].append(key)
            else:
                selection['accepted'].append(key)
                samples.extend(group)
        selection['generation_batches'] = attempt + 1
        complete = len(samples) == args.rollout_batch_size * args.n_samples_per_prompt
        selection['status'] = 'complete' if complete else 'sampling'
        publish(root / f'sampling-selection-{attempt:03d}.json', selection)
        if complete:
            publish(root / 'sampling.json', selection)
            break
    else:
        selection['status'] = 'exhausted'
        publish(root / 'sampling.json', selection)
        raise ValueError(f'DAPO refill exhausted after {limit} batches: '
                         f'{len(selection["accepted"])} accepted groups; see {root}')
    batch = build_batch(args, samples)
    publish(root / 'evidence.json', [sample.metadata for sample in samples])
    publish(root / 'batch.json', {**asdict(batch), 'policy_versions': sorted(batch.policy_versions)})
    statuses = [sample.metadata['math_evidence']['verification']['status'] for sample in samples]
    metrics = {f'math_{status}': float(statuses.count(status))
               for status in ('correct', 'incorrect', 'unparseable', 'unsupported')}
    metrics.update(math_group_count=float(len(batch.groups)),
                   math_nonzero_group_count=float(sum(group.reward_std > 0 for group in batch.groups)),
                   math_response_tokens=float(sum(sample.response_length for sample in samples)))
    if batch.process_reward_identity is not None:
        scores = [sample.metadata['math_evidence']['process_reward'] for sample in samples]
        probabilities = [value for receipt in scores for value in receipt['probabilities']]
        metrics.update(prm_step_count=float(len(probabilities)),
                       prm_probability_mean=math.fsum(probabilities) / len(probabilities),
                       prm_total_reward_mean=math.fsum(row['token_returns'][0] for row in scores) / len(scores),
                       prm_truncated_responses=float(sum(sample.metadata['math_evidence']['truncated']
                                                         for sample in samples)))
    if settings is not None:
        penalties = [sample.metadata['math_evidence']['dapo']['overlong_penalty'] for sample in samples]
        metrics.update(dapo_generated_groups=float(len(selection['accepted']) + len(selection['rejected'])),
                       dapo_rejected_groups=float(len(selection['rejected'])),
                       dapo_generation_batches=float(selection['generation_batches']),
                       dapo_overlong_responses=float(sum(penalty < 0 for penalty in penalties)),
                       dapo_overlong_penalty_sum=math.fsum(penalties),
                       dapo_truncated_responses=float(sum(sample.metadata['math_evidence']['truncated']
                                                         for sample in samples)))
    return RolloutFnTrainOutput(samples=samples, metrics=metrics)


def convert_samples_to_train_data(args, samples):
    samples = [item for sample in samples for item in (sample if isinstance(sample, (list, tuple)) else [sample])]
    batch = build_batch(args, samples)
    units = {unit.trajectory_id: unit for group in batch.groups for unit in group.trajectories}
    groups = {group.group_key: group for group in batch.groups}
    data = {key: [] for key in STANDARD_FIELDS}
    if batch.algorithm == 'opd':
        data['teacher_log_probs'] = []
    for sample in samples:
        evidence = sample.metadata['math_evidence']
        unit = units[evidence['trajectory_id']]
        values = (sample.tokens, sample.response_length, unit.advantage, evidence['verification']['reward'],
                  int(evidence['truncated']), sample.index, unit.rollout_id,
                  sample.loss_mask, sample.rollout_log_probs, unit.token_count)
        if batch.process_reward_identity is not None:
            from harborrl.backends.slime_v032.prm import reward_payload
            values = (*values[:2], reward_payload(unit, groups[unit.group_key], sample.index), *values[3:])
        for key, value in zip(STANDARD_FIELDS, values, strict=True):
            data[key].append(value)
        if batch.algorithm == 'opd':
            data['teacher_log_probs'].append(list(unit.turns[0].teacher_logprobs))
    return data
