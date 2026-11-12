"""Explicit DAPO math settings and complete-group reward semantics."""
import json
import math
import os

from harborrl.trajectories.native import finite

FIELDS = {'clip_low', 'clip_high', 'overlong_buffer_tokens', 'overlong_penalty', 'max_refill_batches'}
ADVANTAGE_EPSILON = 1e-6


def validate(config, max_tokens):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError(f'dapo requires exactly {sorted(FIELDS)}')
    low, high = config['clip_low'], config['clip_high']
    if not finite(low) or not finite(high) or not 0 < low < 1 or high <= low:
        raise ValueError('DAPO requires 0 < clip_low < 1 and clip_high > clip_low')
    buffer = config['overlong_buffer_tokens']
    if type(buffer) is not int or not 0 < buffer < max_tokens:
        raise ValueError('DAPO overlong_buffer_tokens must be between zero and max_tokens')
    penalty = config['overlong_penalty']
    if not finite(penalty) or not 0 < penalty <= 1:
        raise ValueError('DAPO overlong_penalty must be in (0, 1]')
    if type(config['max_refill_batches']) is not int or config['max_refill_batches'] < 1:
        raise ValueError('DAPO max_refill_batches must be a positive integer')
    return config


def environment(max_tokens):
    return validate(json.loads(os.environ['HARBORRL_DAPO']), max_tokens)


def reward(outcome_reward, length, max_tokens, config):
    if outcome_reward not in (-1., 0., 1.) or type(length) is not int or not 0 < length <= max_tokens:
        raise ValueError('DAPO requires a verifier outcome of -1, 0 or 1 and a response within max_tokens')
    buffer = config['overlong_buffer_tokens']
    penalty = -max(0, length - (max_tokens - buffer)) / buffer * config['overlong_penalty']
    return {'reward': outcome_reward + penalty, 'overlong_penalty': penalty}


def advantage(reward, mean, population_std, group_size):
    """Use the sample std convention in the published verl DAPO recipe."""
    sample_std = population_std * math.sqrt(group_size / (group_size - 1))
    return (reward - mean) / (sample_std + ADVANTAGE_EPSILON)
