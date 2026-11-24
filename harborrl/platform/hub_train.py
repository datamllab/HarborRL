"""Run upstream Slime with explicitly configured Hub environment on Ray workers."""
import os
import runpy
import sys
from pathlib import Path


def rollout_environment(environ):
    return {key: value for key, value in environ.items()
            if key.startswith('HARBORRL_') or key in {
                'RUN_DIR', 'RUN_ID', 'MODEL_TAG', 'MODEL_ARGS_FILE',
                'ROLLOUT_MAX_RESPONSE_LEN', 'ROLLOUT_MAX_CONTEXT_LEN', 'PYTHONPATH'}}


def main():
    import ray

    if not ray.is_initialized():
        ray.init(runtime_env={'env_vars': rollout_environment(os.environ)})
    entry = Path(os.environ['SLIME_DIR']) / 'train.py'
    sys.argv[0] = str(entry)
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
