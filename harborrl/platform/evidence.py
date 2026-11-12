"""Run input preparation and reproducible source evidence."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from harborrl.config import ROOT
from harborrl.trajectories.native import publish


def new_run_root(output_root):
    name = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    return Path(output_root) / "training" / name


def run_environment(plan, root, catalog_path, prompt_path):
    """Shared run paths for both local and controller-managed training."""
    return {
        **plan["environment"],
        "RUN_ID": root.name, "RUN_NAME": root.name, "RUN_DIR": str(root),
        "RUNS_ROOT": str(root.parent.parent.resolve()),
        "CKPT_ROOT": str(root / "checkpoints"), "WANDB_DIR": str(root / "metrics/wandb"),
        "HARBORRL_NATIVE_CATALOG": str(catalog_path),
        "ROLLOUT_PROMPT_DATA": str(prompt_path),
        **({"HARBORRL_NATIVE_PROFILE": plan["config"]["harness"]["profile"]}
           if "harness" in plan["config"] else {}),
        "HARBOR_IR_ROOT": str(root / "ir"),
        **({"SAVE_HF": str(root / "hf" / "rollout-{rollout_id}")}
           if plan['config']['training'].get('export_hf') else {}),
        **({'HARBORRL_OPD_TEACHER': str(root / 'config' / 'teacher.json')}
           if plan['math']['algorithm']['id'] == 'opd' else {}),
        **({'HARBORRL_PROCESS_REWARD': str(root / 'config' / 'process_reward.json')
            if 'process_reward' in plan['config'] else ''}
           if plan['config']['execution']['backend'] == 'text_math' else {}),
    }


def _materialize(plan, root):
    if plan['config']['execution']['backend'] == 'text_math':
        from harborrl.math.config import materialize
        return materialize(plan, root)
    from harborrl.math_registry import select_math

    config = plan["config"]
    entries = select_math(config).task.adapter.catalog(config)
    normalized = root / "config"
    normalized.mkdir(parents=True)
    catalog_path = normalized / "native_catalog.json"
    publish(catalog_path, entries)
    prompt_path = normalized / "native_tasks.jsonl"
    rows = select_math(config).task.adapter.prompt_rows(entries)
    prompt_path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))
    return catalog_path, prompt_path


def _source_audit(root):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    files = git("ls-files", "-z", "--cached", "--others", "--exclude-standard", "--",
                "harborrl", "scripts", "examples", "pyproject.toml").decode().split("\0")
    untracked = set(git("ls-files", "-z", "--others", "--exclude-standard", "--",
                        "harborrl", "scripts", "examples").decode().split("\0"))
    (root / "source.patch").write_bytes(git("diff", "--binary", "HEAD", "--",
                                             "harborrl", "scripts", "examples", "pyproject.toml"))
    hashes = {}
    for name in files:
        source = ROOT / name
        if name and source.is_file():
            hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
            if name in untracked:
                target = root / "source-new" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
    payload = {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "source_hashes": hashes,
        "patch": "source.patch",
    }
    publish(root / "source-manifest.json", payload)
    return payload




# Proxy values and provider credentials are deliberately outside this snapshot.
DEPLOYMENT_EVIDENCE_KEYS = ("SLIME_DIR", "MEGATRON_DIR", "SGLANG_IMAGE", "CUDA_VISIBLE_DEVICES")


def effective_config(plan, environment, child_env, source, checks):
    from harborrl.backends.slime_v032.config import field_usage
    config = plan["config"]
    return {
        "schema_version": "harborrl-effective-config-v1",
        "components": {
            "model_preset": config["model"]["args_file"],
            "algorithm": plan["math"]["algorithm"]["id"],
            "harness": plan["math"]["harness"]["id"],
            "execution_backend": config["execution"]["backend"],
            "training_backend_contract": config["training"]["backend_contract"],
        },
        "environment": environment,
        "deployment_environment": {key: child_env[key] for key in DEPLOYMENT_EVIDENCE_KEYS if key in child_env},
        "training_command": plan["training_command"],
        "update_semantics": ({
            "objective": "sampled_token_fixed_prefix_reverse_kl_surrogate",
            "advantage": "detached_teacher_logprob_minus_detached_behavior_logprob",
            "reward_mixing": False,
            "clipping": False,
            "std_normalization": False,
            "reduction": "trajectory_weighted_sampled_opd_v1",
            "teacher_manifest": "config/teacher.json",
        } if plan['math']['algorithm']['id'] == 'opd' else {
            'objective': 'asymmetric_clipped_policy_gradient',
            'advantage': 'shaped_reward_minus_complete_group_mean_div_sample_std_plus_1e-6',
            'std_normalization': True,
            'reduction': 'global_token_mean_asymmetric_clipped_pg_v1',
            'dynamic_sampling': 'reject_zero_variance_outcome_groups_and_refill_complete_groups',
            'truncation': 'retain_all_response_tokens_with_soft_overlong_penalty',
            **config['dapo'],
        } if plan['math']['algorithm']['id'] == 'dapo' else {
            'objective': 'trajectory_weighted_clipped_process_policy_gradient',
            'reward_source': 'qwen-math-prm-v1',
            'step_reward': '2 * positive_probability - 1',
            'advantage': 'undiscounted_step_reward_to_go_minus_complete_group_mean_total_reward',
            'step_boundary': 'double_newline_at_student_token_boundary; boundary_token_belongs_to_previous_step',
            'std_normalization': False,
            'outcome_reward_mixing': False,
            'truncation': 'score_observed_tail; retain_all_response_tokens; no_unobserved_suffix_reward',
            'reduction': 'trajectory_weighted_clipped_pg_v1',
            'scorer_manifest': 'config/process_reward.json',
        } if 'process_reward' in config else {
            "advantage": "reward_minus_complete_group_mean",
            "std_normalization": False,
            "reduction": "trajectory_weighted_clipped_pg_v1",
        }),
        "expanded_command_file": "effective-command.json",
        "source": source,
        "runtime_versions": [check for check in checks
                             if "version" in check or check.get("service") in
                             ("slime commit", "megatron commit", "sglang image")],
        "field_usage": ({f'{section}.{key}': ['text_math configuration']
                         for section, values in config.items() if isinstance(values, dict)
                         for key in values}
                        if config['execution']['backend'] == 'text_math' else field_usage()),
        "precedence": "YAML < --set in order < explicit MATH selectors; declared deployment environment only",
    }


def record_command(root, command, *, dry_run=False):
    """Record argv after shell defaults and model preset expansion."""
    payload = {
        "schema_version": "harborrl-effective-command-v1",
        "command": command,
        "mode": "dry_run" if dry_run else "training",
    }
    target = Path(root) / "effective-command.json"
    if dry_run:
        checksum = hashlib.sha256(json.dumps(command).encode()).hexdigest()
        target = Path(root) / "dry-run-commands" / f"{checksum}.json"
    publish(target, payload)


if __name__ == "__main__":
    # Internal shell boundary: RUN_DIR, mode, then the exact executable argv.
    if len(sys.argv) < 4 or sys.argv[2] not in ("0", "1"):
        raise SystemExit("usage: evidence RUN_DIR DRY_RUN COMMAND [ARG ...]")
    record_command(sys.argv[1], sys.argv[3:], dry_run=sys.argv[2] == "1")
