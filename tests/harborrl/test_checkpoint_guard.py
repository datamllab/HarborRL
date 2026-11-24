import json
from pathlib import Path
import subprocess
import sys
import zipfile

from harborrl.platform.checkpoint_guard import complete_checkpoints, prune


def checkpoint(root, step):
    path = root / f'iter_{step:07d}'
    path.mkdir(parents=True)
    (path / '.metadata').write_bytes(b'metadata')
    (path / 'common.pt').write_bytes(b'common')
    (path / 'metadata.json').write_text(json.dumps({'sharded_backend': 'torch_dist'}))
    with zipfile.ZipFile(path / '0.distcp', 'w') as stream:
        stream.writestr('tensor', b'data')
    (root / 'latest_checkpointed_iteration.txt').write_text(str(step))
    return path


def test_prune_waits_for_published_complete_successor(tmp_path):
    first = checkpoint(tmp_path, 0)
    second = checkpoint(tmp_path, 1)
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('0')
    assert prune(tmp_path, 1) == []
    (tmp_path / 'latest_checkpointed_iteration.txt').write_text('1')
    archive = (second / '0.distcp').read_bytes()
    (second / '0.distcp').write_bytes(archive[:20])
    assert prune(tmp_path, 1) == []
    assert first.exists()
    (second / '0.distcp').write_bytes(archive)
    journal = tmp_path / 'retention.jsonl'
    assert prune(tmp_path, 1, journal=journal) == [str(first)]
    assert second.exists()
    assert [json.loads(line)['event'] for line in journal.read_text().splitlines()] == ['prune_planned', 'pruned']


def test_prune_preserves_checkpoint_until_hf_export_complete(tmp_path):
    root = tmp_path / 'checkpoints'
    first = checkpoint(root, 0)
    second = checkpoint(root, 1)
    template = str(tmp_path / 'hf' / 'rollout-{rollout_id}')
    for step in (0, 1):
        hf = Path(template.format(rollout_id=step))
        hf.mkdir(parents=True)
        (hf / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'w': 'model.safetensors'}}))
        header = json.dumps({'w': {'data_offsets': [0, 4]}}).encode()
        (hf / 'model.safetensors').write_bytes(len(header).to_bytes(8, 'little') + header + b'data')
    tail = Path(template.format(rollout_id=1)) / 'model.safetensors'
    good = tail.read_bytes()
    tail.write_bytes(good[:-1])
    assert prune(root, 1, hf_template=template) == []
    assert first.exists()
    tail.write_bytes(good)
    assert len(prune(root, 1, hf_template=template)) == 2
    assert not first.exists() and second.exists() and tail.exists()


def test_symlink_generation_is_never_pruned(tmp_path):
    outside = tmp_path / 'outside'
    external = checkpoint(outside, 0)
    root = tmp_path / 'root'
    checkpoint(root, 1)
    (root / 'iter_0000000').symlink_to(external, target_is_directory=True)
    assert [record['step'] for record in complete_checkpoints(root)] == [1]
    assert prune(root, 1) == []
    assert external.exists()


def test_supervisor_returns_child_status_and_checks_space_before_launch(tmp_path):
    prefix = [sys.executable, '-m', 'harborrl.platform.checkpoint_guard',
              '--checkpoint-root', str(tmp_path / 'cp')]
    result = subprocess.run(prefix + ['--min-free-gb', '0', '--', sys.executable, '-c', 'raise SystemExit(7)'],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 7, result.stderr
    marker = tmp_path / 'should-not-exist'
    result = subprocess.run(prefix + ['--min-free-gb', '1000000000', '--', sys.executable, '-c',
        f'from pathlib import Path; Path({str(marker)!r}).touch()'], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'checkpoint preflight requires' in result.stderr
    assert not marker.exists()


def test_resume_refuses_corrupt_latest_even_with_valid_older_checkpoint(tmp_path):
    root = tmp_path / 'source'
    checkpoint(root, 0)
    second = checkpoint(root, 1)
    (second / '0.distcp').write_bytes(b'partial')
    result = subprocess.run([sys.executable, '-m', 'harborrl.platform.checkpoint_guard',
        '--checkpoint-root', str(tmp_path / 'target'), '--resume-root', str(root),
        '--min-free-gb', '0', '--', sys.executable, '-c', 'raise SystemExit(0)'],
        capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and 'no complete published checkpoint' in result.stderr
