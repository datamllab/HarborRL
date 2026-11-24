"""Supervise retention because pinned Slime silently ignores unknown CLI flags."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time
import zipfile


def complete_checkpoints(root, hf_template=None):
    """Inspect tracker-published generations; never consume or prune partial saves."""
    root = Path(root).resolve()
    try:
        latest = int((root / 'latest_checkpointed_iteration.txt').read_text().strip())
    except (OSError, ValueError):
        return []
    complete = []
    for path in root.glob('iter_*'):
        if not re.fullmatch(r'iter_[0-9]+', path.name) or path.is_symlink():
            continue
        step = int(path.name[5:])
        if step > latest:
            continue
        try:
            files = list(path.rglob('*'))
            if any(f.is_symlink() for f in files):
                continue
            if not (path / '.metadata').is_file() or not (path / 'common.pt').is_file():
                continue
            metadata = json.loads((path / 'metadata.json').read_text())
            if metadata.get('sharded_backend') != 'torch_dist':
                continue
            shards = list(path.glob('*.distcp'))
            if not shards:
                continue
            for shard in shards:
                with zipfile.ZipFile(shard) as archive:
                    if not archive.infolist():
                        raise ValueError('empty checkpoint archive')
            hf = None
            if hf_template:
                hf = Path(hf_template.format(rollout_id=step))
                if hf.is_symlink():
                    continue
                index = json.loads((hf / 'model.safetensors.index.json').read_text())
                for name in set(index['weight_map'].values()):
                    shard = hf / name
                    if shard.is_symlink() or not shard.resolve().is_relative_to(hf.resolve()):
                        raise ValueError('unsafe HF shard path')
                    with shard.open('rb') as stream:
                        size = int.from_bytes(stream.read(8), 'little')
                        if not 0 < size < 100_000_000:
                            raise ValueError('invalid safetensors header')
                        header = json.loads(stream.read(size))
                    end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
                    if shard.stat().st_size != 8 + size + end:
                        raise ValueError('incomplete HF shard')
            complete.append({'step': step, 'path': path, 'hf': hf,
                             'bytes': sum(f.stat().st_size for f in files if f.is_file())})
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
    return sorted(complete, key=lambda record: record['step'])


def prune(root, keep, *, hf_template=None, journal=None):
    if type(keep) is not int or keep < 1:
        raise ValueError('keep must be a positive integer')
    root = Path(root).resolve()
    complete = complete_checkpoints(root, hf_template)
    removed = []
    for checkpoint in complete[:-keep]:
        for path in (checkpoint['path'], checkpoint['hf']):
            if path is None:
                continue
            parent = root if path == checkpoint['path'] else Path(hf_template).parent.resolve()
            resolved = path.resolve()
            if path.is_symlink() or resolved == parent or not resolved.is_relative_to(parent):
                raise ValueError('unsafe checkpoint retention target')
            record = {'path': str(resolved), 'successor': complete[-1]['step'],
                      'files': [{'path': str(f.relative_to(resolved)), 'bytes': f.stat().st_size}
                                for f in resolved.rglob('*') if f.is_file()], 'time': time.time()}
            if journal:
                with Path(journal).open('a') as stream:
                    stream.write(json.dumps({**record, 'event': 'prune_planned'}) + chr(10))
            shutil.rmtree(resolved)
            removed.append(str(resolved))
            if journal:
                with Path(journal).open('a') as stream:
                    stream.write(json.dumps({'event': 'pruned', 'path': str(resolved)}) + chr(10))
    return removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-root', required=True, type=Path)
    parser.add_argument('--keep', type=int, default=2)
    parser.add_argument('--min-free-gb', type=float, default=128)
    parser.add_argument('--expected-gb', type=float, default=0)
    parser.add_argument('--margin', type=float, default=1.15)
    parser.add_argument('--hf-template')
    parser.add_argument('--resume-root', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or args.keep < 1 or args.min_free_gb < 0 or args.expected_gb < 0 or args.margin < 1
            or not all(math.isfinite(x) for x in (args.min_free_gb, args.expected_gb, args.margin))):
        parser.error('valid command, retention count and space limits required')
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    expected = args.expected_gb * 1024**3
    if args.resume_root:
        source = complete_checkpoints(args.resume_root)
        if (not source or source[-1]['step'] != int(
                (args.resume_root / 'latest_checkpointed_iteration.txt').read_text().strip())):
            raise RuntimeError('resume root contains no complete published checkpoint')
        expected = max(expected, source[-1]['bytes'])
    free = shutil.disk_usage(args.checkpoint_root).free
    required = max(args.min_free_gb * 1024**3, expected * args.margin)
    if free < required:
        raise RuntimeError(f'checkpoint preflight requires {required / 1024**3:.1f} GiB; free={free / 1024**3:.1f}')
    journal = args.checkpoint_root / 'retention.jsonl'
    journal.write_text(json.dumps({'event': 'start', 'keep': args.keep, 'free_bytes': free,
                                   'required_bytes': required, 'command': command}) + chr(10))
    child = subprocess.Popen(command, start_new_session=True)
    stopping = None

    def stop(signum, _frame):
        nonlocal stopping
        if child.poll() is None:
            stopping = stopping or time.monotonic()
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        while child.poll() is None:
            prune(args.checkpoint_root, args.keep, hf_template=args.hf_template, journal=journal)
            if stopping and time.monotonic() - stopping > 30:
                stop(signal.SIGKILL, None)
            time.sleep(2)
        prune(args.checkpoint_root, args.keep, hf_template=args.hf_template, journal=journal)
        return child.returncode
    finally:
        if child.poll() is None:
            stop(signal.SIGTERM, None)
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                stop(signal.SIGKILL, None)
                child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
