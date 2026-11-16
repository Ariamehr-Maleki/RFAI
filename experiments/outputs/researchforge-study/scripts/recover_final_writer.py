"""Audit committed final checkpoints before removing one confirmed-dead lock.

No experimental configuration, labels, rankings or parameters are changed.
Uncommitted files are retained; the frozen runner regenerates them on resume.
"""
import argparse
import json
from pathlib import Path
import psutil
from final_evaluation_gate import verify_lock
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'


def recover(expected_pid):
    lock = verify_lock(ROOT, STUDY / 'final/global-lock.json')
    folder = ROOT / lock['payload']['plan']['datasets']['prescience']['output_directory']
    writer = folder / 'writer.lock'
    owner = json.loads(writer.read_text())
    if owner != {'pid': expected_pid} or psutil.pid_exists(expected_pid):
        raise ValueError('Writer is not the explicitly expected dead process')
    blocked = {'prescience_final_run.py', 'run_frozen_study.py'}
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if any(Path(arg).name in blocked for arg in process.info['cmdline'] or []):
                raise RuntimeError('A final runner/supervisor is still active')
        except psutil.NoSuchProcess:
            continue
    identity = json.loads((folder / 'identity.json').read_text())
    if identity['lock_sha256'] != lock['lock_sha256']:
        raise ValueError('Checkpoint lock differs')
    query_dir = folder / 'queries'
    receipts = sorted(query_dir.glob('*.receipt'))
    checked = {}
    for receipt in receipts:
        committed = json.loads(receipt.read_text())
        if not committed:
            raise ValueError('Empty checkpoint receipt')
        for name, expected in committed.items():
            target = (query_dir / name).resolve()
            if not target.is_relative_to(query_dir.resolve()) or digest(target) != expected:
                raise ValueError('Committed checkpoint differs: ' + name)
        checked[receipt.name] = digest(receipt)
    progress = json.loads((folder / 'progress.json').read_text())
    if progress['completed_queries'] > len(receipts):
        raise ValueError('Missing committed progress')
    # Recheck immediately before removing only this particular stale lock.
    if psutil.pid_exists(expected_pid) or json.loads(writer.read_text()) != owner:
        raise ValueError('Writer ownership changed')
    record = dict(status='PASS committed final checkpoint audit; dead writer lock removed',
        expected_dead_pid=expected_pid, lock_sha256=lock['lock_sha256'], identity=identity,
        receipts=len(receipts), verified_receipt_hashes=checked, progress=progress,
        available_ram_bytes=psutil.virtual_memory().available,
        free_disk_bytes=psutil.disk_usage(str(ROOT)).free, recovery_script_sha256=digest(__file__))
    destination = STUDY / 'data-audit' / f'final-recovery-{expected_pid}.json'
    if destination.exists():
        raise ValueError('A recovery audit already exists for this writer')
    writer.unlink()
    atomic_state(destination, record)
    print(json.dumps({k:v for k,v in record.items() if k != 'verified_receipt_hashes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--expected-pid', type=int, required=True)
    recover(parser.parse_args().expected_pid)
